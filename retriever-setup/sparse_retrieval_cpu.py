import os
import json
import torch
import pickle
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
import traceback
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np
import numba
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from scaling_retriever.dataset.data_collator import LlamaSparseCollectionCollator
from scaling_retriever.modeling.llm_encoder import LlamaBiSparse
from scaling_retriever.utils.inverted_index import IndexDictOfArray
from scaling_retriever.utils.utils import is_first_worker, to_list, supports_bfloat16
from collections import defaultdict

# ==================== LOGGING CONFIG ====================
LOG_FORMAT = "%(asctime)s — %(levelname)s — %(name)s — %(message)s"
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("retriever_server.log"),  
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger("retriever")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.0

class BatchQueryRequest(BaseModel):
    queries: List[str]
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.0

class SparseRetrieval:
    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        scores = np.array(scores)
        filtered_indexes = np.array(filtered_indexes)

        if len(filtered_indexes) > k:
            top_k_unsorted_idx = np.argpartition(-scores, k)[:k]
            top_scores = scores[top_k_unsorted_idx]
            top_indexes = filtered_indexes[top_k_unsorted_idx]
            sorted_top_k = np.argsort(-top_scores)

            return top_indexes[sorted_top_k], top_scores[sorted_top_k]  
        else:
            sorted_all = np.argsort(-scores)
            return filtered_indexes[sorted_all], scores[sorted_all]  


    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]
            query_float = query_values[_idx]
            retrieved_indexes = inverted_index_ids[local_idx]
            retrieved_floats = inverted_index_floats[local_idx]
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]
        return filtered_indexes, scores[filtered_indexes]

    def __init__(self, model, config, dim_voc, device, index_d=None, **kwargs):
        self.model = model
        self.model.eval()

        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = torch.tensor([], dtype=torch.int32)
                    self.sparse_index.index_doc_value[i] = torch.tensor([], dtype=torch.float32)

        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value

        self.doc_map = self.load_document_chunks(config.get("document_chunks_file", None))
        self.device = device
        self.model.to(device)

        logger.info("SparseRetrieval initialized")

    def load_document_chunks(self, file_path):
        doc_map = {}
        if file_path:
            with open(file_path, 'r') as f:
                for line in f:
                    chunk = json.loads(line)
                    doc_map[chunk["chunk_id"]] = chunk["contents"]
        logger.info(f"Loaded {len(doc_map)} document chunks")
        return doc_map

    def _generate_query_vecs(self, q_loader):
        sparse_query_vecs = []
        qids = []

        with torch.inference_mode():
            for t, batch in enumerate(tqdm(q_loader, total=len(q_loader), desc="generate query vecs", disable=not is_first_worker())):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"ids"}}
                with torch.amp.autocast(self.device, dtype=torch.bfloat16 if supports_bfloat16() else torch.float32):
                    batch_sparse_reps = self.model.encode(**inputs)
                qids.extend(batch["ids"] if isinstance(batch["ids"], list) else to_list(batch["ids"]))
                for sparse_rep in batch_sparse_reps:
                    sparse_rep = sparse_rep.unsqueeze(0)
                    row, col = torch.nonzero(sparse_rep, as_tuple=True)
                    data = sparse_rep[row, col]
                    sparse_query_vecs.append((col.cpu().numpy().astype(np.int32), data.cpu().numpy().astype(np.float32)))

        logger.info(f"Generated sparse vectors for {len(qids)} queries")
        return sparse_query_vecs, qids

    def _sparse_retrieve_cpu(self, sparse_query_vecs, qids, threshold=0., topk=5):
        def _process_query_threaded(qid, col, values):
            res = defaultdict(dict)
            stats = defaultdict(float)

            filtered_indexes, scores = self.numba_score_float(
                self.numba_index_doc_ids,
                self.numba_index_doc_values,
                col,
                values,
                threshold=threshold,
                size_collection=self.sparse_index.nb_docs(),
            )

            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=topk)

            for id_, sc in zip(filtered_indexes, scores):
                doc_id = str(self.doc_ids[id_])
                res[str(qid)][doc_id] = {
                    "score": float(sc),
                    "content": self.doc_map.get(doc_id, "")
                }

            stats["L0_q"] = len(values)
            return res, stats

        res = defaultdict(dict)
        stats = defaultdict(float)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_process_query_threaded, qid, col, values)
                for qid, (col, values) in zip(qids, sparse_query_vecs)
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="retrieval by inverted index", disable=not is_first_worker()
            ):
                r, s = future.result()
                for qid, docs in r.items():
                    res[qid].update(docs)
                for k, v in s.items():
                    stats[k] += v / len(qids)

        logger.info(f"Retrieved results for {len(qids)} queries")
        return res, stats

MODEL_PATH = "hzeng/Lion-SP-1B-llama3-marco-mntp"
INDEX_DIR = "/*path to the generated index*/"
CHUNK_FILE = "/*path to chunked fineweb dataset*/"

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LlamaBiSparse.load_from_lora(MODEL_PATH)
q_collator = LlamaSparseCollectionCollator(tokenizer=tokenizer, max_length=512)
config = {"index_dir": INDEX_DIR, "document_chunks_file": CHUNK_FILE}

retriever = SparseRetrieval(model=model, config=config, dim_voc=model.vocab_size, device=device)

@app.get("/")
def root():
    return {"message": "Sparse Retriever is live!"}

@app.post("/search")
def search(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        q_loader = DataLoader([(f"q0", request.query)], batch_size=1, shuffle=False, collate_fn=q_collator)
        results, _ = retriever._sparse_retrieve_cpu(*retriever._generate_query_vecs(q_loader), topk=request.top_k, threshold=request.threshold)

        response = []
        for qid, docs in results.items():
            for doc_id, result in docs.items():
                response.append({
                    "query_id": qid,
                    "id": doc_id,
                    "doc_id":doc_id,
                    "score": result["score"],
                    "text": result["content"]
                })

        if not response:
            logger.warning("No documents found for query.")
            raise HTTPException(status_code=404, detail="No documents found matching the query.")

        logger.info(f"Returning {len(response)} results for query q0")
        return response

    except Exception as e:
        logger.error("Error occurred in /search endpoint:")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_search")
def batch_search(request: BatchQueryRequest):
    try:
        logger.info(f"Received batch of {len(request.queries)} queries")
        batch = [(f"q{i}", query) for i, query in enumerate(request.queries)]
        q_loader = DataLoader(batch, batch_size=8, shuffle=False, collate_fn=q_collator)
        sparse_query_vecs, qids = retriever._generate_query_vecs(q_loader)
        results, _ = retriever._sparse_retrieve_cpu(
            sparse_query_vecs, qids, topk=request.top_k, threshold=request.threshold
        )

        response = []
        for qid, docs in results.items():
            for doc_id, result in docs.items():
                response.append({
                    "query_id": qid,
                    "id": doc_id,
                    "doc_id":doc_id,
                    "score": result["score"],
                    "text": result["content"]
                })

        if not response:
            logger.warning("No documents found for any of the queries.")
            raise HTTPException(status_code=404, detail="No documents found matching any of the queries.")

        logger.info(f"Returning {len(response)} total results across batch queries")
        return response

    except Exception as e:
        logger.error("Error occurred in /batch_search endpoint:")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
