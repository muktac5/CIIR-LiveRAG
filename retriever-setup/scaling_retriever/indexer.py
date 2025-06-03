import os 
import pickle
import logging
from collections import defaultdict
import torch 
from tqdm import tqdm
from transformers.modeling_utils import unwrap_model
import numpy as np
import ujson

from scaling_retriever.utils.utils import is_first_worker, to_list,supports_bfloat16
from scaling_retriever.utils.inverted_index import IndexDictOfArray
from scaling_retriever.modeling.losses.regulariaztion import L0

logger = logging.getLogger()

def store_embs(model, collection_loader, local_rank, index_dir, device,
               chunk_size=2_000_000, use_fp16=False, is_query=False, idx_to_id=None):
    """Encodes and stores document embeddings on GPU."""
    
    write_freq = chunk_size // collection_loader.batch_size
    if is_first_worker():
        print(f"write_freq: {write_freq}, batch_size: {collection_loader.batch_size}, chunk_size: {chunk_size}")

    dtype = torch.bfloat16 if supports_bfloat16() else torch.float32
    print(f"Using {'bfloat16' if dtype == torch.bfloat16 else 'float32'}")

    embeddings = []
    embeddings_ids = []
    chunk_idx = 0
    model.to(device)

    for idx, batch in tqdm(enumerate(collection_loader), disable=not is_first_worker(),
                           desc=f"Encoding {len(collection_loader)} sequences", total=len(collection_loader)):
        with torch.inference_mode():
            with torch.cuda.amp.autocast():  
                inputs = {k: v.to(device) for k, v in batch.items() if k != "ids"}
                if is_query:
                    raise NotImplementedError 
                else:
                    reps = unwrap_model(model).doc_encode(**inputs)  
                
                text_ids = batch["ids"]

        embeddings.append(reps)  
        embeddings_ids.extend(text_ids)

        if (idx + 1) % write_freq == 0:
            embeddings = torch.cat(embeddings).float().cpu().numpy()
            if isinstance(embeddings_ids[0], int):
                embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
            assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))

            text_path = os.path.join(index_dir, f"embs_{local_rank}_{chunk_idx}.npy")
            id_path = os.path.join(index_dir, f"ids_{local_rank}_{chunk_idx}.npy")
            np.save(text_path, embeddings)
            np.save(id_path, embeddings_ids)

            del embeddings, embeddings_ids
            torch.cuda.empty_cache() 
            embeddings, embeddings_ids = [], []
            chunk_idx += 1 

    if len(embeddings) != 0:
        embeddings = torch.cat(embeddings).float().cpu().numpy()
        if isinstance(embeddings_ids[0], int):
            embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
        assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))

        print(f"Last embeddings shape = {embeddings.shape}")
        text_path = os.path.join(index_dir, f"embs_{local_rank}_{chunk_idx}.npy")
        id_path = os.path.join(index_dir, f"ids_{local_rank}_{chunk_idx}.npy")
        np.save(text_path, embeddings)
        np.save(id_path, embeddings_ids)

        del embeddings, embeddings_ids
        torch.cuda.empty_cache()  
        chunk_idx += 1 

    plan = {"nranks": torch.distributed.get_world_size(),
            "num_chunks": chunk_idx,
            "index_path": os.path.join(index_dir, "model.index")}
    
    print("Indexing plan: ", plan)

    if is_first_worker():
        with open(os.path.join(index_dir, "plan.json"), "w") as fout:
            ujson.dump(plan, fout)

shared_doc_ids = None
shared_doc_values = None

def init_worker(doc_ids, doc_values):
    """Initialize shared index for multiprocessing"""
    global shared_doc_ids, shared_doc_values
    shared_doc_ids = torch.tensor(doc_ids, device="cuda")
    shared_doc_values = torch.tensor(doc_values, device="cuda")

def process_query_for_multiprocess(args):
    """Function to process a single query using GPU-optimized retrieval."""
    qid, col, values, threshold, topk, doc_ids, size_collection, numba_score_float, select_topk = args
    
    res = defaultdict(dict)
    stats = defaultdict(float)

    col = torch.tensor(col, device="cuda")
    values = torch.tensor(values, device="cuda")
    scores = torch.zeros(size_collection, dtype=torch.float32, device="cuda")
    
    retrieved_indexes = shared_doc_ids[col]
    retrieved_floats = shared_doc_values[col]
    scores.index_add_(0, retrieved_indexes.flatten(), (values * retrieved_floats).flatten())
    
    filtered_indexes = torch.nonzero(scores > threshold).squeeze(-1)
    scores = -scores[filtered_indexes]
    filtered_indexes, scores = select_topk(filtered_indexes, scores, k=topk)
    
    for id_, sc in zip(filtered_indexes.tolist(), scores.tolist()):
        res[str(qid)][str(doc_ids[id_])] = float(sc)
    stats["L0_q"] = len(values)
    
    return res, stats
        
class SparseIndexer:
    def __init__(self, model, index_dir, device, compute_stats=False, dim_voc=None, force_new=True,
                 filename="array_index.h5py", **kwargs):
        self.model = model
        self.model.eval()
        self.index_dir = index_dir
        
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new, filename=filename)
        self.compute_stats = compute_stats
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        
        if self.compute_stats:
            self.l0 = L0()
        
        self.model.to(self.device)
        
        self.local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        print(f"world_size: {self.world_size}, local_rank: {self.local_rank}")

    def index(self, collection_loader, id_dict=None):
        dtype = torch.bfloat16 if supports_bfloat16() else torch.float32
        print(f"Using {'bfloat16' if dtype == torch.bfloat16 else 'float32'}")

        doc_ids = {}
        stats = defaultdict(float) if self.compute_stats else None
        count = 0

        with torch.inference_mode():
            for t, batch in enumerate(tqdm(collection_loader, disable=not is_first_worker())):
                print(f"Batch {t}: {len(batch['ids'])} documents")
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"ids"}}  
                
                with torch.amp.autocast("cuda", dtype=dtype):  
                    batch_documents = self.model.encode(**inputs)  
                
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                
                row, col = torch.nonzero(batch_documents, as_tuple=True)
                data = batch_documents[row, col]

                row += count
                g_row = (row * self.world_size + self.local_rank).cpu().numpy() 
                
                batch_ids = to_list(batch["ids"]) if isinstance(batch["ids"], torch.Tensor) else batch["ids"]
                
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]

                unique_g_row = np.sort(np.unique(g_row))
                if len(unique_g_row) == len(batch_ids):
                    doc_ids.update({int(x): y for x, y in zip(unique_g_row, batch_ids)})
                else:
                    all_idxes = (count + np.arange(len(batch_ids))) * self.world_size + self.local_rank
                    for _i, _idx in enumerate(all_idxes):
                        if _idx in unique_g_row:
                            doc_ids[_idx] = batch_ids[_i]

                self.sparse_index.add_batch_document(
                    g_row, col.cpu().numpy(), data.float().cpu().numpy(), n_docs=len(batch_ids)
                )  
                count += len(batch_ids)  
                if count % 100000 < len(batch_ids):  
                    print(f"Documents indexed so far: {count}")


        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}

        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("Indexing complete.")
            print(f"Index contains {len(self.sparse_index)} posting lists")
            print(f"Index contains {len(doc_ids)} documents")
            if self.compute_stats:
                with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                    ujson.dump(stats, handler)
        else:
            for key in self.sparse_index.index_doc_id.keys():
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key], dtype=np.float32)

            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out
