from pyserini.search.lucene import LuceneSearcher
import json
from abc import ABC, abstractmethod
import requests
import random

class Retriever(ABC):

    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cached_top_k = {}
        self.id_mapping_original_id_to_new_id = dict()
        self.id_mapping_new_id_to_original_id = dict()
        self.doc_ids = set()
        self.max_id = 1000

    @abstractmethod
    def _search(self, query, top_k=5):
        pass

    def search_next(self, query, cache_k=20):
        if query in self.cache and len(self.cache[query]) > 0:
            # print(f"Cache has {len(self.cache[query])} documents for query: {query}")
            return self.cache[query].pop(0)

        # Get current top_k served; default to base top_k
        current_top_k = self.cached_top_k.get(query, self.default_top_k)
        new_top_k = current_top_k + cache_k
        # print(f"Fetching results {current_top_k} to {new_top_k}")

        results = self._search(query, new_top_k)
        if not results:
            # print(f"No results returned from _search for query: {query}")
            return None
        
        for result in results:
            if result['id'] in self.id_mapping_original_id_to_new_id:
                result['orriginal_id'] = result['id']
                result['id'] = self.id_mapping_original_id_to_new_id[result['id']]
            else:
                result['orriginal_id'] = result['id']
                while True:
                    new_id = random.randint(0, self.max_id)
                    if new_id not in self.doc_ids:
                        break
                new_id = str(new_id)
                self.id_mapping_original_id_to_new_id[result['id']] = new_id
                self.id_mapping_new_id_to_original_id[new_id] = result['id']
                result['id'] = new_id
                self.doc_ids.add(new_id)

        self.cache[query] = results
        self.cached_top_k[query] = new_top_k

        # print(f"Cache updated for query: {query} with {len(self.cache[query])} new documents.")
        return self.cache[query].pop(0) if self.cache[query] else None

class SparseRetriever(Retriever):
    def load_url_from_log_file(self, log_addr: str):
        with open(log_addr, "r") as f:
            lines = f.readlines()
        host = lines[0].strip()
        port = lines[1].strip()
        return f"http://{host}:{port}/search"

    def __init__(self, execute_config):
        super().__init__()
        self.default_top_k = execute_config['top_k']  # this is the base top_k
        self.search_url = self.load_url_from_log_file(execute_config['retriever_log_file'])
        self.threshold = execute_config['threshold']

    def _search(self, query, top_k):
        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": self.threshold
        }
        # print("payload:", payload)
        try:
            response = requests.post(self.search_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None
        except json.JSONDecodeError:
            # print("Failed to decode JSON response.")
            return None

class BM25Retriever(Retriever):
    def __init__(self, index_address):
        super().__init__()
        self.index_address = index_address
        self.index = LuceneSearcher(index_address)
    
    def _search(self, query, top_k=5):
        results = self.index.search(query, k=top_k)
        return [
            {
                "id": result.docid,
                "fineweb_id": json.loads(result.lucene_document.get('raw'))['fineweb-id'],
                "text": json.loads(result.lucene_document.get('raw'))['contents']
            }
            for result in results
        ]

if __name__ == "__main__":
    retriever = BM25Retriever("/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/fine_web_index/index")
    query = input("Enter your query: ")
    for i in range(5):
        results = retriever.search_next(query)
        print(results["text"])
        print('---------')
