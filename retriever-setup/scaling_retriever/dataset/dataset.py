from torch.utils.data import Dataset
import json

def get_doc_text(title, text):
        if title is None:
            return text
        else:
            return f"title: {title} | context: {text}"     

class CollectionDataset(Dataset):
    def __init__(self, corpus_path, data_source=None):
        if data_source == "fineweb": 
            self.pid_to_doc = self.read_fineweb_corpus(corpus_path)
        else:
            raise NotImplementedError(f"Unknown data source: {data_source}")

        self.pids = list(self.pid_to_doc.keys())

    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        text = self.pid_to_doc[pid]  # For FineWeb, it's already processed
        return pid, text

    def read_fineweb_corpus(self, corpus_path):
        """ Custom function to read FineWeb JSONL format """
        pid_to_doc = {}
        with open(corpus_path, "r") as f:
            for line in f:
                data = json.loads(line)
                pid_to_doc[data["chunk_id"]] = data["contents"]  
        return pid_to_doc
    
    