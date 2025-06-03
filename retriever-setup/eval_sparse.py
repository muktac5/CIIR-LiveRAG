import os
import torch.distributed
import ujson 
from dataclasses import field, dataclass
from transformers import AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
import torch
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import hf_hub_download

from scaling_retriever.dataset.dataset import CollectionDataset
from scaling_retriever.dataset.data_collator import LlamaSparseCollectionCollator
from scaling_retriever.modeling.llm_encoder import LlamaBiSparse
from scaling_retriever.indexer import SparseIndexer
import constants

def ddp_setup(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.world_size = torch.distributed.get_world_size()
        print(f"Initialized DDP on Rank {args.local_rank}, World Size: {args.world_size}")
    else:
        args.local_rank = 0
        args.world_size = 1  
        print("Running in single GPU mode (No DDP).")

@dataclass
class SparseArguments:
    model_name_or_path: str = field(default=None)
    corpus_path: str = field(default="")
    index_dir: str = field(default=None)
    data_source: str = field(default="fineweb")
    eval_batch_size: int = field(default=128)
    doc_max_length: int = field(default=192)
    task_name: str = field(default="indexing")     
        
def sparse_index(args, model_type):
    ddp_setup(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    d_collection = CollectionDataset(corpus_path=args.corpus_path, 
                                         data_source=constants.corpus_datasource[args.corpus_path])

    if model_type == "llama":
        print("eval_sparse", args.model_name_or_path)
        model = LlamaBiSparse.load_from_lora(args.model_name_or_path).to("cuda")
        d_collator = LlamaSparseCollectionCollator(tokenizer=tokenizer, max_length=args.doc_max_length)

    # Only use DistributedSampler if running distributed training
    if args.world_size > 1:
        sampler = DistributedSampler(d_collection, shuffle=False)
    else:
        sampler = None

    d_loader = DataLoader(d_collection, batch_size=args.eval_batch_size, shuffle=(sampler is None),
                          collate_fn=d_collator, num_workers=2, sampler=sampler)

    if args.world_size > 1:
        index_dir = args.index_dir.rstrip("/")
        index_dir = f"{index_dir}_{torch.distributed.get_rank()}"
    else:
        index_dir = args.index_dir

    print(index_dir, args.local_rank, model.vocab_size)
    indexer = SparseIndexer(model, index_dir=index_dir, compute_stats=True, dim_voc=model.vocab_size,
                            device="cuda" if torch.cuda.is_available() else "cpu")
    indexer.index(d_loader)

def init_main():
    parser = HfArgumentParser(SparseArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    if os.path.isdir(args.model_name_or_path): 
        with open(os.path.join(args.model_name_or_path, "config.json"), "r") as f:
            model_config = ujson.load(f)
    else:
        config_path = hf_hub_download(args.model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            model_config = ujson.load(f)

    model_type = model_config["model_type"]
    assert model_type in constants.supported_models, model_type
    
    if args.task_name == "indexing":
        sparse_index(args, model_type)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    init_main()
