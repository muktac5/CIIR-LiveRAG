from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from unsloth import FastLanguageModel, FastModel

import argparse
import os
import json
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

def load_dataset(addr, cache_dir):
    def gen():
        with open(addr, 'r') as f:
            dataset = json.load(f)
            for data in dataset:
                yield data
    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)

parser = argparse.ArgumentParser()

parser.add_argument("--inputs_addr", required=True)
parser.add_argument("--cache_dir", default="/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/cache")
parser.add_argument("--model_addr", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--per_device_train_batch_size", type=int, default=64)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--max_seq_length", type=int, default=32768)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = load_dataset(args.inputs_addr, cache_dir=args.cache_dir)

    model, tokenizer = FastModel.from_pretrained(
        model_name = args.model_addr,
        max_seq_length = args.max_seq_length, # Choose any for long context!
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False,
        cache_dir = args.cache_dir,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = args.max_seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        save_only_model=True,
        logging_steps=10,
        # optim="adamw_8bit",
        bf16=True,
        # fp16 = True
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )
    trainer.train()
