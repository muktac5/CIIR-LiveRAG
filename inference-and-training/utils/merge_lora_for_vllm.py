from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, FastModel


cache_dir = "/gypsum/work1/zamani/asalemi/RAG_VS_LoRA_Personalization/cache"
lora_addr = "/work/pi_hzamani_umass_edu/REML/liverag/multi-agent-live-rag/checkpoints/qwen_round1_unsloth_no_concise_new/checkpoint-2200"
full_model_addr = "/work/pi_hzamani_umass_edu/REML/liverag/multi-agent-live-rag/checkpoints/qwen_round1_unsloth_no_concise_new/checkpoint-2200/merged"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_addr,
    max_seq_length = 32000,
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    cache_dir = cache_dir,
)

model.save_pretrained_merged(full_model_addr, tokenizer, save_method = "merged_16bit")