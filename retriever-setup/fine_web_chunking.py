from datasets import load_dataset
from transformers import AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter
import json

# File paths
input_dataset_path = "/*path to where the raw data provided is saved*/"
output_dataset_path = "/*path to chunked fineweb dataset*/"

# Load dataset as a stream
dataset = load_dataset("json", data_files=input_dataset_path, split="train", streaming=True)

# Load tokenizer and sentence splitter
model_name = "hzeng/Lion-SP-1B-llama3-marco-mntp"
tokenizer = AutoTokenizer.from_pretrained(model_name)
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)

# Processing variables
doc_count = 0
buffer = []  # Store documents before writing
BUFFER_SIZE = 10000  # Write every 10k documents

with open(output_dataset_path, "w") as f_out:
    for data in dataset:
        doc_id = data["id"]
        text = data["contents"]

        # Sentence-based chunking
        chunks = splitter.split_text(text)

        # Batch tokenization
        tokenized_outputs = tokenizer(chunks, truncation=True, return_tensors="pt", padding=True)

        for i, (chunk, input_ids) in enumerate(zip(chunks, tokenized_outputs["input_ids"])):
            chunk_length = input_ids.shape[0]  

            if chunk_length <= 512:
                chunk_obj = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}-{i}",
                    "contents": chunk
                }
                buffer.append(json.dumps(chunk_obj) + "\n")

        doc_count += 1

        # Ensure progress + writing every 10,000 docs
        if doc_count % BUFFER_SIZE == 0:
            f_out.writelines(buffer)  # Write buffered data to file
            f_out.flush()  # Force write to disk
            buffer = []  # Reset buffer
            print(f"✅ Processed {doc_count} documents. Flushed to file.")

    # Final flush to make sure no data is lost
    if buffer:
        f_out.writelines(buffer)
        f_out.flush()
        print(f"✅ Final flush: Saved remaining {len(buffer)} chunks.")

print(f"✅ All documents saved to {output_dataset_path}")
