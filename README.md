# SIGIR25-LiveRAG-CIIR-ScaledRAG

This repository contains our system submission to the [SIGIR’25 LiveRAG Challenge](https://liverag.tii.ae/), which focuses on building scalable, faithful, and responsive Retrieval-Augmented Generation (RAG) pipelines at web scale. The competition requires participants to design agent-based systems that operate over 15M+ long-context documents to answer open-domain questions grounded in real-world data.

Our pipeline mainly consists of two components: the Retriever and Multi-agent inference, which includes multiple agents that facilitate the environment (i.e., the Falcon LLM) with the necessary supporting passages to generate responses that are relevant and faithful. Our Multi-agent inference setup includes the following agents: Coordinator, Generator (uses Falcon LLM), Planner, Reasoner, Searcher (uses Retriever), Summarizer, and Validator. The setup is iterative and is trained on an extensive 10K-question dataset across a variety of categories generated using DataMorgana.

## Retriever
In order to setup our retriever server, follow the steps:

Create a virtual environment in python/3.11.7 using

```shell
pip install -r requirements.txt
```

retriever-setup/fine_web_chunking.py: Create document chunks of size 512 tokens and 80 token overlap, which increases the number of documents from ~14 Million to ~29 Million to prepare for Indexing.

Indexing the FineWeb dataset:

We follow prior methodologies from our group and utilize the "hzeng/Lion-SP-1B-llama3-marco-mntp," a 1-billion parameter sparse retrieval model introduced in [Scaling Sparse and Dense Retrieval in Decoder-Only LLMs](https://arxiv.org/abs/2502.15526) for indexing the Fineweb dataset. To index the dataset, you can use the following script:

```shell
# Activate virtual environment
source "/*address to the bin directory of the venv*/"

task_name="indexing"  
corpus_path="/*path to chunked fineweb dataset*/"
index_dir="/*path to the generated index*/"

list_model_name_paths=(
    hzeng/Lion-SP-1B-llama3-marco-mntp
)

if [[ ! -f "$corpus_path" ]]; then
    echo "Error: Corpus file not found at $corpus_path"
    exit 1
fi

if [[ ! -d "$index_dir" ]]; then
    echo "Creating index directory: $index_dir"
    mkdir -p "$index_dir"
fi

for model_name_or_path in "${list_model_name_paths[@]}"; do
    echo "Starting indexing with model: $model_name_or_path"

    if [[ ! -f eval_sparse.py ]]; then
        echo "Error: eval_sparse.py not found!"
        exit 1
    fi
    echo "Found eval_sparse.py, proceeding..."

    echo "Running torchrun..."
    set -x  # Enable debugging

    export CUDA_LAUNCH_BLOCKING=1
    export OMP_NUM_THREADS=8 

    torchrun --nproc_per_node=1 --master_port=4432 --standalone eval_sparse.py \
    --model_name_or_path "$model_name_or_path" \
    --index_dir "$index_dir" \
    --task_name "$task_name" \
    --eval_batch_size 64 \
    --doc_max_length 512 \
    --corpus_path "$corpus_path" \
    --data_source fineweb \
    > torchrun_output_8b.log 2>&1

    set +x  # Disable debugging

    echo "Indexing completed for $model_name_or_path"
done

echo "All indexing tasks finished."
```

Note: While creating the index and ease of inference, we used simplified sequential IDs such as [1, 2, ...], hence we create a mapping between these sequential IDs and the original Fineweb-IDs

our_id_to_fineweb_id_addr: retriever-setup/fine_web_index_doc_mapping.py

## Running Servers

In order to run our code, you need to run a set of servers for different models for efficient execution. One for retriever, one for the agent, and one for the environment that is Falcon.

### Retriever server

In order to run the Retriever server, you can use the following script:

```shell
PORT=8000
ADDRESS="/*the host address that the server is running on*/"
module load python/3.11.7
source "/*address to the bin directory of the venv*/"
echo "$ADDRESS" > "$DETAILS"
echo "$PORT" >> "$DETAILS"
export PYTHONPATH=retriever-setup
uvicorn sparse_retrieval_cpu:app --host 0.0.0.0 --port "$PORT"
```

/search endpoint for retrieval

### Agent server (Qwen2.5 trained model 7b)

First, you can download the trained agent from this [link](https://drive.google.com/file/d/18hW-oUt82691PXJ7mGSiyN_NShxLo0Yd/view?usp=sharing). Then, in order to run the agent server, you can use the following script:

```shell
# agent address
AGENT_ADDR="/*address to the unzipped agent checkpoint*/"

# port number
PORT=$((5000 + RANDOM % 200))
ADDRESS="/*the host address that the server is running on*/"

# Define log file path
LOGS="/*address to the log file*/"
DOWNLOAD_DIR="/*address to the cache directory*/"

# Activate virtual environment
source "/*address to the bin directory of the venv*/"

# Log hostname and port
echo "$ADDRESS" > "$LOGS"
echo "$PORT" >> "$DETAILS"

# Start vllm server
vllm serve "$AGENT_ADDR" --host 0.0.0.0 --port "$PORT" --download-dir "$DOWNLOAD_DIR"
```

### Environment server (Falcon instruct 10b)

In order to run the Falcon server for response generation, you can use the following script:

```shell
# agent address
AGENT_ADDR="tiiuae/Falcon3-10B-Instruct"

# port number
PORT=$((5000 + RANDOM % 200))
ADDRESS="/*the host address that the server is running on*/"

# Define log file path
LOGS="/*address to the log file*/"
DOWNLOAD_DIR="/*address to the cache directory*/"

# Activate virtual environment
source "/*address to the bin directory of the venv*/"

# Log hostname and port
echo "$ADDRESS" > "$LOGS"
echo "$PORT" >> "$DETAILS"

# Start vllm server
vllm serve "$AGENT_ADDR" --host 0.0.0.0 --port "$PORT" --download-dir "$DOWNLOAD_DIR"
```

### Reward Model server (Only for training)

In order to generate data for training the agent, we need access to a reward model for scoring different trajectories. To run this server, you can use the following script:

```shell
# agent address
AGENT_ADDR="Qwen/Qwen2.5-14B-Instruct"

# port number
PORT=$((5000 + RANDOM % 200))
ADDRESS="/*the host address that the server is running on*/"

# Define log file path
LOGS="/*address to the log file*/"
DOWNLOAD_DIR="/*address to the cache directory*/"

# Activate virtual environment
source "/*address to the bin directory of the venv*/"

# Log hostname and port
echo "$ADDRESS" > "$LOGS"
echo "$PORT" >> "$DETAILS"

# Start vllm server
vllm serve "$AGENT_ADDR" --host 0.0.0.0 --port "$PORT" --download-dir "$DOWNLOAD_DIR"
```

## Configurating the system

In order to connect different servers to each other and excecute the codes, we need to config the system. This config is ```inference-and-training/configs/default.py``` file. The default values we used for the competition is there and all you need to change is the addresses to different files. Basically, the following fields:

```python
DEFAULT_CONFIG = {
    "download_path": "/* address to where you want to download the files*/",
    "agent_model": "/* the address to the trained agent we shared with you*/",
    ...,
    "environment_model_server_log_file": "/* address to where the environment model server log file is saved*/",
    ...,
    "agent_model_server_log_file": "/* address to where the agent model server log file is saved*/",
    ...,
    "retriever_log_file": "/* address to where the retriever log file is saved*/",
    "index_addr": "/* address to where the index file is saved*/",
    ...
}
```

During training, you need the config file for the reward model as well, which again you need to set addresses:

```python
DEFAULT_CONFIG_EVALUATION = {
    "download_path": "/* address to where you want to download the files*/",
    ...,
    "judge_model_server_log_file": "/* address to where the judge model server log file is saved*/",
    ...
}
```

The system then automatically can connect to each of them and user them.

## Training

Training the agent (only the qwen model, falcon remains untrained) is a three step process:

### Step 1: generating data from DataMorgana

The first step is to generate data using DataMorgana:

```shell
source "/*address to the bin directory of the venv*/"
python inference-and-training/datamorgana/generate_question.py \
    --output_address "\*address to the directory to save the generated questions*\"
```

then, you can use the following script to divide them into training and testing files:

```shell
source "/*address to the bin directory of the venv*/"
python inference-and-training/utils/train_test_seperate.py \
    --datamorgana_data_addr "\*address to the directory where outputs from previous script is saved*\" \
    --output_dir "directory to save the train and test files"
```

### Step 2: Experience generation

The first step involves using the initial parameters of the agent to generate some experiences and score them using reward model. This step is happening using the following script:

```shell
source "/*address to the bin directory of the venv*/"
python inference-and-training/batch_training_generation_for_agent_self_training.py \
    --queries_addr "/*the address to the train file genrated in the previous step*/" \
    --output_addr "/*address to file to save experiences*/" \
    --num_samples 8 \
    --max_workers 16 \
    --sampling_temperature 0.7 \
    --num_shards 1 \
    --shard_id 0 \
    --no_concise \
```

Then, you can use the following script to filter the good experiences and use them to create training data:

```shell
source "/*address to the bin directory of the venv*/"
python inference-and-training/utils/extract_training_data_for_self_training.py \
    --input_addr "/*address to file with the saved experiences from previous script*/" \
    --output_addr "/*address to saving training data for agent*/"
```

### Step 3: training the agent

Finally, it is time to train the model (you need cuda 12.6):

```shell

source "/*address to the bin directory of the venv*/"
python inference-and-training/train_unsloth.py \
    --inputs_addr "/*address to saving training data for agent*/" \
    --model_addr "unsloth/Qwen2.5-7B-Instruct" \
    --output_dir "/*address to where to store checkpoints*/" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --learning_rate 0.0001 \
    --weight_decay 0.0 \
    --max_steps 5000 \
    --save_steps 100 \
    --warmup_steps 50 \
    --max_seq_length 16000 \
```

After this step, you need to rerun the agent server with the trained agent!

## Inference

For this step, all the servers but the reward model sever needs to be up and fully loaded (the retriever server takes around 2 hours to fully load the index). Then, using the following script, you can run the agent on test questions in the competition format (jsonl file):

```shell
source "/*address to the bin directory of the venv*/"
python inference-and-training/batch_agent_test_day.py \
    --queries_addr "/*address to the queries in jsonl format*/" \
    --output_addr "/*address to the response file to be saved*/" \
    --our_id_to_fineweb_id_addr "/*address to the file for mapping our ids to Fineweb ids (explained earlier)*/" \
    --max_workers 32 \
    --num_config 1 \
    --no_concise \
    --agent_name "/*the address to the trained agent model*/" \
```
