from agents.coordinator.agent import generate_response
from configs.default import DEFAULT_CONFIG_2, DEFAULT_CONFIG
import json
import pandas as pd
import argparse
import concurrent.futures
from utils.general import batchify
import tqdm
import json
import copy
import time
import os
from transformers import AutoTokenizer

def get_the_id_from_doc_id(doc_id, our_id_to_fineweb_id):
    if '-' in doc_id:
        doc_id = doc_id.split('-')[0]
    return our_id_to_fineweb_id[doc_id]

def convert_to_live_rag_format_and_save(results, addr, our_id_to_fineweb_id):
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-10B-Instruct")
    with open(addr + "_us", "w") as f:
        json.dump(results, f, indent=4)
    with open(addr, "w") as f:
        for result in results:
            output_obj = {
                "id": result["id"],
                "question": result["question"],
                "answer": result["response"]['response'] if result['success'] else "",
                "passages": [
                    {
                        "passage": document['text'],
                        "doc_IDs": [get_the_id_from_doc_id(document['doc_id'], our_id_to_fineweb_id)],
                    } for document in result['response']['verified_documents']
                ] if result['success'] else [],
                "final_prompt": tokenizer.apply_chat_template(result['response']['memory']['generator'][:-1], tokenize=False, add_generation_prompt=True) if result['success'] else "",
            }
            f.write(json.dumps(output_obj) + "\n")

def run_agent(query, answer, config):
    """
    Run the agent with the given query and configuration.
    """
    counter = 0
    errors = []
    config = copy.deepcopy(config)
    saved_result = None
    while counter < config['max_retries']:
        try:
            response = generate_response(query, config)
            response['memory']['generator']
            saved_result = response
            assert len(response['verified_documents']) > 0, "No documents found in the response."
            return {
                "question": query,
                "response": response,
                "ground_truth": answer,
                "success": True
            }
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            errors.append(str(e))
            counter += 1
            if config['temperature_agent'] < 1:
                config['temperature_agent'] += 0.1
            if config['temperature_environment'] < 1:
                config['temperature_environment'] += 0.1
    if saved_result:
        return {
            "question": query,
            "response": saved_result,
            "ground_truth": answer,
            "success": True,
            "error": errors
        }
    return {
        "question": query,
        "response": None,
        "ground_truth": answer,
        "success": False,
        "error": errors
    }

parser = argparse.ArgumentParser(description="Run the batch agent")
parser.add_argument("--queries_addr", type=str, required=True)
parser.add_argument("--our_id_to_fineweb_id_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--max_workers", type=int, default=8, help="Number of threads to use for parallel processing")
parser.add_argument("--num_config", default=1, type=int, help="Configuration to use for the agent (1 or 2)")
parser.add_argument("--agent_name", type=str, default="", help="Name of the agent to use")
parser.add_argument("--no_concise", action="store_true", help="Indicate if the agent should be concise")

if __name__ == "__main__":
    args = parser.parse_args()
    queries_addr = args.queries_addr

    # Load the queries from the CSV file
    with open(queries_addr, "r") as f:
        queries = []
        ids = []
        ground_truth = []
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries.append(data["question"])
                ids.append(data["id"])
                ground_truth.append("")
    with open(args.our_id_to_fineweb_id_addr, "r") as f:
        our_id_to_fineweb_id = json.load(f)


    results = []
    start_time = time.time()
    total_batches = len(queries) // args.max_workers + (1 if len(queries) % args.max_workers > 0 else 0)
    for batch_ids, batch_data, batch_ground_truth in tqdm.tqdm(zip(batchify(ids, args.max_workers), batchify(queries, args.max_workers), batchify(ground_truth, args.max_workers)), total=total_batches):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            counter = 0
            for query, answer in zip(batch_data, batch_ground_truth):
                if args.num_config == 1:
                    config = DEFAULT_CONFIG
                elif args.num_config == 2:
                    if counter % 2 == 0:
                        config = copy.deepcopy(DEFAULT_CONFIG)
                    else:
                        config = copy.deepcopy(DEFAULT_CONFIG_2)
                else:
                    raise ValueError("Invalid configuration number. Use 1 or 2.")
                if args.agent_name:
                    config["agent_model"] = args.agent_name
                if args.no_concise:
                    config["concise"] = False
                futures.append(executor.submit(run_agent, query, answer, config))
                counter += 1
            for id, result in zip(batch_ids, futures):
                output = result.result()
                output["id"] = id
                results.append(output)
        convert_to_live_rag_format_and_save(results, args.output_addr, our_id_to_fineweb_id)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")