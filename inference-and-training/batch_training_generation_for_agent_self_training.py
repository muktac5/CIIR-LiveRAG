from agents.coordinator.agent import generate_response
from configs.default import DEFAULT_CONFIG, DEFAULT_CONFIG_2, DEFAULT_CONFIG_EVALUATION, DEFAULT_CONFIG_EVALUATION_2
from evaluation.metric import metric
from evaluation.metric_coverage import metric_coverage
import json
import pandas as pd
import argparse
import concurrent.futures
import copy
import tqdm
from utils.general import batchify
import json


def score_generation(data, eval_config):
    try:
        question = data["question"]
        generated_output = str(data["response"]['response']) if data["response"] is not None else str(data["response"])
        ground_truth = data["ground_truth"]
        context = data["response"]["verified_documents"]
        metric_output = metric(question, generated_output, ground_truth, context, eval_config)
        data["metrics"] = {"metric": metric_output, "success": True}
        metric_output_coverage = metric_coverage(question, generated_output, ground_truth, context, eval_config)
        data["metrics_coverage"] = {"metric": metric_output_coverage, "success": True}
    except Exception as e:
        data["metrics"] = {"metric": {}, "success": False}
        data["metrics"]["error"] = str(e)
        data["metrics_coverage"] = {"metric": {}, "success": False}
        data["metrics_coverage"]["error"] = str(e)
    return data

def run_agent(query, answer, config, eval_config):
    """
    Run the agent with the given query and configuration.
    """
    try:
        response = generate_response(query, config)
        obj = {
            "question": query,
            "response": response,
            "ground_truth": answer,
            "success": True
        }
        obj = score_generation(obj, eval_config)
        return obj
    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        return {
            "question": query,
            "response": None,
            "ground_truth": answer,
            "success": False,
            "error": str(e)
        }
    
def save_results(results, output_addr):
    outputs_final = {i:result for i, result in enumerate(results)}
    with open(output_addr, "w") as f:
        json.dump(outputs_final, f, indent=4)

parser = argparse.ArgumentParser(description="Run the batch agent")
parser.add_argument("--queries_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to process")
parser.add_argument("--sampling_temperature", type=float, default=0.7, help="Temperature for sampling")
parser.add_argument("--max_workers", type=int, default=8, help="Number of threads to use for parallel processing")
parser.add_argument("--csv", action="store_true", help="Whether the input file is a CSV file")
parser.add_argument("--save_every_n", type=int, default=5, help="Save every n samples")
parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the data into")
parser.add_argument("--shard_id", type=int, default=0, help="Shard ID to process")
parser.add_argument("--no_concise", action="store_true", help="Whether to use concise mode for the agent")


if __name__ == "__main__":
    args = parser.parse_args()
    queries_addr = args.queries_addr

    # Load the queries from the CSV file

    if args.csv:
        df = pd.read_csv(queries_addr)
        queries = df["Question"].tolist()
        if "Answer" in df.columns:
            ground_truth = df["Answer"].tolist()
        else:
            ground_truth = [""] * len(queries)
    else:
        with open(queries_addr, "r") as f:
            data = json.load(f)
            queries = [item["question"] for item in data]
            ground_truth = [item["answer"] for item in data]
    
    # Split the data into shards
    if args.num_shards > 1:
        shard_size = len(queries) // args.num_shards
        start_index = args.shard_id * shard_size
        end_index = (args.shard_id + 1) * shard_size if args.shard_id < args.num_shards - 1 else len(queries)
        queries = queries[start_index:end_index]
        ground_truth = ground_truth[start_index:end_index]
    if args.shard_id % 2 == 0:
        config = copy.deepcopy(DEFAULT_CONFIG)
        eval_config = copy.deepcopy(DEFAULT_CONFIG_EVALUATION)
    else:
        config = copy.deepcopy(DEFAULT_CONFIG_2)
        eval_config = copy.deepcopy(DEFAULT_CONFIG_EVALUATION_2)
    config['temperature_agent'] = args.sampling_temperature
    config['num_samples_for_training'] = args.num_samples
    if args.no_concise:
        config['concise'] = False
        eval_config['concise'] = False
    batch_size = args.max_workers // args.num_samples
    results = []
    save_counter = 0
    output_addr = args.output_addr + "_" + str(args.shard_id)
    for batch_data, batch_ground_truth in tqdm.tqdm(zip(batchify(queries, batch_size), batchify(ground_truth, batch_size)), total=len(queries)//batch_size):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            features = []
            for query, answer in zip(batch_data, batch_ground_truth):
                sub_futures = []
                for i in range(args.num_samples):
                    sub_futures.append(executor.submit(run_agent, query, answer, config, eval_config))
                features.append(sub_futures)
            for future_list in features:
                query_outputs = []
                for future in future_list:
                    output = future.result()
                    query_outputs.append(output)
                results.append(query_outputs)
        save_counter += 1
        if save_counter % args.save_every_n == 0:
            save_results(results, output_addr)
            print(f"Saved intermediate results to {output_addr}")
    save_results(results, output_addr)
    print(f"Final results saved to {output_addr}")