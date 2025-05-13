import json
from collections import Counter
from transformers import AutoTokenizer
import argparse

def load_logs(addr):
    with open(addr, 'r') as f:
        logs = json.load(f)
    values = list(logs.values())
    for value in values:
        for path in value:
            if path['success'] and path['metrics']['success'] and path['metrics_coverage']['success']:
                path['metrics']['metric']['relevant_score']['score_equivalence_normalized'] = path['metrics_coverage']['metric']['score_normalized']
                path['metrics']['metric']['relevant_score']['score_equivalence'] = path['metrics_coverage']['metric']['score']
            
    return logs.values()

def get_extract_scores(scores):
    score = {
        "relevance": scores['metric']['relevant_score']['score_relevance_normalized'],
        "equivalence": scores['metric']['relevant_score']['score_equivalence_normalized'],
        "faithfulness": scores['metric']['faithful_score']['scor_faithfulness_normalized'],
    }
    return score

def find_score_distribution_for_example(scores):
    dist_score = {
        "relevance": [],
        "equivalence": [],
        "faithfulness": [],
    }
    for score in scores:
        score = get_extract_scores(score)
        dist_score['relevance'].append(score['relevance'])
        dist_score['equivalence'].append(score['equivalence'])
        dist_score['faithfulness'].append(score['faithfulness'])
    return {
        "relevance": Counter(dist_score['relevance']),
        "equivalence": Counter(dist_score['equivalence']),
        "faithfulness": Counter(dist_score['faithfulness']),
    }

def find_average_score_distribution_for_example(scores):
    dist_score = {
        "relevance": [],
        "equivalence": [],
        "faithfulness": [],
    }
    for score in scores:
        score = get_extract_scores(score)
        dist_score['relevance'].append(score['relevance'])
        dist_score['equivalence'].append(score['equivalence'])
        dist_score['faithfulness'].append(score['faithfulness'])
    return {
        "relevance": sum(dist_score['relevance']) / len(dist_score['relevance']),
        "equivalence": sum(dist_score['equivalence']) / len(dist_score['equivalence']),
        "faithfulness": sum(dist_score['faithfulness']) / len(dist_score['faithfulness']),
    }

def get_combined_reward_score(score):
    score = get_extract_scores(score)
    return (score['relevance'] + 4 * score['equivalence'] + score['faithfulness']) / 6

def collect_all_path_scores(example):
    all_scores = []
    for path in example:
        if not path['success'] or not path['metrics']['success']:
            continue
        all_scores.append(path['metrics'])
    return all_scores

def filter_out_example_based_on_reward(scores):
    rewards = []
    for score in scores:
        reward = get_combined_reward_score(score)
        rewards.append(reward)
    if len(set(rewards)) == 1:
        return True
    no_score_above_0_5_equivalence = True
    no_score_above_0_5_relevance = True
    no_score_above_0_5_faithfulness = True
    for score in scores:
        if score['metric']['relevant_score']['score_equivalence_normalized'] > 0.5:
            no_score_above_0_5_equivalence = False
        if score['metric']['relevant_score']['score_relevance_normalized'] > 0.5:
            no_score_above_0_5_relevance = False
        if score['metric']['faithful_score']['scor_faithfulness_normalized'] > 0.5:
            no_score_above_0_5_faithfulness = False
    if no_score_above_0_5_equivalence or no_score_above_0_5_relevance or no_score_above_0_5_faithfulness:
        return True
    return False

def get_all_rewards_for_example(scores):
    rewards = []
    for score in scores:
        reward = get_combined_reward_score(score)
        rewards.append(reward)
    return rewards

def get_examples_with_this_reward(data, reward):
    examples = []
    for path in data:
        if not path['success'] or not path['metrics']['success']:
            continue
        reward_score = get_combined_reward_score(path['metrics'])
        if reward_score == reward:
            examples.append(path)
    return examples

def sort_paths_based_on_importance(paths):
    paths.sort(key=lambda x: (x['metrics']['metric']['relevant_score']['score_equivalence_normalized'],
                               x['metrics']['metric']['faithful_score']['scor_faithfulness_normalized'],
                               x['metrics']['metric']['relevant_score']['score_relevance_normalized']), reverse=True)
    return paths

def collect_all_sub_conversations(path):
    path = path['response']
    all_sub_conversations = []
    all_sub_conversations.append(path['agent_conversation'])
    if 'planner' in path['memory'] and path['memory']['planner'] is not None:
        all_sub_conversations.append(path['memory']['planner'])
    if 'searcher' in path['memory'] and path['memory']['searcher'] is not None:
        all_sub_conversations.append(path['memory']['searcher']['conversation'])
    if 'reasoner' in path['memory'] and path['memory']['reasoner'] is not None:
        all_sub_conversations.append(path['memory']['reasoner'])
    if 'validator' in path['memory'] and path['memory']['validator'] is not None:
        all_sub_conversations.append(path['memory']['validator'])
    if 'summarizer' in path['memory'] and path['memory']['summarizer'] is not None:
        all_sub_conversations.append(path['memory']['summarizer'])
    return all_sub_conversations
    

parser = argparse.ArgumentParser(description="Extract training data for self-training")
parser.add_argument("--input_addr", type=str, help="Path to the input file")
parser.add_argument("--output_addr", type=str, help="Path to the output file")

if __name__ == "__main__":
    args = parser.parse_args()
    input_addr = args.input_addr
    output_addr = args.output_addr

    max_to_keep_per_example = 3
    max_tokens_to_keep = 16000


    sum_reward = 0
    total_examples = 0
    total_data = 0
    average_equivalence = 0
    average_good_path = 0
    all_logs = []
    for addr in [input_addr]:
        all_logs.extend(load_logs(addr))

    dataset_final = []
    for data in all_logs:
        scores = collect_all_path_scores(data)
        if len(scores) == 0 or filter_out_example_based_on_reward(scores):
            continue
        rewards = get_all_rewards_for_example(scores)
        max_reward = max(rewards)
        good_paths = get_examples_with_this_reward(data, max_reward)
        sorted_good_paths = sort_paths_based_on_importance(good_paths)[:max_to_keep_per_example]
        sum_reward += max_reward * len(sorted_good_paths)
        total_examples += len(sorted_good_paths)
        average_good_path += len(sorted_good_paths)
        total_data += 1
        for path in sorted_good_paths:
            average_equivalence += path['metrics']['metric']['relevant_score']['score_equivalence_normalized']
            all_sub_conversations = collect_all_sub_conversations(path)
            for sub_conversation in all_sub_conversations:
                dataset_final.append(sub_conversation)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    print(len(dataset_final))
    print(sum_reward / total_examples)
    print(average_good_path / total_data)
    print(average_equivalence / total_examples)
    print(total_data)

    max_tokens = 0
    exceed_max_tokens = 0

    dataset_final_with_style = []

    for data in dataset_final:
        chat_template = tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=False)
        num_tokens = len(tokenizer(chat_template)['input_ids'])
        max_tokens = max(max_tokens, num_tokens)
        if num_tokens > max_tokens_to_keep:
            exceed_max_tokens += 1
        else:
            dataset_final_with_style.append({"messages": data})
    print(max_tokens)
    print(exceed_max_tokens)
    print(len(dataset_final_with_style))

    with open(output_addr, 'w') as f:
        json.dump(
            dataset_final_with_style,
            f,
            indent=4
        )

    