from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json
from utils.server_llm import SeverLLM, load_url_from_log_file
from evaluation.metric import faithful_score
import copy

_PROMPT_PROMPT_REWARD_MODEL = """You are an impartial judge. Your task is to evaluate the quality of the generated response to the query. This is a multi-turn task so you need to follow the instructions step by step. When you complete each step, the next step will be provided to you. 

# step 1: read the query, the generated response, and the reference response carefully. Then, extract the key information and aspects from the generated response and the reference response. The key information includes the main points, arguments, and any important details. 
## input format:
    - "query": the given query to the system.
    - "response": the generated response to the query.
    - "reference": the reference response to the query.
## output format: You should generated a valid JSON object enclosed in ```json ``` block (starts with ```json and ends with ```)  containing the following fields:
    - "generated_response_aspects": a list of key information and aspects extracted from the generated response. Each aspect is a json object with the following fields:
        - "aspect": the key information or aspect extracted from the generated response.
        - "explanation": a brief explanation of why this aspect is important or relevant to the query.
        - "evidence": the evidence from the generated response that supports this aspect.
    - "reference_response_aspects": a list of key information and aspects extracted from the reference response. Each aspect is a json object with the following fields:
        - "aspect": the key information or aspect extracted from the reference response.
        - "explanation": a brief explanation of why this aspect is important or relevant to the query.
        - "evidence": the evidence from the reference response that supports this aspect.
You should strictly follow the output format and output a valid json object. You should not provide any other information or explanation before or after the json object.
"""

_PROMPT_INPUT_REWARD_MODEL = """# query: {query}
# response: {response}
# reference: {reference}
"""

_PROMPT_CONFIRM_REWARD_MODEL_STEP_2 = """I confirm your output for this task. You can proceed to the next step. 

# step 2: read the extracted aspects for both the generated response and the reference response. Then, compare the two lists of aspects. Finally, you need to match the aspects that both the generated response and the reference response have in common.
## input format:
    - the user will provide a confirmation of the extracted aspects from the previous step.
## output format: You should generated a valid JSON object enclosed in ```json ``` block (starts with ```json and ends with ```) containing the following fields:
    - "common_aspects": a list of json objects that contains the aspects that both the generated response and the reference response have in common. Each aspect is a json object with the following fields:
        - "generated_output_aspect": the aspect from the generated response.
        - "reference_output_aspect": the aspect from the reference response.
        - "explanation": a brief explanation of why this aspect is important or relevant to the query.
        - "match_score": a score between 0 and 2 that indicates the degree of match between the two aspects. A score of 2 means that the two aspects are semantically equivalent, a score of 1 means that the two aspects are similar but not semantically equivalent, and a score of 0 means that the two aspects are not similar at all.
You should strictly follow the output format and output a valid json object. You should not provide any other information or explanation before or after the json object.
"""

_PROMPT_CONFIRM_REWARD_MODEL_STEP_3 = """I confirm your output for this task. You can proceed to the next step. 

# step 3: read the matched common aspects between the generated response and the reference response. Also read all the aspects your previously extracted. Then, you need to evaluate the quality of the generated response based on the matched aspects and the extracted aspects. You need to provide a score between -1 and 2 that indicates the quality of the generated response, using the following criteria:
## scoring criteria:
    - 2: the generated response has a very high quality and contains almost all the key information and aspects that the reference response has. The generated response is very similar to the reference response.
    - 1: the generated response has a high quality and contains some of the key information and aspects that the reference response has, however, it might have some missing or very different key information and aspects than the reference response. The generated response is somewhat similar to the reference response.
    - 0: the generated response is empty or none. Basically, if the genrated response is not provided, the score is 0.
    - -1: the generated response is low quality and contains almost none of the key information and aspects that the reference response has. The generated response is very different from the reference response.
## output format: You should generated a valid JSON object enclosed in ```json ``` block (starts with ```json and ends with ```) containing the following fields:
    - "score": a score between -1 and 2 based on the scoring criteria.`
    - "explanation": a brief explanation of why you gave this score to the generated response.

You should strictly follow the output format and output a valid json object. You should not provide any other information or explanation before or after the json object. This is the last step of the task and there will be no more steps after you complete this step.
"""

def get_reward(question, answer, ground_truth, execute_config):
    question = question.replace('"', "'")
    ground_truth = ground_truth.replace('"', "'")
    answer = answer.replace('"', "'")
    if execute_config['judge_model_server']:
        llm_url = load_url_from_log_file(execute_config['judge_model_server_log_file'])
        llm = SeverLLM(
            base_url=llm_url,
            model=execute_config['judge_model'],
            assume_json=False
        )
    else:
        llm = LLM(execute_config['judge_model'], download_dir=execute_config["download_path"], gpu_memory_utilization=0.9, max_model_len=8192)
    sampling_params = SamplingParams(n=1, temperature=execute_config['temperature_judge'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_judge'], logprobs=1)
    conversation = [
        {
            "role": "system",
            "content": _PROMPT_PROMPT_REWARD_MODEL
        },
        {
            "role": "user",
            "content": _PROMPT_INPUT_REWARD_MODEL.format(query=question, response=answer, reference=ground_truth)
        }
    ]
    score_dist = {2:0, 1:0, 0:0, -1:0}
    explanations = []
    total_success = 0
    for i in range(execute_config['num_samples_judge']):
        new_conversation = copy.deepcopy(conversation)
        if execute_config['judge_model_server']:
            prompt_judge = new_conversation
        else:
            prompt_judge = llm.get_tokenizer().apply_chat_template(new_conversation, tokenize=False, add_generation_prompt=True)
        response = llm.generate(prompt_judge, sampling_params)[0].outputs[0].text
        new_conversation.append({
            "role": "assistant",
            "content": response
        })
        new_conversation.append({
            "role": "user",
            "content": _PROMPT_CONFIRM_REWARD_MODEL_STEP_2
        })
        if execute_config['judge_model_server']:
            prompt_judge = new_conversation
        else:
            prompt_judge = llm.get_tokenizer().apply_chat_template(new_conversation, tokenize=False, add_generation_prompt=True)
        response = llm.generate(prompt_judge, sampling_params)[0].outputs[0].text
        new_conversation.append({
            "role": "assistant",
            "content": response
        })
        new_conversation.append({
            "role": "user",
            "content": _PROMPT_CONFIRM_REWARD_MODEL_STEP_3
        })
        if execute_config['judge_model_server']:
            prompt_judge = new_conversation
        else:
            prompt_judge = llm.get_tokenizer().apply_chat_template(new_conversation, tokenize=False, add_generation_prompt=True)
        if execute_config['judge_model_server']:
            llm.assume_json = True
        response = llm.generate(prompt_judge, sampling_params)[0].outputs[0].text
        new_conversation.append({
            "role": "assistant",
            "content": response
        })
        try:
            response = str_to_json(response)
        except Exception as e:
            continue
        score = response['score']
        explanations.append({
            "score": score,
            "explanation": response['explanation'],
            "conversation": new_conversation
        })
        score_dist[score] += 1
        if execute_config['judge_model_server']:
            llm.assume_json = False
        total_success += 1
    score = 0
    for k, v in score_dist.items():
        score += (k * v) / total_success
    return {
        "reward": score,
        "reward_normalized": (score + 1) / 3,
        "explanation": explanations,
    }