from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json
from utils.server_llm import SeverLLM, load_url_from_log_file

_EVAL_SYSTEM_PROMPT_EXTRACT_ASPECTS = """You are an impartial judge who has been asked to evaluate how the generated output answers the question. Your task is to given an expected output for a given question, extract all atomic pieces of information from the expected output. These atomic pieces of information are the smallest units of information that discuss a single unique aspect of the answer to the question.

# Your input:
    - "question": The question that was asked
    - "expected_output": The expected output.
# your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "expected_output_aspects": a list of atomic pieces of information extracted from the expected output. Each atomic piece of information is a json object with the following fields:
        - "aspect": the atomic piece of information extracted from the expected output.
        - "explanation": a brief explanation of why this atomic piece of information is important or relevant to the question.
        - "evidence": the evidence from the expected output that supports this atomic piece of information.
"""

_EVAL_USER_PROMPT_EXTRACT_ASPECTS = """# question: {QUESTION}
# expected_output: {EXPECTED_OUTPUT}

Your output should be a JSON object enclosed in ```json ``` block in the given format.
"""

_EVAL_SYSTEM_PROMPT_MATCHING_ASPECTS = """You are an impartial judge who has been asked to evaluate how the generated output answers the question. You will be given a question, the expected output, a generated output, and a single aspect that you need to evaluate if the generated output contains the same information about from that aspect point of view. Your task is to compare the generated output and the expected output based on the given aspect. You need to provide a score between -1 and 2 that indicates the degree of match between the two outputs from the aspect point of view, using the following criteria:

# scoring criteria:
2: Correct and relevant (no irrelevant information).
1: Correct but contains irrelevant information.
0: No answer provided (abstention).
-1: Incorrect answer.

# Your input:
    - "question": The question that was asked
    - "expected_output": The expected output.
    - "generated_output": The generated output.
    - "aspect": The aspect that you need to evaluate the generated output based on it, containing the following fields:
        - "aspect": the atomic piece of information extracted from the expected output.
        - "explanation": a brief explanation of why this atomic piece of information is important or relevant to the question.
        - "evidence": the evidence from the expected output that supports this atomic piece of information.
# your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "score": an int indicating the score you assign to the generated output based on the given aspect.
    - "rationale": a str indicating the rationale behind your score.
"""

_EVAL_USER_PROMPT_MATCHING_ASPECTS = """# question: {QUESTION}
# expected_output: {EXPECTED_OUTPUT}
# generated_output: {GENERATED_OUTPUT}
# aspect: {ASPECT}
Your output should be a JSON object enclosed in ```json ``` block in the given format.
"""

def extract_aspects(question, ground_truth, llm, execute_config):
    sampling_params = SamplingParams(n=1, temperature=0, top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_judge'], logprobs=1)
    conversation = [
        {
            "role": "system",
            "content": _EVAL_SYSTEM_PROMPT_EXTRACT_ASPECTS
        },
        {
            "role": "user",
            "content": _EVAL_USER_PROMPT_EXTRACT_ASPECTS.format(QUESTION=question, EXPECTED_OUTPUT=ground_truth)
        }
    ]
    if execute_config['judge_model_server']:
        prompt_judge = conversation
    else:
        prompt_judge = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    aspects_raw = llm.generate(prompt_judge, sampling_params)
    aspects_json = str_to_json(aspects_raw[0].outputs[0].text)
    return aspects_json['expected_output_aspects']

def matching_aspects(question, ground_truth, geenrated_output, aspect, llm, execute_config):
    sampling_params = SamplingParams(n=execute_config['num_samples_judge'], temperature=execute_config['temperature_judge'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_judge'], logprobs=1)
    conversation = [
        {
            "role": "system",
            "content": _EVAL_SYSTEM_PROMPT_MATCHING_ASPECTS
        },
        {
            "role": "user",
            "content": _EVAL_USER_PROMPT_MATCHING_ASPECTS.format(QUESTION=question, EXPECTED_OUTPUT=ground_truth, GENERATED_OUTPUT=geenrated_output, ASPECT=aspect)
        }
    ]
    if execute_config['judge_model_server']:
        prompt_judge = conversation
    else:
        prompt_judge = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    score_raw = llm.generate(prompt_judge, sampling_params)
    dist = {2:0, 1:0, 0:0, -1:0}
    rationale = []
    for score_raw in score_raw[0].outputs:
        score = str_to_json(score_raw.text)
        dist[score['score']] += 1
        rationale.append(score['rationale'])
    score = sum([k*v for k,v in dist.items()]) / execute_config['num_samples_judge']
    score_normalized = (score + 1) / 3
    output = {
        "score": score,
        "score_normalized": score_normalized,
        "rationale": rationale
    }
    return output

def metric_coverage(question, geenrated_output, ground_truth, context, execute_config):
    question = question.replace('"', "'")
    ground_truth = ground_truth.replace('"', "'")
    geenrated_output = geenrated_output.replace('"', "'")
    if execute_config['judge_model_server']:
        llm_url = load_url_from_log_file(execute_config['judge_model_server_log_file'])
        llm = SeverLLM(
            base_url=llm_url,
            model=execute_config['judge_model']
        )
    else:
        llm = LLM(execute_config['judge_model'], download_dir=execute_config["download_path"], gpu_memory_utilization=0.9, max_model_len=8192)
    extracted_aspects = extract_aspects(question, ground_truth, llm, execute_config)
    aspect_scores = []
    for aspect in extracted_aspects:
        score = matching_aspects(question, ground_truth, geenrated_output, aspect, llm, execute_config)
        aspect_scores.append({
            "aspect": aspect,
            "score": score
        })
    score_final = sum([score['score']['score'] for score in aspect_scores]) / len(aspect_scores)
    score_final_normalized = (score_final + 1) / 3
    output = {
        "score": score_final,
        "score_normalized": score_final_normalized,
        "aspect_scores": aspect_scores
    }
    return output