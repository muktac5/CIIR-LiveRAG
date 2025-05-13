from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json
from utils.server_llm import SeverLLM, load_url_from_log_file

RELEVANT_SYSTEM_PROMPT = """You are an impartial judge who has been asked to evaluate how the generated answers the question directly and aligns with the ground truth based on the given criteria.

# Your input:
    - "question": The question that was asked
    - "answer": The generated answer.
    - "ground_truth": The ground truth answer.
    - "criteria": The criteria based on which you evaluate the answer.

# your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "score_equivalence": an int indicating the score you assign to the generated answer based on the given criteria in comparison with the ground truth.
    - "score_relevance": an int indicating the score you assign to the generated answer based on the given criteria based on the degree to which the answer directly addresses the question.
    - "rationale_equivalence": a str indicating the rationale behind your score_equivalence.
    - "rationale_relevance": a str indicating the rationale behind your score_relevance.

"""

# RELEVANT_USER_PROMPT = """# question: {QUESTION}
# # answer: {ANSWER}
# # ground truth: {GROUND_TRUTH}
# # criteria: You should generate a score based on the following criteria:
# 2: The answer is correct and relevant according to the ground truth and contains no irrelevant information.
# 1: The answer is correct and relevant according to the ground truth but contains irrelevant information.
# 0: The answer is not provided.
# -1: The answer is incorrect or irrelevant according to the ground truth.

# You need to provide a JSON object enclosed in ```json ``` block.
# """

RELEVANT_USER_PROMPT = """# question: {QUESTION}
# answer: {ANSWER}
# ground truth: {GROUND_TRUTH}
# criteria: You should generate a score based on the following criteria:
2: Correct and relevant (no irrelevant information).
1: Correct but contains irrelevant information.
0: No answer provided (abstention).
-1: Incorrect answer.

You need to provide a valid JSON object enclosed in ```json ``` block.
"""

FAITHFULNESS_SYSTEM_PROMPT = """You are an impartial judge who has been asked to evaluate how well the generated answer is grounded in the supporting knowledge provided based on the given criteria.

# Your input:
    - "question": The question that was asked
    - "answer": The generated answer.
    - "context": The a list of supporting documents that you should check the answer based on.
    - "criteria": The criteria based on which you evaluate the answer.

# your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "score": an int indicating the score you assign to the generated answer based on the given criteria.
    - "rationale": a str indicating the rationale behind your score.

"""

FAITHFULNESS_SYSTEM_PROMPT_RAGAS_EXTRACT = """You are an impartial judge who has been asked to evaluate how the generated output is faithful to the supporting documents provided. Your task is to given a generated output for a given question, extract all atomic pieces of information from the generated output. These atomic pieces of information are the smallest units of information that discuss a single unique aspect of the answer to the question.

# Your input:
    - "question": The question that was asked
    - "answer": The generated answer.

# your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "answer_aspects": a list of atomic pieces of information extracted from the generated output. Each atomic piece of information is a json object with the following fields:
        - "id": the id of the atomic piece of information.
        - "aspect": the atomic piece of information extracted from the generated output.
        - "explanation": a brief explanation of why this atomic piece of information is important or relevant to the question.
        - "evidence": the evidence from the generated output that supports this atomic piece of information.
"""

FAITHFULNESS_USER_PROMPT_RAGAS_EXTRACT = """# question: {QUESTION}
# answer: {ANSWER}
"""

FAITHFULNESS_SYSTEM_PROMPT_RAGAS_SCORE = """You are an impartial judge who has been asked to evaluate how well the generated answer is grounded in the supporting knowledge provided based on the given criteria. You will be given a list of atomic aspects from the generated output and a list of supporting documents. Your task is to go through each extracted atomic aspect and check if its evidence from the generated output is grounded in the supporting documents. You need to provide a score between -1 and 1 that indicates the degree of match between the two outputs from the aspect point of view, using the following criteria:

1: Full support. All answer parts are grounded in the supporting documents.
0: Partial support. Not all answer parts are grounded in the supporting documents.
-1: No support. All answer parts are not grounded in the supporting documents.

# Your input:
    - "question": The question that was asked
    - "answer_aspects": a list of atomic pieces of information extracted from the generated output. Each atomic piece of information is a json object with the following fields:
        - "id": the id of the atomic piece of information.
        - "aspect": the atomic piece of information extracted from the generated output.
        - "explanation": a brief explanation of why this atomic piece of information is important or relevant to the question.
        - "evidence": the evidence from the generated output that supports this atomic piece of information.
    - "context": The a list of supporting documents that you should check the answer based on.

# your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "scores_list": a list of scores for each atomic aspect, indicating the score you assign to the generated answer based on the given criteria. Each object in the list should be a valid json object that contain the following fields:
        - "id": the id of the atomic piece of information.
        - "score": an int indicating the score you assign to the generated answer based on the given criteria.
        - "rationale": a str indicating the rationale behind your score.
"""

FAITHFULNESS_USER_PROMPT_RAGAS_SCORE = """# question: {QUESTION}
# answer_aspects: {ASPECT}
# context: {CONTEXT}
"""

FAITHFULNESS_USER_PROMPT = """# question: {QUESTION}
# answer: {ANSWER}
# context: {CONTEXT}
# criteria: You should generate a score based on the following criteria:
1: Full support. All answer parts are grounded in the supporting documents.
0: Partial support. Not all answer parts are grounded in the supporting documents.
-1: No support. All answer parts are not grounded in the supporting documents.

You need to provide a valid JSON object enclosed in ```json ``` block.
"""

def relevant_score(question, answer, ground_truth, llm, execute_config):
    sampling_params = SamplingParams(n=execute_config['num_samples_judge'], temperature=execute_config['temperature_judge'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_judge'], logprobs=1)
    conversation = [
        {
            "role": "system",
            "content": RELEVANT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": RELEVANT_USER_PROMPT.format(QUESTION=question, ANSWER=answer, GROUND_TRUTH=ground_truth)
        }
    ]
    score_dist_equality = {2:0, 1:0, 0:0, -1:0}
    score_dist_relevance = {2:0, 1:0, 0:0, -1:0}
    if execute_config['judge_model_server']:
        prompt_judge = conversation
    else:
        prompt_judge = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    scores_raw = llm.generate(prompt_judge, sampling_params)
    rationals = []
    for score_raw in scores_raw[0].outputs:
        score = str_to_json(score_raw.text)
        score_dist_equality[score['score_equivalence']] += (1 / execute_config['num_samples_judge'])
        score_dist_relevance[score['score_relevance']] += (1 / execute_config['num_samples_judge'])
        rationals.append(score)
    score_equality = sum([k*v for k,v in score_dist_equality.items()])
    score_relevance = sum([k*v for k,v in score_dist_relevance.items()])
    output = {
        "score_relevance": score_relevance,
        "score_relevance_normalized": (score_relevance + 1) / 3,
        "score_equivalence": score_equality,
        "score_equivalence_normalized": (score_equality + 1) / 3,
        "rationals": rationals
    }
    return output

def faithful_score(question, answer, context, llm, execute_config):
    sampling_params = SamplingParams(n=execute_config['num_samples_judge'], temperature=execute_config['temperature_judge'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_judge'], logprobs=1)
    conversation = [
        {
            "role": "system",
            "content": FAITHFULNESS_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": FAITHFULNESS_USER_PROMPT.format(QUESTION=question, ANSWER=answer, CONTEXT=context)
        }
    ]
    score_dist = {1:0, 0:0, -1:0}
    if execute_config['judge_model_server']:
        prompt_judge = conversation
    else:
        prompt_judge = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    scores_raw = llm.generate(prompt_judge, sampling_params)
    rationals = []
    for score_raw in scores_raw[0].outputs:
        score = str_to_json(score_raw.text)
        score_dist[score['score']] += (1 / execute_config['num_samples_judge'])
        rationals.append(score)
    score = sum([k*v for k,v in score_dist.items()])
    output = {
        "scor_faithfulness": score,
        "scor_faithfulness_normalized": (score + 1) / 2,
        "rationals": rationals
    }
    return output

def faithful_score_ragas_extract(question, answer, llm, execute_config):
    sampling_params = SamplingParams(n=1, temperature=0, top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_judge'], logprobs=1)
    conversation = [
        {
            "role": "system",
            "content": FAITHFULNESS_SYSTEM_PROMPT_RAGAS_EXTRACT
        },
        {
            "role": "user",
            "content": FAITHFULNESS_USER_PROMPT_RAGAS_EXTRACT.format(QUESTION=question, ANSWER=answer)
        }
    ]
    if execute_config['judge_model_server']:
        prompt_judge = conversation
    else:
        prompt_judge = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    asoects_raw = llm.generate(prompt_judge, sampling_params)
    aspects_json = str_to_json(asoects_raw[0].outputs[0].text)
    return aspects_json['answer_aspects']

def faithful_score_ragas_score(question, answer, context, llm, execute_config):
    answer_aspects = faithful_score_ragas_extract(question, answer, llm, execute_config)
    sampling_params = SamplingParams(n=execute_config['num_samples_judge'], temperature=execute_config['temperature_judge'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_judge'], logprobs=1)
    conversation = [
        {
            "role": "system",
            "content": FAITHFULNESS_SYSTEM_PROMPT_RAGAS_SCORE
        },
        {
            "role": "user",
            "content": FAITHFULNESS_USER_PROMPT_RAGAS_SCORE.format(QUESTION=question, ASPECT=answer_aspects, CONTEXT=context)
        }
    ]
    if execute_config['judge_model_server']:
        prompt_judge = conversation
    else:
        prompt_judge = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    scores_raw = llm.generate(prompt_judge, sampling_params)
    score_final = 0
    all_scores = []
    for score_raw in scores_raw[0].outputs:
        score = str_to_json(score_raw.text)
        for aspect in score['scores_list']:
            score_final += aspect['score'] / (len(answer_aspects) * execute_config['num_samples_judge'])
            all_scores.append(aspect)
    score_final_normalized = (score_final + 1) / 2
    output = {
        "scor_faithfulness": score_final,
        "scor_faithfulness_normalized": score_final_normalized,
        "rationals": all_scores
    }
    return output


def metric(question, geenrated_output, ground_truth, context, execute_config):
    context = "\n".join([f"document {i}: {ctx['text'].replace('"', "'")}" for i, ctx in enumerate(context)])
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
    relevant_output = relevant_score(question, geenrated_output, ground_truth, llm, execute_config)
    if execute_config['ragas']:
        faithful_output = faithful_score_ragas_score(question, geenrated_output, context, llm, execute_config)
    else:
        faithful_output = faithful_score(question, geenrated_output, context, llm, execute_config)
    return {
        "question": question,
        "generated_output": geenrated_output,
        "ground_truth": ground_truth,
        "context": context,
        "relevant_score": relevant_output,
        "faithful_score": faithful_output,
    }