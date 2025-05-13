from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json


ANSWER_GENERATION_SIMPLE_SYSTEM_PROMPT = """You are a helpful and capable agent, and your goal is to help the user generate a concise response to the given question. The user provide you with a question, some information, and a plan. Your goal is to help the user generate a response that is correct, concise, and grounded with supporting information. You should not provide any new information that is not grounded with the information provided by the user. Also, you should not provide any information that is not directly related to the question and unnecessary.

# Your input:
    - "question": the question the user wants to answer.
    - "supporting information": the information the user has gathered so far.
    - "plan": the plan the user has made to answer the question.
    - "important information": the most important information that should be included in the response.

# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "response": the response to the question using the plan and information provided. Your response should be concise, correct, and grounded with supporting information. You should generate at most 100 words.
"""

ANSWER_GENERATION_SIMPLE_SYSTEM_PROMPT_NO_CONCISE = """You are a helpful and capable agent, and your goal is to help the user generate a response to the given question. The user provide you with a question, some information, and a plan. Your goal is to help the user generate a response that is correct and grounded with supporting information. You should not provide any new information that is not grounded with the information provided by the user. Also, you should not provide any information that is not directly related to the question and unnecessary.

# Your input:
    - "question": the question the user wants to answer.
    - "supporting information": the information the user has gathered so far.
    - "plan": the plan the user has made to answer the question.
    - "important information": the most important information that should be included in the response.

# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "response": the response to the question using the plan and information provided. Your response should be correct, and grounded with supporting information.
"""

ANSWER_GENERATION_SIMPLE_USER_PROMPT = """ # question: {QUESTION}
# supporting information: {SUPPORTING_INFORMATION}
# plan: {PLAN}
# important information: {IMPORTANT_INFORMATION}

You should strirctly follow the instructions below to generate a response to the question in a valid JSON format without any additional text or explanation.
"""

ANSWER_REVISE_SIMPLE_USER_PROMPT = """Thanks for your response. Here are some suggestions to revise your response to make it more accurate, concise, and grounded with supporting information. Please revise your response accordingly and provide the revised response.

# Your input:
    - "suggestion": the suggested revisions to your response.

# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "response": the revised response to the question. Your response should be concise, correct, and grounded with supporting information. You should generate at most 100 words.

# suggestion: {SUGGESTION}

You should strirctly follow the instructions below to generate a response to the question in a valid JSON format without any additional text or explanation.
"""

ANSWER_REVISE_SIMPLE_USER_PROMPT_NO_CONCISE = """Thanks for your response. Here are some suggestions to revise your response to make it more accurate and grounded with supporting information. Please revise your response accordingly and provide the revised response.

# Your input:
    - "suggestion": the suggested revisions to your response.

# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "response": the revised response to the question. Your response should be correct and grounded with supporting information.

# suggestion: {SUGGESTION}

You should strirctly follow the instructions below to generate a response to the question in a valid JSON format without any additional text or explanation.
"""

def initilize_conversation(concise=True):
    conversation = [
        {
            "role": "system",
            "content": ANSWER_GENERATION_SIMPLE_SYSTEM_PROMPT if concise else ANSWER_GENERATION_SIMPLE_SYSTEM_PROMPT_NO_CONCISE
        }
    ]
    return conversation

def generate_answer(question, context, plan, important_information, memory, llm, execute_config):
    if 'generator' not in memory:
        memory['generator'] = initilize_conversation(execute_config['concise'])
    context = "\n".join([f"document {i+1}: {d['text']}\n" for i, d in enumerate(context)])
    conversation = memory['generator']
    conversation.append({
        "role": "user",
        "content": ANSWER_GENERATION_SIMPLE_USER_PROMPT.format(QUESTION=question, SUPPORTING_INFORMATION=context, PLAN=plan, IMPORTANT_INFORMATION=important_information)
    })
    if execute_config['environment_model_server']:
        conversation_text = conversation
    else:
        conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    sampling_parmas = SamplingParams(temperature=execute_config['temperature_environment'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_environment'], logprobs=1)
    response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
    response_obj = str_to_json(response_text)
    conversation.append({
        "role": "assistant",
        "content": response_text
    })
    return response_obj

def revise_answer(suggestion, memory, llm, execute_config):
    concise = execute_config['concise']
    conversation = memory['generator']
    conversation.append({
        "role": "user",
        "content": ANSWER_REVISE_SIMPLE_USER_PROMPT.format(SUGGESTION=suggestion) if concise else ANSWER_REVISE_SIMPLE_USER_PROMPT_NO_CONCISE.format(SUGGESTION=suggestion)
    })
    if execute_config['environment_model_server']:
        conversation_text = conversation
    else:
        conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    sampling_parmas = SamplingParams(temperature=execute_config['temperature_environment'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_environment'], logprobs=1)
    response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
    response_obj = str_to_json(response_text)
    conversation.append({
        "role": "assistant",
        "content": response_text
    })
    return response_obj