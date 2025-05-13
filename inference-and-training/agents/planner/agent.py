from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json


PLANNER_SYSTEM_PROMPT = """You are a helpful and capable agent who's task is to help the user generate a plan on how to answer the given question. The user provides you with a question and some information. Your goal is to help the user generate a good plan about the steps to take to answer the question. You should provide a plan that result in a correct, concise, and grounded with supporting information response. 

# Your input:
    - "question": the question the user wants to answer.
    - "information": a summary of the information the user has gathered so far. This can be empty if the user has not gathered any information yet.
# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "plan": the plan to answer the question. Your plan should be a set of steps that the user should take to generate a response that is correct, concise, and grounded with supporting information. 
Your output should be a valid JSON object enclosed in ```json ``` that contains the fields mentioned above.
"""

PLANNER_SYSTEM_PROMPT_NO_CONCISE = """You are a helpful and capable agent who's task is to help the user generate a plan on how to answer the given question. The user provides you with a question and some information. Your goal is to help the user generate a good plan about the steps to take to answer the question. You should provide a plan that result in a correct and grounded with supporting information response. 

# Your input:
    - "question": the question the user wants to answer.
    - "information": a summary of the information the user has gathered so far. This can be empty if the user has not gathered any information yet.
# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "plan": the plan to answer the question. Your plan should be a set of steps that the user should take to generate a response that is correct and grounded with supporting information. 
Your output should be a valid JSON object enclosed in ```json ``` that contains the fields mentioned above.
"""

PLANNER_USER_PROMPT = """ # question: {QUESTION}
# information: {INFORMATION}
"""

def initilize_conversation(concise=True):
    conversation = [
        {
            "role": "system",
            "content": PLANNER_SYSTEM_PROMPT if concise else PLANNER_SYSTEM_PROMPT_NO_CONCISE
        }
    ]
    return conversation

def generate_plan(question, context, memory, llm, execute_config):
    if 'planner' not in memory:
        memory['planner'] = initilize_conversation(execute_config['concise'])
    conversation = memory['planner']
    conversation.append({
        "role": "user",
        "content": PLANNER_USER_PROMPT.format(QUESTION=question, INFORMATION=context)
    })
    if execute_config['agent_model_server']:
        conversation_text = conversation
    else:
        conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    sampling_parmas = SamplingParams(temperature=execute_config['temperature_agent'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_environment'], logprobs=1)
    response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
    response_obj = str_to_json(response_text)
    conversation.append({
        "role": "assistant",
        "content": response_text
    })
    return response_obj