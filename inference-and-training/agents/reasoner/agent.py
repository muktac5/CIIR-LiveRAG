from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json

REASONER_SYSTEM_PROMPT = """You are a helpful and capable agent who's task is to help the user analyze the given information from the given aspect aspect by the user to help them answer the given question. In reasoning, you should provide a logical, coherent, and step by step analysis of the information in the requested aspect in a way to be helpful for answering the question. Your goal is to help the user understand the information from the given aspect. You should not answer the question but your analysis should be helpful for the user to answer the question.

# Your input:
    - "question": the question the user wants to answer.
    - "information": a summary of the information the user has gathered so far. This can be empty if the user has not gathered any information yet.
    - "aspect": the aspect the user wants to analyze the information from.
# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "analysis": a list of strings that containing step by step analysis of the information from the given aspect, where each string is a step in this analysis. Your analysis should be a logical, coherent, and step by step analysis of the information in the requested aspect.
Your output should be a valid JSON object enclosed in ```json ``` that contains the fields mentioned above.
"""

REASONER_USER_PROMPT = """ # question: {QUESTION}
# information: {INFORMATION}
# aspect: {ASPECT}
"""

def initilize_conversation():
    conversation = [
        {
            "role": "system",
            "content": REASONER_SYSTEM_PROMPT
        }
    ]
    return conversation

def generate_analysis(question, context, aspect, memory, llm, execute_config):
    if 'reasoner' not in memory:
        memory['reasoner'] = initilize_conversation()
    conversation = memory['reasoner']
    conversation.append({
        "role": "user",
        "content": REASONER_USER_PROMPT.format(QUESTION=question, INFORMATION=context, ASPECT=aspect)
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