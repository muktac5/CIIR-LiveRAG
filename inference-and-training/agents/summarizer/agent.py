from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json

SUMMARIZER_SYSTEM_PROMPT = """You are a helpful and capable agent who's task is to help the user summarize the given information. The user provides you with some information and a question. Your goal is to help the user generate a summary of the information in a way that can help the user to answer the question. Your summary should be concise, informative, and relevant to the question.

# Your input:
    - "question": the question the user wants to answer.
    - "information": a summary of the information the user has gathered so far. This can be empty if the user has not gathered any information yet.
# Your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "summary": the summary of the information. Your summary should be concise, informative, and relevant to the question.
Your output should be a valid JSON object enclosed in ```json ``` that contains the fields mentioned above.
"""

SUMMARIZER_USER_PROMPT = """ # question: {QUESTION}
# information: {INFORMATION}
"""

def initilize_conversation():
    conversation = [
        {
            "role": "system",
            "content": SUMMARIZER_SYSTEM_PROMPT
        }
    ]
    return conversation

def generate_summary(question, context, memory, llm, execute_config):
    if 'summarizer' not in memory:
        memory['summarizer'] = initilize_conversation()
    conversation = memory['summarizer']
    conversation.append({
        "role": "user",
        "content": SUMMARIZER_USER_PROMPT.format(QUESTION=question, INFORMATION=context)
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