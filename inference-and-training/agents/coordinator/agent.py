from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json
from retrieval.retrievers import BM25Retriever, SparseRetriever
from enum import Enum
from agents.generator.agent import generate_answer, revise_answer
from agents.summarizer.agent import generate_summary
from agents.reasoner.agent import generate_analysis
from agents.validator.agent import validate_response
from agents.searcher.agent import search
from agents.planner.agent import generate_plan
import json
from utils.server_llm import SeverLLM, load_url_from_log_file


class AgentType(Enum):
    VALIDATOR = "validator"
    SUMMARIZER = "summarizer"
    PLANNER = "planner"
    REASONER = "reasoner"
    ANSWERER = "answerer"
    REVISER = "reviser"
    SEARCHER = "searcher"
    FINISHER = "finisher"


COORDINATOR_SYSTEM_PROMPT = """You are a highly capable agent, and your goal is to  generate a concise response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is concise, informative, and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.

## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.

## searcher: This agent can help you search for information that can help you answer the given question. You can use this agent to search for information that can help you answer the question. You can use this agent multiple times in any time during the task to search for information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
### output:
    - "found_information": a boolean value indicating whether the search found relevant information.
    - "documents": a list of documents that the agent has found that are relevant to the search query.

## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

"""

COORDINATOR_SYSTEM_PROMPT_NO_CONCISE = """You are a highly capable agent, and your goal is to  generate a response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is informative and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.

## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.

## searcher: This agent can help you search for information that can help you answer the given question. You can use this agent to search for information that can help you answer the question. You can use this agent multiple times in any time during the task to search for information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
### output:
    - "found_information": a boolean value indicating whether the search found relevant information.
    - "documents": a list of documents that the agent has found that are relevant to the search query.

## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

"""

COORDINATOR_USER_PROMPT = """To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. In selecting an agent, you should select the agent that you think is the most appropriate to take next. Using the same agent multiple times is allowed if you think it is necessary, but might not be useful always. In response, the agent will provide you with the necessary information to continue the conversation.
"""

def initilize_conversation(question, concise=True):
    conversation = [
        {
            "role": "system",
            "content": COORDINATOR_SYSTEM_PROMPT.format(QUESTION=question) if concise else COORDINATOR_SYSTEM_PROMPT_NO_CONCISE.format(QUESTION=question)
        }
    ]
    return conversation

def initilize_memory():
    memory = {
        "responses" : [],
        "final_response": None
    }
    return memory

def update_conversation(conversation, role, content):
    conversation.append({
        "role": role,
        "content": content
    })
    return conversation

def get_next_user_prompt(question, output_obj, agent_llm, retriever, environment_llm, memory, execute_config):
    agent_type = output_obj['agent']
    if agent_type == AgentType.VALIDATOR.value:
        context = output_obj['input']['information']
        response = output_obj['input']['response']
        output_agent = validate_response(question, context, response, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.SUMMARIZER.value:
        context = output_obj['input']['information']
        output_agent = generate_summary(question, context, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.PLANNER.value:
        context = output_obj['input']['information']
        output_agent = generate_plan(question, context, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.REASONER.value:
        context = output_obj['input']['information']
        aspect = output_obj['input']['aspect']
        output_agent = generate_analysis(question, context, aspect, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.ANSWERER.value:
        guidance = output_obj['input']['guidance']
        important_information = output_obj['input']['important_information']
        verified_documents = memory['searcher']['verified_documents'] if 'searcher' in memory else []
        output_agent = generate_answer(question, verified_documents, guidance, important_information, memory, environment_llm, execute_config)
        memory['responses'].append(output_agent['response'])
        memory['final_response'] = output_agent['response']
        return json.dumps(output_agent)
    elif agent_type == AgentType.REVISER.value:
        suggestion = output_obj['input']['suggestion']
        output_agent = revise_answer(suggestion, memory, environment_llm, execute_config)
        memory['responses'].append(output_agent['response'])
        memory['final_response'] = output_agent['response']
        return json.dumps(output_agent)
    elif agent_type == AgentType.SEARCHER.value:
        context = output_obj['input']['information']
        suggestions = output_obj['input']['suggestions']
        output_agent = search(question, context, suggestions, memory, agent_llm, retriever, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.FINISHER.value:
        return None
    else:
        raise ValueError("Invalid agent type")

def generate_response(question, execute_config):
    if execute_config['environment_model_server']:
        environment_llm_url = load_url_from_log_file(execute_config['environment_model_server_log_file'])
        environment_llm = SeverLLM(environment_llm_url, execute_config['environment_model'])
    else:
        environment_llm = LLM(execute_config['environment_model'], download_dir=execute_config["download_path"], gpu_memory_utilization=0.45)
    if execute_config['agent_model_server']:
        agent_llm_url = load_url_from_log_file(execute_config['agent_model_server_log_file'])
        agent_llm = SeverLLM(agent_llm_url, execute_config['agent_model'])
    else:
        agent_llm = LLM(execute_config['agent_model'], download_dir=execute_config["download_path"], gpu_memory_utilization=0.45)
    if execute_config['retriever'] == "BM25":
        retriever = BM25Retriever(execute_config['index_addr'])
    if execute_config['retriever'] == "lion_sp_llama3_1b":
        retriever = SparseRetriever(execute_config)
    else:
        raise ValueError("Invalid retriever")
    sampling_params = SamplingParams(temperature=execute_config['temperature_agent'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_agent'], logprobs=1)
    query = question
    conversation = initilize_conversation(query, execute_config['concise'])
    memory = initilize_memory()
    counter = 0
    while counter < execute_config['max_actions']:
        if execute_config['agent_model_server']:
            conversation_text = conversation
        else:
            conversation_text = agent_llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        output_raw = agent_llm.generate(conversation_text, sampling_params)
        output_text = output_raw[0].outputs[0].text
        output_obj = str_to_json(output_text)
        # print(output_obj)
        conversation = update_conversation(conversation, "assistant", output_text)
        user_action = get_next_user_prompt(query, output_obj, agent_llm, retriever, environment_llm, memory, execute_config)
        # print(user_action)
        if user_action is None:
            break
        conversation = update_conversation(conversation, "user", user_action)
        counter += 1
    output_object = {
        "question": query,
        "agent_conversation_text": agent_llm.get_tokenizer().apply_chat_template(conversation, tokenize=False) if not execute_config['environment_model_server'] else conversation,
        "agent_conversation": conversation,
        "verified_documents": memory['searcher']['verified_documents'] if 'searcher' in memory else [],
        "response": memory['final_response'],
        "memory": memory
    }
    return output_object