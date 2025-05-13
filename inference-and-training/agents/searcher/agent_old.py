from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json


# add a query understanding step to the agent
SEARCH_SYSTEM_PROMPT = """You are a helpful and capable agent who's task is to help find information that can help the user answer the given question. The user provides you with a question and some information and suggestions about what aspect to search for. Your goal is to help the user find information that can help them answer the question. This is a multi-turn action. In this action, first, you need to provide a search query that will help you find relevant information to accurately answer the question. Note that search does not provide a direct answer to the question and should be used to gather useful information. In response, the user will provide you with information about the search results. Then, in the second turn, you need to analyze the provided information to verify the accuracy and relevance of the information. In response, the user will provide you with confirmation about the accuracy and relevance of the information. This continues until you no longer need more information and the provided information is sufficient and useful.

# Step 1: 
## your input: The user provides you with the following information:
    - "question": the question the user wants to answer.
    - "information": a summary of the information the user has gathered so far. This can be empty if the user has not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
## your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "search_query": the search query you suggest to find information that can help in answering the user's question.
    - "search_query_explanation": an explanation of why you suggest this search query.
    
# Step 2:
## your input: The user provides you with the search results.
    - "search_results": the search results using the query that you suggested.
## your output: you need to provide a JSON object enclosed in ```json ``` that contains the following fields:
    - "document_id": the document ID of the search result.
    - "query_id": the query ID of the search result.
    - "is_relevant": a boolean value indicating whether the search results are relevant to the query and contains useful information for collecting the information that can help answer the question.
    - "is_relevant_explanation": an explanation of why the search results are relevant or not relevant.
    - "retrieve_more_same_query": a boolean value indicating whether use the same query and look at the next document to find relevant information that can help answer the question. If you set this to true, the user will provide you with more information from the search results of the same query. This is helpful when you think you need to find more relevant information and you want to give the current search query another chance. It is suggest to use the same query at least twice and maximum of {MAX_SAME_QUERY} times. Note that you can not set both "change_search_query" and "retrieve_more_same_query" to true at the same time.
    - "retrieve_more_same_query_explanation": an explanation of why you need more information.
    - "change_search_query": a boolean value indicating whether you think we need to change the search query to find relevant information that can help answer the question. This is helpful when after multiple retrieval results from the same query is given to you and you couldn't find the relevant information. By setting this to true, you suggest a new search query to find relevant information. Note that you can not set both "change_search_query" and "retrieve_more_same_query" to true at the same time.
    - "change_search_query_explanation": an explanation of why you need to change the search query.
    - "new_search_query": the new search query you suggest to find relevant information that can help answer the question. This is only needed if you set "change_search_query" to true. Otherwise, you can leave this field empty.

This is a multi-turn action. You will continue to provide search queries and analyze the search results until you no longer need more information and the provided information is sufficient and useful. Your output should be a valid JSON object enclosed in ```json ``` that contains the fields mentioned above.
"""

SEARCH_USER_PROMPT_TURN_1 = """# question: {QUESTION}"
# information: {INFORMATION}
# suggestions: {SUGGESTIONS}
"""

SEARCH_USER_PROMPT_TURN_2 = """"This is the information resulted from your search query:
Query ID: {QID}
Documnet ID: {ID}
Document text: {ANSWER}

Now I need you to verify the information. Be as objective as possible in your verfication.
"""

def initilize_conversation():
    conversation = [
        {
            "role": "system",
            "content": SEARCH_SYSTEM_PROMPT
        }
    ]
    return conversation

def search(question, context, suggestions, memory, llm, retriever, execute_config):
    if 'searcher' not in memory:
        memory['searcher'] = {
            "conversation": initilize_conversation(),
            "queries": {},
            "id_to_query": {},
            "query_id": 0,
            "verified_documents": [],
            "verified_ids": []
        }
    conversation = memory['searcher']['conversation']
    conversation.append({
        "role": "user",
        "content": SEARCH_USER_PROMPT_TURN_1.format(QUESTION=question, INFORMATION=context, SUGGESTIONS=suggestions)
    })
    if execute_config['agent_model_server']:
        conversation_text = conversation
    else:
        conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    sampling_parmas = SamplingParams(temperature=execute_config['temperature_agent'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_environment'], logprobs=1)
    response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
    response_obj = str_to_json(response_text)
    print(response_obj)
    conversation.append({
        "role": "assistant",
        "content": response_text
    })
    counter = 0
    query = response_obj['search_query']
    verified_documents = memory['searcher']['verified_documents']
    verified_ids = memory['searcher']['verified_ids']
    while execute_config['max_verifcation_same_query'] > counter:
        ## save the query in memory
        if query in memory['searcher']['queries']:
            query_id = memory['searcher']['queries'][query]['id']
        else:
            query_id = str(memory['searcher']['query_id'])
            memory['searcher']['queries'][query] = {
                "id": query_id,
                "query": query,
                "documents": {}
            }
            memory['searcher']['id_to_query'][query_id] = query
            memory['searcher']['query_id'] += 1
        document = retriever.search_next(query)
        memory['searcher']['queries'][query]['documents'][document['id']] = {"doc": document, "verified": False}
        conversation.append({
            "role": "user",
            "content": SEARCH_USER_PROMPT_TURN_2.format(QID=query_id, ID=document['id'], ANSWER=document['text'], MAX_SAME_QUERY=execute_config['max_verifcation_same_query'])
        })
        if execute_config['agent_model_server']:
            conversation_text = conversation
        else:
            conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
        response_obj = str_to_json(response_text)
        print(response_obj)
        conversation.append({
            "role": "assistant",
            "content": response_text
        })
        is_relevant = response_obj['is_relevant']
        retrieve_more_same_query = response_obj['retrieve_more_same_query']
        change_search_query = response_obj['change_search_query']
        if is_relevant:
            memory['searcher']['queries'][query]['documents'][document['id']]['verified'] = True
            if document['id'] not in verified_ids:
                verified_ids.append(document['id'])
                verified_documents.append(document)
        if retrieve_more_same_query:
            continue
        if change_search_query:
            query = response_obj['new_search_query']
            continue
        if not retrieve_more_same_query and not change_search_query and is_relevant:
            break
        counter += 1
    return {
        "documents": [doc['text'] for doc in verified_documents],
        "found_information": len(verified_documents) > 0
    }