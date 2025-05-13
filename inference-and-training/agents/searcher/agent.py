from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json
from retrieval.retrievers import BM25Retriever, SparseRetriever
import requests
import json

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
    - "query_id": the query ID of the search result.
    - "relevance": a valid json list that each object in the list contains the following fields:
        - "doc_id": The document ID of the search result that you want to verify.
        - "is_relevant": a boolean value indicating whether the document is relevant to the query and contains useful information for collecting the information that can help answer the question.
        - "is_relevant_explanation": an explanation of why the document is relevant or not relevant.
    - "change_search_query": a boolean value indicating whether you think we need to change the search query to find relevant information that can help answer the question. This is helpful when after multiple retrieval results from the same query is given to you and you couldn't find the relevant information. By setting this to true, you suggest a new search query to find relevant information. If you need to use the same query again, you can set this to false.
    - "change_search_query_explanation": an explanation of why you need to change the search query.
    - "new_search_query": the new search query you suggest to find relevant information that can help answer the question. This is only needed if you set "change_search_query" to true. Otherwise, you can leave this field empty.
    - "end_search": a boolean that show that you are satisfied with the search results and you don't need to search anymore. This is helpful when you think you have found enough information and you don't need to search anymore. By setting this to true, you suggest to end the search. If you need to continue searching, you can set this to false. This should be false if you set "change_search_query" to true.
    - "end_search_explanation": an explanation of why you need to end the search.


This is a multi-turn action. You will continue to provide search queries and analyze the search results until you no longer need more information and the provided information is sufficient and useful. Your output should be a valid JSON object enclosed in ```json ``` that contains the fields mentioned above.
"""

SEARCH_USER_PROMPT_TURN_1 = """# question: {QUESTION}"
# information: {INFORMATION}
# suggestions: {SUGGESTIONS}
"""

SEARCH_USER_PROMPT_TURN_2 = """"This is the information resulted from your search query:
Query ID: {QID}

Documnet 1 ID: {ID}
Document 1 text: {ANSWER}

Documnet 2 ID: {ID_2}
Document 2 text: {ANSWER_2}


Now I need you to verify the information. Be as objective as possible in your verfication. Note you should strictly follow the given format and do not add any extra explanation or information to your output.
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
    #print(response_obj)
    conversation.append({
        "role": "assistant",
        "content": response_text
    })
    counter = 0
    query = response_obj['search_query']
    verified_documents = memory['searcher']['verified_documents']
    verified_ids = memory['searcher']['verified_ids']
    while execute_config['max_verifcation_same_query'] > counter:
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
        if type(retriever) == SparseRetriever:
            document_1 = retriever.search_next(query)
            document_2 = retriever.search_next(query)
        elif type(retriever) == BM25Retriever:
            document_1 = retriever.search_next(query)
            document_2 = retriever.search_next(query)

        #print("document from retriever",document_1)
        memory['searcher']['queries'][query]['documents'][document_1['id']] = {"doc": document_1, "verified": False}
        memory['searcher']['queries'][query]['documents'][document_2['id']] = {"doc": document_2, "verified": False}
        conversation.append({
            "role": "user",
            "content": SEARCH_USER_PROMPT_TURN_2.format(QID=query_id, ID=document_1['id'], ANSWER=document_1['text'].replace('"', "'"), ID_2=document_2['id'], ANSWER_2=document_2['text'].replace('"', "'"))
        })
        if execute_config['agent_model_server']:
            conversation_text = conversation
        else:
            conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
        #print(response_text)
        response_obj = str_to_json(response_text)
        #print(response_obj)
        conversation.append({
            "role": "assistant",
            "content": response_text
        })
        judgements = response_obj['relevance']
        for judgement in judgements:
            # print(memory['searcher']['queries'][query]['documents'].keys())
            doc_id = str(judgement['doc_id'])
            is_relevant = judgement['is_relevant']
            if is_relevant:
                memory['searcher']['queries'][query]['documents'][doc_id]['verified'] = True
                if doc_id not in verified_ids:
                    verified_ids.append(doc_id)
                    verified_documents.append(memory['searcher']['queries'][query]['documents'][doc_id]['doc'])
        
        change_search_query = response_obj['change_search_query']  
        if change_search_query:
            query = response_obj['new_search_query']
            continue
        is_ending_search = response_obj['end_search']
        if is_ending_search:
            break
        counter += 1
    return {
        "documents": [doc['text'] for doc in verified_documents],
        "found_information": len(verified_documents) > 0
    }