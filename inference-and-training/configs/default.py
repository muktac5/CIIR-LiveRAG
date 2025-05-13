DEFAULT_CONFIG = {
    "download_path": "/* address to where you want to download the files*/",
    "agent_model": "/* the address to the trained agent we shared with you*/",
    "environment_model": "tiiuae/Falcon3-10B-Instruct",
    "environment_model_server": True,
    "environment_model_server_log_file": "/* address to where the environment model server log file is saved*/",
    "agent_model_server": True,
    "agent_model_server_log_file": "/* address to where the agent model server log file is saved*/",
    "retriever": "lion_sp_llama3_1b",
    "retriever_log_file": "/* address to where the retriever log file is saved*/",
    "index_addr": "/* address to where the index file is saved*/",
    "max_actions": 30,
    "max_tokens_agent": 32768,
    "max_tokens_environment": 8192,
    "temperature_agent": 0.1,
    "temperature_environment": 0.1,
    "top_p": 0.95,
    "top_k":10,
    "threshold":0.0,
    "max_verifcation_same_query": 5,
    "max_retries": 32,
    "concise": True,
}

DEFAULT_CONFIG_2 = {
    "download_path": "/* address to where you want to download the files*/",
    "agent_model": "/* the address to the trained agent we shared with you*/",
    "environment_model": "tiiuae/Falcon3-10B-Instruct",
    "environment_model_server": True,
    "environment_model_server_log_file": "/* address to where the environment model server log file is saved*/",
    "agent_model_server": True,
    "agent_model_server_log_file": "/* address to where the agent model server log file is saved*/",
    "retriever": "lion_sp_llama3_1b",
    "retriever_log_file": "/* address to where the retriever log file is saved*/",
    "index_addr": "/* address to where the index file is saved*/",
    "max_actions": 30,
    "max_tokens_agent": 32768,
    "max_tokens_environment": 8192,
    "temperature_agent": 0.1,
    "temperature_environment": 0.1,
    "top_p": 0.95,
    "top_k":10,
    "threshold":0.0,
    "max_verifcation_same_query": 5,
    "max_retries": 32,
    "concise": True
}

DEFAULT_CONFIG_EVALUATION = {
    "download_path": "/* address to where you want to download the files*/",
    "judge_model": "Qwen/Qwen2.5-14B-Instruct",
    "judge_model_server": True,
    "judge_model_server_log_file": "/* address to where the judge model server log file is saved*/",
    "max_tokens_judge": 16384,
    "temperature_judge": 0.5,
    "top_p": 0.95,
    "num_samples_judge": 5,
    "ragas": True
}

DEFAULT_CONFIG_EVALUATION_2 = {
    "download_path": "/* address to where you want to download the files*/",
    "judge_model": "Qwen/Qwen2.5-14B-Instruct",
    "judge_model_server": True,
    "judge_model_server_log_file": "/* address to where the judge model server log file is saved*/",
    "max_tokens_judge": 16384,
    "temperature_judge": 0.5,
    "top_p": 0.95,
    "num_samples_judge": 5,
    "ragas": True
}