from openai import OpenAI
from vllm import SamplingParams
from utils.json_utils import str_to_json, MyJsonException
import time


def load_url_from_log_file(log_addr: str):
    with open(log_addr, "r") as f:
        lines = f.readlines()
    host = lines[0].strip()
    port = lines[1].strip()
    url = f"http://{host}:{port}/v1"
    return url

class SeverLLMOutput:
    def __init__(self, text):
        self.text = text

    def __getitem__(self, index):
        return self.text

class SeverLLMResponse:
    def __init__(self, texts):
        self.outputs = [SeverLLMOutput(text) for text in texts]

    def __getitem__(self, index):
        return self.outputs[index]


class SeverLLM:
    def __init__(self, base_url: str, model: str, max_retries: int = 10, assume_json: bool = True):
        self.api_key = "EMPTY"
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        self.max_retries = max_retries
        self.assume_json = assume_json

    def generate(self, messages: list, sampling_params: SamplingParams = SamplingParams()):
        temperature = sampling_params.temperature
        retries = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # max_tokens=sampling_params.max_tokens,
                    temperature=temperature,
                    top_p=sampling_params.top_p,
                    n=sampling_params.n,
                )
                response = self.post_process(response)
                return response
            except Exception as e:
                retries += 1
                if isinstance(e, MyJsonException):
                    if temperature < 0.5:
                        temperature += 0.1
                print(f"Error: {e}")
                print(self.model)
                # print(e.with_traceback())
                print("Retrying...")
                if retries >= self.max_retries:
                    return [SeverLLMResponse(["cannot generate a response"])]
                time.sleep(1)
    
    def post_process(self, response):
        contents = []
        for choice in response.choices:
            content = choice.message.content
            if self.assume_json:
                json_obj = str_to_json(content)
                # print(json_obj)
            contents.append(content)        
        return [SeverLLMResponse(contents)]