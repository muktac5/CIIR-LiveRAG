import re
import json, json5

class MyJsonException(Exception):
    def __init__(self, message : list):
        super().__init__(str(message))
        self.message = message
    def __str__(self):
        return str([str(e) for e in self.message])

def str_to_json(input_str):
    original = input_str
    lines = input_str.strip().splitlines()

    cleaned_lines = [line for line in lines if line.strip().lower() != "json"]
    input_str = "\n".join(cleaned_lines).strip()

    if input_str.startswith("```json"):
        input_str = input_str[len("```json"):].strip()
    elif input_str.startswith("```"):
        input_str = input_str[len("```"):].strip()
    if input_str.endswith("```"):
        input_str = input_str[:-3].strip()

    if input_str.endswith("/json"):
        input_str = input_str[:-5].strip()

    if input_str.startswith("json.dumps(") and input_str.endswith(")"):
        input_str = input_str[len("json.dumps("):-1].strip()
    
    if input_str.startswith("json.loads('") and input_str.endswith("')"):
        input_str = input_str[len("json.loads('"):-2].strip()

    if input_str.endswith("```json"):
        input_str = input_str[:-7].strip()

    input_str = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", input_str)

    # Try parsing with json5 first, then fallback to standard json
    errros = []
    for parser in (json5.loads, json.loads):
        try:
            return parser(input_str)
        except Exception as e:
            print(f"[DEBUG] Parser {parser.__name__} failed with: {e}")
            print("[DEBUG] Invalid JSON string:")
            print(input_str)
            errros.append(e)
        
    print("[DEBUG] Final cleaned input (still invalid):")
    print(input_str)
    raise MyJsonException(errros)

