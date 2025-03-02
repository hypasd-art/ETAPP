import os
import time
from typing import Optional
import re
from typing import Optional, List
from termcolor import colored
import time
import openai
import json
import logging
from PLA.Inference.constant import *


def react_parser(message):
    thought = r"Thought:(.*)Final Answer:"
    pattern = r"(.*)Final Answer:(.*)"
    thought_match = re.search(thought, message, re.DOTALL)
    match = re.search(pattern, message, re.DOTALL)

    if match:
        thought_content = thought_match.group(1).strip() if thought_match is not None else ""
        action_content = 'finish'
        action_input_content = match.group(2).strip() 
        return thought_content, action_content, action_input_content
    

    try:
        thought_content = re.search("(?:Thought:(.*?))?(.*?)(?=Action:)", message, re.DOTALL).group(2).strip() 
    except:
        thought_content = '\n'

    try:
        action_content = re.search("Action:(.*?)Action Input:", message, re.DOTALL).group(1).strip() 
    except:
        action_content = '\n'

    try:
        action_input_content = re.search("Action Input:(.*)", message, re.DOTALL).group(1).replace("\n ", "").strip()   
    except:
        action_input_content = '{}'

    return thought_content, action_content, action_input_content
    
def llama_parser(result):
    if "<|eot_id|>" in result:
        return "", {}
    if "name" not in result and "paramters" not in result:
        return "", {}
    all_action = {}
    result = result.replace("<|python_tag|>", "")
    result = result.replace("<|eom_id|>", "")
    result = result.replace("<|eot_id|>", "")


    if ";" in result:
        """
        "<|python_tag|>{\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"10\", \"k\": \"3\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"15\", \"k\": \"5\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"20\", \"k\": \"7\", \"p\": \"0\"}}"
        """
        function_calls = result.split(";")
        function_calls = [json.loads(func_call) for func_call in function_calls]
    else:
        """
        "[\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"20\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"12\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"10\", \"k\": \"3\"}}\n]"
        """
        function_calls = eval(result)
        if type(function_calls) == dict:
            function_calls = [function_calls]

    for i, func_call in enumerate(function_calls):
        name = func_call["name"]
        params = func_call["parameters"]
        all_action[i] = ({"action": name, "action_input": params})

    
    return "", all_action 

def watt_parser(result):
    
    f = True
    for tool_name in list(transfor_dict.keys()):
        if tool_name in result:
            f = False
    if f:
        return "", {}
    all_action = {}
    result = result.replace("<|python_tag|>", "")
    result = result.replace("<|eom_id|>", "")
    result = result.replace("<|eot_id|>", "")


    if ";" in result:
        """
        "<|python_tag|>{\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"10\", \"k\": \"3\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"15\", \"k\": \"5\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"20\", \"k\": \"7\", \"p\": \"0\"}}"
        """
        function_calls = result.split(";")
        fixed_calls = []
        for func_str in function_calls:
            corrected_str = re.sub(
                r'"name":\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                r'"name": "\1"',
                func_str.strip()
            )
            fixed_calls.append(corrected_str)
        function_calls = [json.loads(func_call) for func_call in fixed_calls]
        for i, function_call in enumerate(function_calls):
            if type(function_call["parameters"]) == dict:
                function_call["parameters"] = str(function_call["parameters"])
                function_calls[i] = function_call
    else:
        """
        "[\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"20\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"12\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"10\", \"k\": \"3\"}}\n]"
        """
        try:
            corrected_result = re.sub(
                r'"name":\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                r'"name": "\1"',  
                result
            )
            function_calls = json.loads(corrected_result)
            if type(function_calls) == dict:
                function_calls = [function_calls]
        except json.JSONDecodeError:
            
            function_calls = []
            pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)(?=\s*(?:[,;\]}]|$))'
            matches = re.finditer(pattern, result)
            for match in matches:
                
                func_name, params_str = match.groups()
                params = {}
                if params_str:
                    
                    param_list = re.split(r',\s*(?![^()]*\))', params_str)
                    
                    for param in param_list:
                        param = param.strip()
                        if '=' in param:
                            key, value = param.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                            
                            try:
                                value = int(value)
                            except ValueError:
                                try:
                                    value = float(value)
                                except ValueError:
                                    pass  
                            params[key] = value
                function_calls.append({"name": func_name, "parameters": params})
        for i, function_call in enumerate(function_calls):
            if type(function_call["parameters"]) == dict:
                function_call["parameters"] = str(function_call["parameters"])
                function_calls[i] = function_call

    for i, func_call in enumerate(function_calls):
        name = func_call["name"]
        params = func_call["parameters"]
        all_action[i] = ({"action": name, "action_input": params})

    
    
    return "", all_action 

def qwen_parser(result):
    if "<tool_call>" not in result:
        return result.replace("<|im_end|>", ""), {}
    all_action = {}
    
    result = result.replace("<|im_end|>", "")
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"

    matches = re.findall(pattern, result, re.DOTALL)
    text_content = re.sub(pattern, "", result, flags=re.DOTALL).strip()
    function_calls = []
    for func_call in matches:
        function_calls.append(eval(func_call))

    for i, func_call in enumerate(function_calls):
        name = func_call["name"]
        params = func_call["arguments"]
        all_action[i] = ({"action": name, "action_input": params})

    
    return text_content, all_action 


def llama_format_prompt(messages, function, timestamp=""):
    formatted_prompt = "<|begin_of_text|>"

    system_message = ""
    remaining_messages = messages
    if messages[0]["role"] == "system":
        system_message = messages[0]["content"].strip()
        remaining_messages = messages[1:]
    from datetime import datetime
    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
    timestamp = f"{timestamp.day} {timestamp.strftime('%b')} {timestamp.year}"
    timestamp = "26 Jul 2024" if timestamp == "" else timestamp 
    formatted_prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
    formatted_prompt += "Environment: ipython\n"
    formatted_prompt += "Cutting Knowledge Date: December 2023\n"
    formatted_prompt += f"Today Date: {timestamp}\n\n"
    formatted_prompt += system_message + "<|eot_id|>"

    # Llama pass in custom tools in first user message
    is_first_user_message = True
    for message in remaining_messages:
        if message["role"] == "user" and is_first_user_message:
            is_first_user_message = False
            formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
            formatted_prompt += "Given the following functions, please respond with a JSON for a function call "
            formatted_prompt += (
                "with its proper arguments that best answers the given prompt.\n\n"
            )
            formatted_prompt += 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
            formatted_prompt += "Do not use variables.\n\n"
            for func in function:
                formatted_prompt += json.dumps(func, indent=4) + "\n\n"
            formatted_prompt += f"{message['content'].strip()}<|eot_id|>"

        elif message["role"] == "tool":
            formatted_prompt += "<|start_header_id|>ipython<|end_header_id|>\n\n"
            if isinstance(message["content"], (dict, list)):
                formatted_prompt += json.dumps(message["content"])
            else:
                formatted_prompt += message["content"]
            formatted_prompt += "<|eot_id|>"

        else:
            formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return formatted_prompt


def watt_format_prompt(messages, function, timestamp=""):
    formatted_prompt = "<|begin_of_text|>"

    system_message = ""
    remaining_messages = messages
    if messages[0]["role"] == "system":
        system_message = messages[0]["content"].strip()
        remaining_messages = messages[1:]
    processed_function = json.dumps(function, indent=4)
    formatted_prompt += system_message 
    formatted_prompt += """
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
""".format(functions=processed_function)
    formatted_prompt += "<|eot_id|>"

    # Llama pass in custom tools in first user message
    is_first_user_message = True
    for message in remaining_messages:
        if message["role"] == "user" and is_first_user_message:
            is_first_user_message = False
            formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
            
            formatted_prompt += f"{message['content'].strip()}<|eot_id|>"

        elif message["role"] == "tool":
            formatted_prompt += "<|start_header_id|>ipython<|end_header_id|>\n\n"
            if isinstance(message["content"], (dict, list)):
                formatted_prompt += json.dumps(message["content"])
            else:
                formatted_prompt += message["content"]
            formatted_prompt += "<|eot_id|>"

        else:
            formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return formatted_prompt



def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch
    if 'torch_dtype' not in model_kwargs:
        torch_dtype = torch.float16
    else:
        torch_dtype = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float': torch.float,
            'auto': 'auto',
            'None': None,
        }.get(model_kwargs['torch_dtype'])
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    return model_kwargs

PARSER_DICT = {
    "llama": llama_parser,
    "watt": watt_parser,
    "qwen": qwen_parser,
}


def filter_llama(result):
    return result.replace("<|eot_id|>", "").replace("<|eom_id|>", "")

def filter_qwen(result):
    return result.replace("<|im_end|>", "")

 
FILTER_DICT = {
    "llama": filter_llama,
    "watt": filter_llama,
    "qwen": filter_qwen,
}

class OpenModel:
    def __init__(
        self,
        model_name: str,
        model_name_or_path: str,
        max_sequence_length: int = 8192,
        stop_words: list = [],
        use_vllm: bool = True,
        peft_path: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.max_sequence_length = max_sequence_length
        tokenizer_config = {
            "max_sequence_length": max_sequence_length
        }
        self.fc_tokenizer_template = None
        if 'llama' in model_name.lower():
            model_name='llama'
            self.vllmurl_model_name=os.environ.get("llama_model_name")
            self.fc_tokenizer_template = llama_format_prompt
        elif 'qwen' in model_name.lower():
            model_name='qwen'
            self.vllmurl_model_name=os.environ.get("qwen_model_name")
        elif 'watt' in model_name.lower():
            model_name='watt'
            self.vllmurl_model_name=os.environ.get("watt_model_name")
            self.fc_tokenizer_template = watt_format_prompt
        self.model_type = model_name
        self.filter=FILTER_DICT[model_name]
        self.parser=PARSER_DICT[model_name]
        self.use_vllm = use_vllm
        self._load_tokenizer(
            model_name_or_path, 
            tokenizer_config,
        )
        if self.use_vllm:
            openai_api_base = os.environ.get("url")
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=openai_api_base,
            )
        else:
            self._load_model(
                model_name_or_path,
                peft_path=peft_path
            )
            self.stop_words = list(set(stop_words + self._get_potential_stop_words(self.model_name)))
        self.conversation_history = []


    def _load_tokenizer(self, path: Optional[str], kwargs: dict):
        from transformers import AutoTokenizer

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs.update(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)


    def _load_model(self, path: str, kwargs: dict = {}, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False


    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words


    def prediction(self, method, timestamp, max_length: int = 16384, max_new_tokens: int = 1024, tool_list = None) -> str:
        if method == "react" or method == "e-react":
            inputs = [self.tokenizer.apply_chat_template(self.conversation_history, add_generation_prompt=True, tokenize=False)]
            
            if self.use_vllm:
                completion = self.client.completions.create(
                    model=self.vllmurl_model_name,
                    prompt = inputs[0],
                    max_tokens=max_length
                )
                prediction = completion.choices[0].text.strip()
            else:
                tokenize_kwargs = dict(
                    return_tensors='pt',
                    padding=False,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=max_length
                )
                tokens = self.tokenizer.batch_encode_plus(inputs, **tokenize_kwargs).to("cuda")
                output = self.model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=[self.tokenizer.convert_tokens_to_ids(word) for word in self.stop_words],
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=1,
                    top_p=1,
                )
                output = output[:, tokens['input_ids'].shape[1]:]
                response = self.tokenizer.batch_decode(output, skip_special_tokens=False)[0]
                prediction = response.strip()
            return prediction
        
        elif method == "fine-tuned":
            if self.fc_tokenizer_template is not None:
                inputs = [self.fc_tokenizer_template(self.conversation_history, function=tool_list, timestamp=timestamp)]
            else:
                inputs = [self.tokenizer.apply_chat_template(self.conversation_history, tools=tool_list, add_generation_prompt=True, tokenize=False)]
            
            if self.use_vllm:
                completion = self.client.completions.create(
                    model=self.vllmurl_model_name,
                    prompt = inputs[0],
                    max_tokens=max_length
                    )
                prediction = completion.choices[0].text
                if self.model_type == "llama":
                    if completion.choices[0].stop_reason == 128008:
                        prediction += "<|eom_id|>"
                    elif completion.choices[0].stop_reason == 128009 or completion.choices[0].stop_reason == None:
                        prediction += "<|eot_id|>"
            else:
                tokenize_kwargs = dict(
                    return_tensors='pt',
                    padding=False,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=max_length
                )
                tokens = self.tokenizer.batch_encode_plus(inputs, **tokenize_kwargs).to("cuda")
                
                output = self.model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=[self.tokenizer.convert_tokens_to_ids(word) for word in self.stop_words],
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=1,
                    top_p=1,
                )
                output = output[:, tokens['input_ids'].shape[1]:]
                
                response = self.tokenizer.batch_decode(output, skip_special_tokens=False)[0]
                prediction = response.strip()
            return prediction
        

    def add_message(self, message):
        self.conversation_history.append(message)


    def change_messages(self, messages):
        self.conversation_history = messages


    def display_conversation(self):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "observation": "magenta",
            "tool": "magenta",
            "function": "magenta"
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            
            print_obj = f"{message['role']}: {message['content']} "
            if "tool_calls" in message.keys():
                print_obj += str(message['tool_calls']) if message['role'] == 'assistant' and message['tool_calls'] is not [] else ''
            
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)


    def parse(self, method, tools_list=None, plan=False, timestamp=None, reasoning=False):
        predictions = self.prediction(method, tool_list=tools_list, timestamp=timestamp)
        print(f"Assistant: ------------------------\n{predictions}")
        logging.info(f"Assistant: ------------------------\n{predictions}")
        

        # react format prediction
        if method == 'react':
            if reasoning:
                pattern = r"<think>(.*?)</think>"
                match = re.search(pattern, predictions, re.DOTALL)
                if match:
                    think_content = match.group(1)
                    print(think_content)
                predictions = re.sub(pattern, "", predictions, flags=re.DOTALL).strip()
            predictions=self.filter(predictions)
            thought, action, action_input = react_parser(predictions)
            message = {
                "role": "assistant",
                "content": thought.strip(),
                "tool_calls": [{
                    "name": action.strip(),
                    "arguments": action_input.strip()
                }]
            }
            return predictions, message
        elif method == 'e-react':
            predictions=self.filter(predictions)
            if plan == True:
                return predictions
            
            thought, action, action_input = react_parser(predictions)
            message = {
                "role": "assistant",
                "content": thought.strip(),
                "tool_calls": [{
                    "name": action.strip(),
                    "arguments": action_input.strip()
                }]
            }
            return predictions, message
        elif method == 'fine-tuned':
            
            finish = True
            try:
                content, all_action = self.parser(predictions) # , timestamp
            except Exception as e:
                all_action = {}
                finish = False
                content = self.filter(predictions)
            if self.model_type in ["llama", "watt"]:
                if all_action == {}:
                    content = self.filter(predictions)
            return content, all_action, finish

            


class ChatGPT:
    def __init__(
        self,
        orgnization=None,
        mode="chat",
        model_name_or_path=None,
        **kwargs
    ) -> None:
        super().__init__()
        
        openai.api_key = os.environ.get("API_KEY") 
        if orgnization is not None:
            openai.organization = orgnization
        

        print(f"---------The tool using mode: {mode}----------")
        self.mode = mode
        self.model = model_name_or_path
        self.conversation_history = []


    def prediction(self, prompt: str, tools=None) -> str:
        
        error_times = 0
        if self.mode == "chat":
            while error_times < 20:
                try:
                    x = openai.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        max_tokens=1024,
                        temperature=0,
                        top_p=1,
                    )
                except Exception as e:
                    print(str(e))
                    error_times += 1
                    continue
                
                logging.info(x)
                return x.choices[0].message.content 
                error_times += 1 
        elif self.mode == "function_call":
            while error_times < 20:
                
                x = openai.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=1024,
                    tools=tools,
                    temperature=0,
                    top_p=1,
                )
                
                logging.info(x)
                if x.choices[0].finish_reason == 'stop':
                    return x.choices[0].message.content, x, {}
                message = x.choices[0].message

                all_action = {}
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_call_id = tool_call.id
                        action = tool_call.function.name
                        action_input = tool_call.function.arguments
                        all_action[tool_call_id] = {"action": action, "action_input": action_input}
                else:
                    action = ''
                    action_input = "{}"

                return x.choices[0].message.content, x, all_action
                error_times += 1


    def add_message(self, message):
        self.conversation_history.append(message)


    def change_messages(self, messages):
        self.conversation_history = messages
    

    def display_conversation(self):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "observation": "magenta",
            "tool": "magenta",
            "function": "magenta"
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if self.mode == "function_call":
                print_obj += str(message['tool_call_id']) if message['role'] == 'tool' else ''
                print_obj += str(message['tool_calls']) if message['role'] == 'assistant' and message['tool_calls'] is not None else ''
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)


    def parse(self, method, tools_list=None, plan=False, timestamp=None):
        self.time = time.time()
        
        if method == 'react':
            if self.mode != "chat":
                raise Exception("The mode must be 'chat' is using react format prompt")
            predictions = self.prediction(self.conversation_history)

            thought, action, action_input = react_parser(predictions)
            message = {
                "role": "assistant",
                "content": thought.strip(),
                "tool_calls": [{
                    "name": action.strip(),
                    "arguments": action_input.strip()
                }]
            }
            return predictions, message
        elif method == 'e-react':
            if self.mode != "chat":
                raise Exception("The mode must be 'chat' is using react format prompt")
            predictions = self.prediction(self.conversation_history)
            if plan == True:
                return predictions
            thought, action, action_input = react_parser(predictions)
            message = {
                "role": "assistant",
                "content": thought.strip(),
                "tool_calls": [{
                    "name": action.strip(),
                    "arguments": action_input.strip()
                }]
            }
            return predictions, message 
        elif method == 'fine-tuned':
            x, predictions, all_action = self.prediction(self.conversation_history, tools=tools_list)
            
            return predictions, all_action


from openai import OpenAI
class DeepSeek:
    def __init__(
        self,
        api_key=None,
        mode="chat",
        model_name_or_path=None,
        **kwargs
    ) -> None:
        super().__init__()
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        

        print(f"---------The tool using mode: {mode}----------")
        self.mode = mode
        self.model = model_name_or_path
        self.conversation_history = []


    def prediction(self, prompt: str, tools=None) -> str:
        error_times = 0
        if self.mode == "chat":
            while error_times < 20:
                
                try:
                    x = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        max_tokens=1024,
                        temperature=0,
                        top_p=1,
                    )
                except Exception as e:
                    print(str(e))
                    error_times += 1
                    continue
                
                logging.info(x)
                return x.choices[0].message.content, x 
                error_times += 1
        elif self.mode == "function_call":
            while error_times < 20:
                
                x = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=1024,
                    tools=tools,
                    temperature=0.0,
                    top_p=1,
                )
                
                logging.info(x)
                if x.choices[0].finish_reason == 'stop':
                    return x.choices[0].message.content, x, {}
                message = x.choices[0].message
                
                all_action = {}
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_call_id = tool_call.id
                        action = tool_call.function.name
                        action_input = tool_call.function.arguments
                        
                        
                        all_action[tool_call_id] = {"action": action, "action_input": action_input}
                        
                else:
                    action = ''
                    action_input = "{}"

                return x.choices[0].message.content, x, all_action
                error_times += 1
                


    def add_message(self, message):
        self.conversation_history.append(message)


    def change_messages(self, messages):
        self.conversation_history = messages
    

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "observation": "magenta",
            "tool": "magenta",
            "function": "magenta"
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:

            print_obj = f"{message['role']}: {message['content']} "
            if self.mode == "function_call":
                print_obj += str(message['tool_call_id']) if message['role'] == 'tool' else ''
                print_obj += str(message['tool_calls']) if message['role'] == 'assistant' and message['tool_calls'] is not None else ''

            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)


    def parse(self, method, tools_list=None, plan=False, reasoning=False, timestamp=None):
        self.time = time.time()
        
        if method == 'react':
            if self.mode != "chat":
                raise Exception("The mode must be 'chat' is using react format prompt")
            predictions, x = self.prediction(self.conversation_history)
            if reasoning:
                pattern = r"<think>(.*?)</think>"
                match = re.search(pattern, predictions, re.DOTALL)
                if match:
                    think_content = match.group(1)
                    print(think_content)
                predictions = re.sub(pattern, "", predictions, flags=re.DOTALL).strip()
            
            thought, action, action_input = react_parser(predictions)
            message = {
                "role": "assistant",
                "content": thought.strip(),
                "tool_calls": [{
                    "name": action.strip(),
                    "arguments": action_input.strip()
                }]
            }
            
            return predictions, message
        elif method == 'e-react':
            if self.mode != "chat":
                raise Exception("The mode must be 'chat' is using react format prompt")
            predictions, x = self.prediction(self.conversation_history)
            if plan == True:
                return predictions
            thought, action, action_input = react_parser(predictions)
            message = {
                "role": "assistant",
                "content": thought.strip(),
                "tool_calls": [{
                    "name": action.strip(),
                    "arguments": action_input.strip()
                }]
            }
            return predictions, message 
        elif method == 'fine-tuned':
            x, predictions, all_action = self.prediction(self.conversation_history, tools=tools_list)
            return predictions, all_action
        



     
        
    
MODELS = {
    "CHATGPT": ChatGPT,
    "OpenModel": OpenModel,
    "DeepSeek": DeepSeek
}   