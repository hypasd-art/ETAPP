from sentence_transformers import SentenceTransformer, util
import logging
from typing import Union
import os
import json
import importlib

from tqdm import tqdm

logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


class Tool_And_History_Searcher:
    def __init__(self, name: str):
        
        all_config_file = {}
        processed_all_config = []
        # import all the file in the toolkit folder, and load all the classes
        apis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        except_files = ['__init__.py', 'tool_manager.py', 'Toolsearcher', '__pycache__', "tool_list.json", "utils.py"]
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')) + "/./tool_list.json", "r") as f:
            tool_list = json.load(f)
        for folder in os.listdir(apis_dir):
            if folder not in except_files:
                for file in os.listdir(os.path.join(apis_dir, folder)):
                    if file != "__init__.py" and file != "config.json" and file != '__pycache__':
                        module = importlib.import_module("PLA.toolkit." + folder + "." + file.split('.')[0])
                        classes = [getattr(module, x) for x in dir(module) if isinstance(getattr(module, x), type)]
                        classes = [cls for cls in classes if cls.__name__ in tool_list.values()]
                        if len(classes) == 0:
                            continue
                        assert len(classes) == 1
                        cls = classes[0]
                        with open(os.path.join(apis_dir, folder, "config.json"), "r") as config_file:
                            cls_name = cls.__name__
                            all_config_file[cls_name] = {}
                            
                            config_dict = json.load(config_file)
                            for item in config_dict:
                                all_config_file[cls_name][item['function']['name']] = item

        self.name = name

        def api_summery(cls_name, fun_name, fun):
            
            cls_name = ''.join([' ' + i.lower() if i.isupper() else i for i in cls_name]).strip()
            return cls_name + "." + fun_name + ":" + fun["function"]['description'] 

        # Get the description parameter for each class
        for cls_name, cls_item in all_config_file.items():
            for fun_name, fun in cls_item.items():
                desc_for_search = api_summery(cls_name, fun_name, fun)
                fun['cls_name'] = cls_name
                fun['desc_for_search'] = desc_for_search
                processed_all_config.append(fun)
        self.apis = processed_all_config
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    
    

    def get_tool_doc(self, tools_name: Union[str, list]) -> dict:
        """
        Retrieves documentation for the specified tools.

        Args:
            tools_name: The name(s) of the tool(s) for which to retrieve documentation. Can be a single string or a list of strings.

        Returns:
            A dictionary containing the status of the operation and a dictionary of tool documentation, where each key is a tool name and each value is the corresponding tool's API documentation.
        """
        result = {}
        if type(tools_name) == str:
            tools_name = [tools_name]
        for api in self.apis:
            for tool_name in tools_name:
                if tool_name == api["function"]['name']:
                    result[tool_name] = api.copy()
                    result[tool_name].pop('desc_for_search')
                    result[tool_name].pop('cls_name')
        return {"status": "success", "data": result}
    
    def search_tools(self, keywords: Union[str, list], K: int = 3) -> dict:
        """
        Searches for relevant tools in various libraries based on the provided keywords.

        Args:
            keywords: The keywords to search for. Can be a single string or a list of strings.

        Returns:
            A dictionary containing the status of the operation, the input parameters, the list of matched tool names, and any exceptions that occurred.
        """
        input_parameters = {
            'keywords': keywords
        }
        all_tools_name = []
        if type(keywords) == str:
            keywords = [keywords]
        try:
            # best_match = self.best_match_api(keywords)
            for keyword in keywords:
                best_match = self.top_k_api(keyword, K)
                all_tools_name.extend(best_match)
            all_tools_name = list(set(all_tools_name))
            # print(best_match)
        except Exception as e:
            exception = str(e)
            return {'status': "failure", 'input': input_parameters, 'output': None, 'exception': exception}
        else:
            return {'status': "success", 'input': input_parameters, 'output': all_tools_name, 'exception': None}
    
    
    def best_match_api(self, keywords):
        kw_emb = self.model.encode(keywords)
        best_match = None
        best_match_score = 0
        for api in self.apis:
            re_emb = self.model.encode(api['desc_for_search'])
            cos_sim = util.cos_sim(kw_emb, re_emb).item()
            if cos_sim > best_match_score:
                best_match = api.copy()
                best_match_score = cos_sim
        best_match.pop('desc_for_search')
        return best_match
    
    def top_k_api(self, keywords, K=3):
        """
        Searches for the top K APIs that best match the given keywords.

        Args:
            keywords: The keywords to search for in the API descriptions.
            K: The number of top matching APIs to return. Defaults to 3.

        Returns:
            A list of the top K matching APIs, excluding their 'desc_for_search' field.
        """
        kw_emb = self.model.encode(keywords)
        top_matches = []  # Initialize a list to store the top matches

        # Iterate over each API and calculate the cosine similarity
        for api in self.apis:
            re_emb = self.model.encode(api['desc_for_search'])
            cos_sim = util.cos_sim(kw_emb, re_emb).item()
            
            # Insert the API and its score into the top_matches list
            # The list is kept sorted by score in descending order
            if len(top_matches) < K:
                top_matches.append({'api': api.copy(), 'score': cos_sim})
            else:
                # If the current API's score is higher than the lowest score in the list
                min_score_idx = top_matches.index(min(top_matches, key=lambda x: x['score']))
                if cos_sim > top_matches[min_score_idx]['score']:
                    top_matches[min_score_idx] = {'api': api.copy(), 'score': cos_sim}
            
            # Sort the list by score in descending order after each iteration
            top_matches.sort(key=lambda x: x['score'], reverse=True)

        # Remove the 'desc_for_search' field from each matched API
        for match in top_matches:
            match['api'].pop('desc_for_search')

        return [match['api']['function']['name'] for match in top_matches]
    
if __name__ == "__main__":
    toolsearcher = Tool_And_History_Searcher("John_Doe")
    
    print(toolsearcher.search_tools(["health status"]))
    print(toolsearcher.get_tool_doc(["get_recent_health_and_mood_summary", "get_user_recent_workout_records"]))