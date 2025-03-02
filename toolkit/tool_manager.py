import os
import json
import inspect

class ToolManager:
    def __init__(self, name: str) -> None:
        import importlib.util

        all_apis = []
        all_config_file = {}
        # import all the file in the toolkit folder, and load all the classes
        apis_dir = os.path.abspath(os.path.dirname(__file__))
        # print(apis_dir)
        except_files = ['__init__.py', 'tool_manager.py', '__pycache__', "tool_list.json", "utils.py", "Messaging_platform"]
        with open(os.path.dirname(__file__) + "/./tool_list.json", "r") as f:
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
                        # load config 
                        with open(os.path.join(apis_dir, folder, "config.json"), "r") as config_file:
                            cls_name = cls.__name__
                            all_config_file[cls_name] = {}
                            
                            config_dict = json.load(config_file)
                            for item in config_dict:
                                all_config_file[cls_name][item['function']['name']] = item

                        
                        
                        all_apis.append(cls)

        classes = all_apis
        self.name = name

        
        # Get the description parameter for each class
        apis = []
        nameprint = {}
        for cls in classes:
            if issubclass(cls, object) and cls is not object:
                cls_name = cls.__name__
                for api_name in all_config_file[cls_name].keys():
                    functions = [member for member in inspect.getmembers(cls, inspect.isfunction) if member[0] in all_config_file[cls_name].keys()]
                    if api_name in [x[0] for x in functions]:
                        print(f"{cls_name}--{api_name}")
                        nameprint[api_name] = ""
                        # Get all functions in the class
                        
                        cls_info = {
                            'name': self.name,
                            'class': cls,
                            'class_name': cls_name,
                            'api_name': api_name,
                            'description': all_config_file[cls_name][api_name]['function']['description'],
                            'config': all_config_file[cls_name][api_name],
                            'instance': cls(self.name),
                        }
                        
                        apis.append(cls_info)
            else:
                print("error")
        print(nameprint)
        print(f"{len(apis)} apis are loaded in total")
        self.apis = apis

    def get_api_by_name(self, name: str):
        """
        Gets the API with the given name.

        Parameters:
        - name (str): the name of the API to get.

        Returns:
        - api (dict): the API with the given name.
        """
        for api in self.apis:
            if api['api_name'] == name:
                return api
        # print(self.apis)
        raise Exception(f'invalid tool name {name}.')
    
    def get_api_description(self, name: str):
        """
        Gets the description of the API with the given name.

        Parameters:
        - name (str): the name of the API to get the description of.

        Returns:
        - desc (str): the description of the API with the given name.
        """
        api_info = self.get_api_by_name(name).copy()
        return api_info["description"]

    
    def process_parameters(self, tool_name: str, parameters: list):
        input_parameters = self.get_api_by_name(tool_name)['input_parameters'].values()
        assert len(parameters) == len(input_parameters), 'invalid number of parameters.'

        processed_parameters = []
        for this_para, input_para in zip(parameters, input_parameters):
            para_type = input_para['type']
            if para_type == 'int':
                assert this_para.isdigit(), 'invalid parameter type. parameter: {}'.format(this_para)
                processed_parameters.append(int(this_para))
            elif para_type == 'float':
                assert this_para.replace('.', '', 1).isdigit(), 'invalid parameter type.'
                processed_parameters.append(float(this_para))
            elif para_type == 'str':
                processed_parameters.append(this_para)
            else:
                raise Exception('invalid parameter type.')
        return processed_parameters
    
    def api_call(self, tool_name: str, **kwargs): 
        """
        Calls the API with the given name and parameters.
        """
        

        processed_parameters = kwargs # {}
        
        
        tool_instance = self.get_api_by_name(tool_name)['instance']
        if callable(getattr(tool_instance, tool_name)):
            result = getattr(tool_instance, tool_name)(**processed_parameters)
        else:
            result = f'{tool_name} is not a valid function'
        
        return result
    
    def command_line(self):
        """
        Starts the command line interface for the tool manager.
        """
        while True:
            command = input('Please enter the command for the tool you want to use {"name": "", "arguments": ""}: \n')
            if command == 'exit':
                break
            json_command = json.loads(command)
            name = json_command['name']
            arguments = json.loads(json_command['arguments'])
            result = self.api_call(name, **arguments)

            
            print(result)

    def list_all_apis(self):
        """
        Lists all the APIs.

        Returns:
        - apis (list): a list of all the APIs.
        """
        return [api['api_name'] for api in self.apis]
    
if __name__ == '__main__':
    with open("/data/yphao/llama/PLA/PLA/profile/profiles.json", 'r') as profile_f:
        profiles = json.load(profile_f)

    for i, (person, profile) in enumerate(profiles.items()):
        name = person.replace(" ", "_")
        tool_manager = ToolManager(name)
        tool_manager.command_line()