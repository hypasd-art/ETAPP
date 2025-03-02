import os
import json
from tqdm import tqdm 
import argparse
import logging
from PLA.Inference.prompts.generation_prompt import PROMPT_DICT
from PLA.Inference.prompts.generation_retrieval_prompt import RETRIEVAL_PROMPT_DICT
from PLA.Inference.prompts.generation_prompt import *
from PLA.Inference.prompts.generation_retrieval_prompt import *
from PLA.Inference.models import MODELS
from PLA.toolkit.tool_manager import ToolManager
from PLA.toolkit.Toolsearcher.Toolsearcher import Tool_And_History_Searcher
import openai
import re
import random
random.seed(1)
openai.api_key = os.environ.get('API_KEY')
random_index = [[] for i in range(16)]
random_index_t = []
for i in range(50):
    random_index_t.append(random.sample(range(16), 2))
for i in range(16):
    for j in range(50):
        for k in range(2):
            if random_index_t[j][k] == i:
                random_index[i].append(j)
print(random_index)
random_index = [[6, 8, 12, 13, 14, 17, 24, 43, 47], [5, 26, 33, 37, 44], [1, 25, 46], [2, 11, 25, 39], [0, 1, 9, 23, 29, 44], [20, 36, 40, 41, 46], [5, 7, 16, 24, 29, 35, 48, 49], [2, 3, 10, 18, 19, 20, 21, 22, 33, 34, 43, 47, 49], [17, 28, 40, 48], [0, 10, 26, 30, 32], [12, 15, 21, 27, 38, 45], [8, 34, 37, 42], [4, 7, 15, 18, 22, 32, 35, 39, 41, 45], [28, 31], [3, 9, 11, 14, 19, 23, 27, 38], [4, 6, 13, 16, 30, 31, 36, 42]]
transfor_dict = {'get_current_health_and_mood_status': 'health', 
                 'get_recent_health_and_mood_summary': 'health', 
                 'get_user_recent_workout_records': 'health', 
                 'add_event_in_calendar': 'calendar', 
                 'view_today_events_in_calendar': 'calendar', 
                 'view_events_in_calendar_by_providing_time_range': 'calendar', 
                 'delete_event_in_calendar': 'calendar', 
                 'add_alarm': 'calendar', 
                 'remove_alarm': 'calendar', 
                 'view_today_alarms': 'calendar', 
                 'add_product_to_cart': 'shopping', 
                 'remove_product_from_cart': 'shopping', 
                 'purchase_product_in_shopping_manager': 'shopping', 
                 'search_products_in_shopping_manager': 'shopping', 
                 'get_status_information_of_purchased_products': 'shopping', 
                 'view_cart_in_shopping_manager': 'shopping', 
                 'send_email': 'email', 
                 'get_today_emails_until_now': 'email', 
                 'search_email_by_sender_and_receiver': 'email', 
                 'search_email_by_content': 'email', 
                 'delete_email': 'email', 
                 'send_message': 'message', 
                 'get_unread_messages': 'message', 
                 'search_messages': 'message', 
                 'search_news_by_category': 'web_browsing', 
                 'search_heat_news': 'web_browsing', 
                 'search_from_wikipedia': 'web_browsing',
                 'play_music': 'music', 
                 'search_music_by_name': 'music', 
                 'get_music_list_in_favorites': 'music', 
                 'find_accommodations': 'navigation', 
                 'find_attractions': 'navigation', 
                 'find_restaurants': 'navigation', 
                 'find_flight': 'navigation', 
                 'set_temperature_and_humidity_in_home': 'smart_home_devices', 
                 'get_home_temperature_and_humidity': 'smart_home_devices', 
                 'control_light_in_home': 'smart_home_devices', 
                 'get_lighting_status_in_home': 'smart_home_devices', 
                 'control_curtains_in_home': 'smart_home_devices', 
                 'control_bathtub_in_home': 'smart_home_devices', 
                 'boil_water_in_home': 'smart_home_devices',
                 'get_today_weather': '',
                 'get_future_weather': '',
                 'get_weather_for_current_hour': ''}

def get_tool_preferences(result, profile_dict):

    tools_name = result["data"].keys()
    preferences = []
    already_add = []
    for tool_name in tools_name:
        if transfor_dict[tool_name] != "" and transfor_dict[tool_name] not in already_add:
            already_add.append(transfor_dict[tool_name])
            preferences.append(profile_dict[transfor_dict[tool_name]])
    return str(result) + "\nCorresponding preferences related to the tools you retrieve:\n" + str(preferences)
    
def method_converter(model, method, params, tool_manager, tools_list, profile_dict, available_tool_names=None, timestamp=None):
    conversation_history = []
    if method == 'react':

        finish = False
        turn = 0
        while not finish:
            turn += 1
            print('-' * 30 + 'Turn ' + str(turn) + '-' * 30)
            logging.info('-' * 30 + 'Turn ' + str(turn) + '-' * 30)
            try:
                if params["reasoning"]:
                    assert params["model_type"] == "OpenModel" or params["model_name_or_path"] == "deepseek-r1"
                    result, parse_result = model.parse(method=method, reasoning=True, timestamp=timestamp)
                else:
                    result, parse_result = model.parse(method=method, timestamp=timestamp)
            except Exception as e:
                assert "Please reduce the length of the" in str(e)
                
                conversation_history.append(
                        {"role": "assistant", "content": "Cannot solve this task within specified token .", "tool_calls": []}
                    )
                return conversation_history
            model.add_message({"role": "assistant", "content": result})
            conversation_history.append(parse_result)
            if params['max_turn'] >= 0 and turn < params['max_turn']:
                assert len(parse_result["tool_calls"]) == 1
                action = parse_result["tool_calls"][0]["name"]
                action_input = parse_result["tool_calls"][0]["arguments"]

                if action == "":
                    model.add_message(
                        {
                            "role": "user",
                            "content": "You did not call the tool to solve the problem or give the final answer. Please check your output format and content, and regenerate the answer based on the current status.\nIf you want to finish and give the final answer, generate in the following format: 'Thought: I now know the final answer\nFinal Answer: Once sufficient information or progress has been made, conclude with a clear and concise response that addresses the original question or task.'",
                        }
                    )
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": "You did not call the tool to solve the problem or give the final answer. Please check your output format and content, and regenerate the answer based on the current status.\nIf you want to finish and give the final answer, generate in the following format: 'Thought: I now know the final answer\nFinal Answer: Once sufficient information or progress has been made, conclude with a clear and concise response that addresses the original question or task.'",
                        }
                    )
                    continue
                if action.lower() == 'finish':
                    finish = True
                    continue
                    
                action_input = action_input.replace("null", "'null'")
                try:
                    param_dict = eval(action_input)
                    for key, value in param_dict.items():
                        if value == "null":
                            param_dict[key] = None
                except Exception as e:
                    model.add_message(
                        {
                        "role": "user", 
                        "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\nDesired format:\nThought: <The thought>\nAction: <The tool you take to>\nAction Input: <The parameters for the tool, you should provide a dict similar to {parameter_1: value_1, parameter_2: value 2} to call action.>\nPlease note that, You must only call one tool at a time. Wait for feedback from the tool before initiating any further actions or reasoning.",
                        }
                    )
                    conversation_history.append(
                        {
                        "role": "user", 
                        "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\nDesired format:\nThought: <The thought>\nAction: <The tool you take to>\nAction Input: <The parameters for the tool, you should provide a dict similar to {parameter_1: value_1, parameter_2: value 2} to call action.>\nPlease note that, You must only call one tool at a time. Wait for feedback from the tool before initiating any further actions or reasoning.",
                        }
                    )
                    continue
                try:
                    result = tool_manager.api_call(action, **param_dict)
                    
                    if action == "get_tool_doc" and params["use_retrieval"] == True:
                        output = get_tool_preferences(result, profile_dict)
                        
                        model.add_message(
                            {
                                "role": "user",
                                "content": output,
                            }
                        )
                        conversation_history.append(
                            {
                                "role": "tool",
                                "content": output,
                            }
                        )

                    else:
                        result = json.dumps(result, ensure_ascii=False)
                        if len(result) > params['max_observation_length']:
                            result = result[:params['max_observation_length']]
                        model.add_message(
                            {
                                "role": "user",
                                "content": result,
                            }
                        )
                        conversation_history.append(
                            {
                                "role": "tool",
                                "content": result,
                            }
                        )
                    
                except Exception as e:
                    # having error when calling api
                    model.add_message(
                        {
                            "role": "user", 
                            "content": str(e),
                        }
                    )
                    conversation_history.append(
                        {
                            "role": "tool",
                            "content": str(e)
                        }
                    )

                
                if turn > params['max_turn'] and not finish:
                    model.add_message(
                        {"role": "assistant", "content": "Thought: I cannot solve this task due to trying too many times.\n\nFinal Answer: I cannot handle the task."})
                    conversation_history.append(
                        {"role": "assistant", "content": "Cannot solve this task due to trying too many times.", "tool_calls": []}
                    )
                    finish = True
            else:
                model.add_message(
                    {"role": "assistant", "content": "Thought: I cannot solve this task due to trying too many times.\n\nFinal Answer: I cannot handle the task."})
                conversation_history.append(
                    {"role": "assistant", "content": "Cannot solve this task due to trying too many times.", "tool_calls": []}
                )
                finish = True
        model.display_conversation()
        print("\n"*3)
        
    elif method == 'e-react':
        try:
            result = model.parse(method, plan=True, timestamp=timestamp)
        except Exception as e:
            logging.info(str(e))
            assert "Please reduce the length of the" in str(e)
            
            conversation_history.append(
                    {"role": "assistant", "content": "Cannot solve this task within specified token .", "tool_calls": []}
                )
            return conversation_history
        model.add_message({"role": "assistant", "content": result})
        conversation_history.append(
            {
                "role": "assistant",
                "content": result,
                "tool_calls": []
            }
        )
        if params["add_example"] == True:
            if params["use_retrieval"] == True:
                model.add_message({'role': 'user', 'content': EXEC_PROMPT.format(available_tool_names=available_tool_names) + RETRIEVAL_REACT_ONESHOT_EXAMPLE})
                conversation_history.append(
                    {'role': 'user', 'content': EXEC_PROMPT.format(available_tool_names=available_tool_names) + RETRIEVAL_REACT_ONESHOT_EXAMPLE}
                )
            else:
                model.add_message({'role': 'user', 'content': EXEC_PROMPT.format(available_tool_names=available_tool_names) + REACT_ONESHOT_EXAMPLE})
                conversation_history.append(
                    {'role': 'user', 'content': EXEC_PROMPT.format(available_tool_names=available_tool_names) + REACT_ONESHOT_EXAMPLE}
                )
        else:
            model.add_message({'role': 'user', 'content': EXEC_PROMPT.format(available_tool_names=available_tool_names)})
            conversation_history.append(
                {'role': 'user', 'content': EXEC_PROMPT.format(available_tool_names=available_tool_names)}
            )
        finish = False
        turn = 0
        while not finish:
            turn += 1
            print('-' * 30 + 'Turn ' + str(turn) + '-' * 30)
            logging.info('-' * 30 + 'Turn ' + str(turn) + '-' * 30)

            result, parse_result = model.parse(method=method, timestamp=timestamp)

            model.add_message({"role": "assistant", "content": result})
            conversation_history.append(parse_result)
            if params['max_turn'] >= 0 and turn < params['max_turn']:
                assert len(parse_result["tool_calls"]) == 1
                action = parse_result["tool_calls"][0]["name"]
                action_input = parse_result["tool_calls"][0]["arguments"]

                if action == "":
                    model.add_message(
                        {
                            "role": "user",
                            "content": "You did not call the tool to solve the problem or give the final answer. Please check your output format and content, and regenerate the answer based on the current status.\nIf you want to finish and give the final answer, generate in the following format: 'Thought: I now know the final answer\nFinal Answer: Once sufficient information or progress has been made, conclude with a clear and concise response that addresses the original question or task.'",
                        }
                    )
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": "You did not call the tool to solve the problem or give the final answer. Please check your output format and content, and regenerate the answer based on the current status.\nIf you want to finish and give the final answer, generate in the following format: 'Thought: I now know the final answer\nFinal Answer: Once sufficient information or progress has been made, conclude with a clear and concise response that addresses the original question or task.'",
                        }
                    )
                    continue
                if action.lower() == 'finish':
                    finish = True
                    continue

                action_input = action_input.replace("null", "'null'")
                try: 
                    param_dict = eval(action_input)
                    for key, value in param_dict.items():
                        if value == "null":
                            param_dict[key] = None
                except Exception as e:
                    model.add_message(
                        {
                        "role": "user", 
                        "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\nDesired format:\nThought: <The thought>\nAction: <The tool you take to>\nAction Input: <The parameters for the tool, you should provide a dict similar to {parameter_1: value_1, parameter_2: value 2} to call action.>",
                        }
                    )
                    conversation_history.append(
                        {
                        "role": "user", 
                        "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\nDesired format:\nThought: <The thought>\nAction: <The tool you take to>\nAction Input: <The parameters for the tool, you should provide a dict similar to {parameter_1: value_1, parameter_2: value 2} to call action.>",
                        }
                    )
                    continue
                    
                try:
                    result = tool_manager.api_call(action, **param_dict)
                    
                    if action == "get_tool_doc" and params["use_retrieval"] == True:
                        output = get_tool_preferences(result, profile_dict)
                        
                        model.add_message(
                            {
                                "role": "user",
                                "content": output,
                            }
                        )
                        conversation_history.append(
                            {
                                "role": "tool",
                                "content": output,
                            }
                        )

                    else:
                        result = json.dumps(result, ensure_ascii=False)
                        if len(result) > params['max_observation_length']:
                            result = result[1:params['max_observation_length']]
                        model.add_message(
                            {
                                "role": "user",
                                "content": result,
                            }
                        )
                        conversation_history.append(
                            {
                                "role": "tool",
                                "content": result,
                            }
                        )
                    
                except Exception as e:
                    # having error when calling api
                    model.add_message(
                        {
                            "role": "user", 
                            "content": str(e),
                        }
                    )
                    conversation_history.append(
                        {
                            "role": "tool", 
                            "content": str(e),
                        }
                    )

                
                if turn > params['max_turn'] and not finish:
                    model.add_message(
                        {"role": "assistant", "content": "Thought: I cannot solve this task due to trying too many times.\n\nFinal Answer: I cannot handle the task."})
                    conversation_history.append(
                        {"role": "assistant", "content": "Cannot solve this task due to trying too many times.", "tool_calls": []}
                    )
                    finish = True
            else:
                model.add_message(
                    {"role": "assistant", "content": "Thought: I cannot solve this task due to trying too many times.\n\nFinal Answer: I cannot handle the task."})
                conversation_history.append(
                    {"role": "assistant", "content": "Cannot solve this task due to trying too many times.", "tool_calls": []}
                )
                finish = True
        model.display_conversation()
        print("\n"*3)

    elif method == 'fine-tuned':
        if params['model_type'] == 'OpenModel':
            finish = False
            turn = 0
            while not finish:
                turn += 1
                print('-' * 30 + 'Turn ' + str(turn) + '-' * 30)
                logging.info('-' * 30 + 'Turn ' + str(turn) + '-' * 30)
                try:
                    predictions, all_action, whether_finish = model.parse(method=method, tools_list=tools_list, timestamp=timestamp)
                except Exception as e:
                    logging.info(str(e))
                    assert "Please reduce the length of the" in str(e)
                    # conversation_history.append({"role": "user", "content": "error when generating"})
                    conversation_history.append(
                            {"role": "assistant", "content": "Cannot solve this task within specified token .", "tool_calls": []}
                        )
                    return conversation_history
                if len(all_action) == 0 and whether_finish:
                    finish = True
                    model.add_message({"role": "assistant", "content": predictions, "tool_calls": [{"type": "function", "function": {"name": item["action"], "arguments": item["action_input"]}} for item in all_action.values()]})
                    conversation_history.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"name": "finish", "arguments": predictions}]
                        }
                    )
                elif len(all_action) == 0 and not whether_finish:
                    model.add_message({"role": "assistant", "content": predictions, "tool_calls": None})
                    conversation_history.append(
                        {
                            "role": "assistant",
                            "content": predictions,
                            "tool_calls": []
                        }
                    )
                    model.add_message(
                        {
                            "role": "user",
                            "content": "Tool calling format is wrong, please regenerate."
                        }
                    )
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": "Tool calling format is wrong, please regenerate."
                        }
                    )
                else:
                    model.add_message({"role": "assistant", "content": predictions, "tool_calls": [{"type": "function", "function": {"name": item["action"], "arguments": item["action_input"]}} for item in all_action.values()]})
                    conversation_history.append(
                        {
                            "role": "assistant",
                            "content": predictions,
                            "tool_calls": [{"name": item["action"], "arguments": str(item["action_input"])} for item in all_action.values()]
                        }
                    )
                
                
                if params['max_turn'] >= 0 and turn < params['max_turn']:

                    
                    for i, (id, item) in enumerate(all_action.items()):
                        action = item['action']
                        action_input = str(item['action_input'])
                        try:
                            param_dict = eval(action_input)
                        except Exception as e:
                            model.add_message(
                                {
                                "role": "tool", 
                                "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\n",
                                }
                            )
                            conversation_history.append(
                                {
                                "role": "tool", 
                                "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\n",
                                }
                            )
                            continue
                        try:
                            result = tool_manager.api_call(action, **param_dict) # handling the api calling observation
                            
                            if action == "get_tool_doc" and params['use_retrieval'] == True:
                                output = get_tool_preferences(result, profile_dict)
                        
                                model.add_message(
                                    {
                                        "role": "tool",
                                        "content": output,
                                    }
                                )
                                conversation_history.append(
                                    {
                                        "role": "tool",
                                        "content": output,
                                    }
                                )
                            else:
                                result = json.dumps(result, ensure_ascii=False)
                                if len(result) > params['max_observation_length']:
                                    result = result[:params['max_observation_length']]

                                model.add_message(
                                    {
                                        "role": "tool",
                                        "content": result,

                                    }
                                )
                                conversation_history.append(
                                    {
                                        "role": "tool",
                                        "content": result,

                                    }
                                )
                        # having error when calling api
                        except Exception as e:
                            model.add_message(
                                {
                                    "role": "tool", 
                                    "content": str(e),
                                }
                            )
                            conversation_history.append(
                                {
                                    "role": "tool", 
                                    "content": str(e),
                                }
                            )
                        
                    if turn > params['max_turn'] and not finish:
                        model.add_message(
                            {"role": "assistant", "content": "Stop generation as the current step exceeds max turn."})
                        conversation_history.append(
                            {"role": "assistant", "content": "Cannot solve this task due to trying too many times."}
                        )
                        finish = True
                else:
                    model.add_message(
                        {"role": "assistant", "content": "Stop generation as the current step exceeds max turn."})
                    conversation_history.append(
                        {"role": "assistant", "content": "Cannot solve this task due to trying too many times.", "tool_calls": []}
                    )
                    finish = True
            model.display_conversation()
            print("\n"*3)
        else:
            assert model.mode == "function_call"
            finish = False
            turn = 0
            total_tool_calls = 0
            while not finish:
                turn += 1
                print('-' * 30 + 'Turn ' + str(turn) + '-' * 30)
                logging.info('-' * 30 + 'Turn ' + str(turn) + '-' * 30)
                
                predictions, all_action = model.parse(method=method, tools_list=tools_list, timestamp=timestamp)
                
                if predictions.choices[0].finish_reason == 'tool_calls' or (predictions.choices[0].finish_reason == 'length' and predictions.choices[0].message.tool_calls is not None):
                    total_tool_calls += len(predictions.choices[0].message.tool_calls)
                    if len(predictions.choices[0].message.tool_calls) > params['max_parallel_calls']:
                        conversation_history.append(
                            {"role": "assistant", "content": "Cannot solve this task due to calling superabundant tools.", "tool_calls": []}
                        )
                        return conversation_history
                    if total_tool_calls > params['max_tool_calls']:
                        conversation_history.append(
                            {"role": "assistant", "content": "Cannot solve this task due to calling superabundant tools.", "tool_calls": []}
                        )
                        return conversation_history
                if predictions.usage.total_tokens > params['total_max_tokens']:
                    conversation_history.append(
                        {"role": "assistant", "content": "Cannot solve this task within specified token .", "tool_calls": []}
                    )
                    return conversation_history

                
                model.add_message({"role": "assistant", "content": predictions.choices[0].message.content, "tool_calls": predictions.choices[0].message.tool_calls})
                if predictions.choices[0].finish_reason == 'tool_calls' or (predictions.choices[0].finish_reason == 'length' and predictions.choices[0].message.tool_calls is not None):
                    tool_calls = []
                    for tool_call in predictions.choices[0].message.tool_calls:
                        tool_calls.append({"name": tool_call.function.name, "arguments": tool_call.function.arguments})
                else:
                    tool_calls = None
                if predictions.choices[0].finish_reason != 'stop':
                    predict_content = predictions.choices[0].message.content if predictions.choices[0].message.content is not None else ""
                    conversation_history.append({"role": "assistant", "content": predict_content, "tool_calls": tool_calls})
                elif predictions.choices[0].finish_reason == 'stop':
                    predict_content = predictions.choices[0].message.content if predictions.choices[0].message.content is not None else ""
                    conversation_history.append({"role": "assistant", "content": "", "tool_calls": [{"name": "finish", "arguments": predict_content}]})
                    finish = True
                

                if params['max_turn'] >= 0 and turn < params['max_turn']:
                    if params['model_type'] == "Claude":
                        response = predictions.content[0].text
                    else:
                        response = predictions.choices[0]
                    
                    for i, (id, item) in enumerate(all_action.items()):
                        action = item['action']
                        action_input = item['action_input']
                        try:
                            param_dict = eval(action_input)
                        except Exception as e:
                            model.add_message(
                                {
                                "role": "tool", 
                                "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\n",
                                "tool_call_id": response.message.tool_calls[i].id
                                }
                            )
                            conversation_history.append(
                                {
                                "role": "tool", 
                                "content": "Your output format does not match the specified format, please revise your answer to follow the desired format.\n\n",
                                }
                            )
                            continue
                        try:
                            result = tool_manager.api_call(action, **param_dict) # handling the api calling observation
                            
                            if action == "get_tool_doc" and params["use_retrieval"] == True:
                                output = get_tool_preferences(result, profile_dict)
                                
                                model.add_message(
                                    {
                                        "role": "tool",
                                        "content": output,
                                        "tool_call_id": response.message.tool_calls[i].id
                                    }
                                )
                                conversation_history.append(
                                    {
                                        "role": "tool",
                                        "content": output,
                                    }
                                )
                            else:
                                result = json.dumps(result, ensure_ascii=False)
                                if len(result) > params['max_observation_length']:
                                    result = result[:params['max_observation_length']]

                                model.add_message(
                                    {
                                        "role": "tool",
                                        "content": result,
                                        "tool_call_id": response.message.tool_calls[i].id

                                    }
                                )
                                conversation_history.append(
                                    {
                                        "role": "tool",
                                        "content": result,
                                    }
                                )
                        # having error when calling api
                        except Exception as e:
                            model.add_message(
                                {
                                    "role": "tool", 
                                    "content": str(e),
                                    "tool_call_id": response.message.tool_calls[i].id # ']
                                }
                            )
                            conversation_history.append(
                                {
                                    "role": "tool", 
                                    "content": str(e),
                                }
                            )
                        
                    if turn > params['max_turn'] and not finish:
                        model.add_message(
                            {"role": "assistant", "content": "Stop generation as the current step exceeds max turn.", "tool_calls": None})
                        conversation_history.append(
                            {"role": "assistant", "content": "Cannot solve this task due to trying too many times.", "tool_calls": []}
                        )
                        finish = True
                else:
                    model.add_message(
                        {"role": "assistant", "content": "Stop generation as the current step exceeds max turn.", "tool_calls": None})
                    conversation_history.append(
                        {"role": "assistant", "content": "Cannot solve this task due to trying too many times.", "tool_calls": []}
                    )
                    finish = True
            model.display_conversation()
            print("\n"*3)

    return conversation_history 
        

def predict_instruction(system_prompt, user_prompt, params, item, model, user_profile, user_name, tool_manager, profile_dict, position):
    instruction = item['query']
    timestamp = item['timestamp']
    user_status = {"user location": position + ":" + item["location"], "time": item["timestamp"]}
    print("-" * 30 + "CURRENT_DATE: " + timestamp.split(" ")[0] + "-" * 30)
    print("-" * 30 + "CURRENT_TIME: " + timestamp + "-" * 30)
    os.environ['CURRENT_DATE'] = timestamp.split(" ")[0]
    os.environ['CURRENT_TIME'] = timestamp
    os.environ['max_observation_length'] = str(params['max_observation_length'])


    if params["use_retrieval"] == True:
        retriever = Tool_And_History_Searcher(user_name)
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_tools",
                    "description": "Searches for relevant tools in various libraries based on the provided keywords.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": [
                                    "object",
                                    "string"
                                ],
                                "description": "The keywords to search for. Can be a single string or a list of strings."
                            }
                        },
                        "required": [
                            "keywords"
                        ]
                    },
                    "return": {
                        "type": "object",
                        "description": "A dictionary containing the status of the operation, the input parameters, the list of matched tool names, and any exceptions that occurred."
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_tool_doc",
                    "description": "Retrieves documentation for the specified tools.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tools_name": {
                                "type": [
                                    "object",
                                    "string"
                                ],
                                "description": "The name(s) of the tool(s) for which to retrieve documentation. Can be a single string or a list of strings."
                            }
                        },
                        "required": [
                            "tools_name"
                        ]
                    },
                    "return": {
                        "type": "object",
                        "description": "A dictionary containing the status of the operation and a dictionary of tool documentation, where each key is a tool name and each value is the corresponding tool's API documentation."
                    }
                }
            }
        ]
        available_tool_names = ""
        available_tool_descriptions = str(available_tools)

        system_prompt_filled = system_prompt.format(user_profile=user_profile, 
                                                    user_status = user_status, 
                                                    available_tool_descriptions = available_tool_descriptions
                                                    )
        user_prompt_filled = user_prompt.format( 
                                                query = instruction,
                                                )
    else:
        retriever = None
        preferences = []
        available_tool_names = item["available_tools_name"]
        available_tools = []
        already_add = []
        for tool_name in available_tool_names:
            api = tool_manager.get_api_by_name(tool_name)
            available_tools.append(api['config'])
            if transfor_dict[tool_name] != "" and transfor_dict[tool_name] not in already_add:
                already_add.append(transfor_dict[tool_name])
                preferences.append(profile_dict[transfor_dict[tool_name]])
        available_tool_names = ", ".join(available_tool_names)
        available_tool_descriptions = str(available_tools)
        print("---------------------------------------------available tools---------------------------------------------------")
        for item in available_tools:
            print(item)

        system_prompt_filled = system_prompt.format(user_profile=user_profile, 
                                                    preferences=preferences,
                                                    user_status = user_status, 
                                                    available_tool_descriptions = available_tool_descriptions, 
                                                    available_tool_names = available_tool_names
                                                    )
        user_prompt_filled = user_prompt.format( 
                                                query = instruction,
                                                )
    
    
    model.change_messages([])
    if params["model_type"] == "CHATGPT":
        if params["method"] == "react":
            if params["add_example"] == True:
                if params["use_retrieval"]:
                    model.add_message({"role": "user", "content": system_prompt_filled + RETRIEVAL_REACT_ONESHOT_EXAMPLE + "\nThe query is:\n" + user_prompt_filled})
                else:
                    model.add_message({"role": "user", "content": system_prompt_filled + REACT_ONESHOT_EXAMPLE + "\nThe query is:\n" + user_prompt_filled})
                
            
            else:
                model.add_message({"role": "user", "content": system_prompt_filled + "\nThe query is:\n" + user_prompt_filled})
    else:
        if params["method"] == "react":
            if params["add_example"] == True:
                if params["use_retrieval"]:
                    model.add_message({"role": "system", "content": system_prompt_filled + RETRIEVAL_REACT_ONESHOT_EXAMPLE})
                else:
                    model.add_message({"role": "system", "content": system_prompt_filled + REACT_ONESHOT_EXAMPLE})
            
            else:
                model.add_message({"role": "system", "content": system_prompt_filled})
        else:
            model.add_message({"role": "system", "content": system_prompt_filled})
    

    query_result = [{"role": "system", "content": system_prompt_filled},
                    {"role": "user", "content": user_prompt_filled}]
    result = method_converter(
        model=model,
        method=params['method'],
        params=params,
        tool_manager=tool_manager,
        tools_list=available_tools,
        profile_dict = profile_dict,
        available_tool_names=available_tool_names,
        timestamp=timestamp
    )
    complete_result = query_result + result
    return complete_result, available_tools # , message
        
def main(params):
    logging.info(params)
    model = MODELS[params['model_type']](**params)
    if params["use_retrieval"]:
        system_prompt, user_prompt = RETRIEVAL_PROMPT_DICT[params['method']]
    else:
        system_prompt, user_prompt = PROMPT_DICT[params['method']]
    output_dir = params["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    instruction_data_dir = '../data/instruction'
    
    with open(params['profile_file'], 'r') as profile_f:
        profiles = json.load(profile_f)

    for i, (person, profile) in enumerate(profiles.items()):
        file_name = f"instruction.json"
        print(f"--------------testing {person}----------------")
        person_name = person.replace(" ", "_")

        
        with open(os.path.join("../profile/concrete_profile", f"profile_{person_name}.json"), "r") as f:
            profile_dict = json.load(f)

        logging.info("loading ToolManager")
        tool_manager = ToolManager(person_name)
        logging.info("ToolManager loaded successfully")
        instruction_file = os.path.join(instruction_data_dir, file_name)
        
        output_file = os.path.join(output_dir, person_name + "_" + file_name)
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.close()
        
        with open(instruction_file, 'r') as f:
            data = json.load(f)  

        already_output = []
        position = profile['DemographicData']['BasicInformation']['Location']
        for id, e_data in enumerate(tqdm(data)):
            print(f"process[{id}] doing task {id}/{len(data)}; instruction: {e_data['query']}")
            if id not in random_index[i]:
                continue
            logging.info(f"process[{id}] doing task {id}/{len(data)}; instruction: {e_data['query']}")

            output, tools = predict_instruction(system_prompt, user_prompt, params, e_data, model, profile, person_name, tool_manager, profile_dict, position)
            e_data['output'] = output
            e_data['tools'] = tools
            
            already_output.append(e_data)

        with open(output_file, 'w') as f:
            f.write(json.dumps(already_output, indent=4))
            f.close()
        # breakpoint()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="Type of the model to evaluate")
    
    parser.add_argument("--model_name", help="Name of the model to evaluate", default='')
    parser.add_argument("--base_name_or_path", help="Name of the model to evaluate", default='')
    parser.add_argument("--model_name_or_path", help="Name of the model to evaluate", default='')
    parser.add_argument("--max_sequence_length", help="The max sequence length", type=int, default=8192)


    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting")
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type")
    parser.add_argument("--logging_dir", default="")
    parser.add_argument("--output_dir", default=lambda: f"")
    parser.add_argument("--method", default="general")
    parser.add_argument("--add_example", action="store_true")

    parser.add_argument("--reasoning", action="store_true")

    parser.add_argument("--profile_file", default="../profile/profiles.json")
    parser.add_argument("--mode", type=str, default="function_calling")
    parser.add_argument("--use_retrieval", action="store_true")
    parser.add_argument("--add_reminder", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument("--max_turn", type=int, default=3)
    parser.add_argument("--max_observation_length", type=int, default=8192) 

    parser.add_argument("--total_max_tokens", type=int, default=16384) 
    parser.add_argument("--max_parallel_calls", type=int, default=10)
    parser.add_argument("--max_tool_calls", type=int, default=50)


    args = parser.parse_args()
    if args.use_retrieval == True:
        args.output_dir = f"../output/prompted_{args.model_type}_{args.model_name}_{args.method}_retrieve"
        args.logging_dir = f"../output/prompted_{args.model_type}_{args.model_name}_{args.method}_retrieve/inference.log"
    else:
        args.output_dir = f"../output/prompted_{args.model_type}_{args.model_name}_{args.method}"
        args.logging_dir = f"../output/prompted_{args.model_type}_{args.model_name}_{args.method}/inference.log"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    params = args.__dict__

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=args.logging_dir, 
        filemode='w' 
    )
    # tool_manager = ToolManager()
    main(params)
