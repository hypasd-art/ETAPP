import os
import json
from tqdm import tqdm 
import argparse
import logging
from PLA.evaluation.evaluate_prompt import *
from PLA.Inference.models import MODELS
import openai
import re
from termcolor import colored
from openai import OpenAI
import time

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






random_index = [[6, 8, 12, 13, 14, 17, 24, 43, 47], [5, 26, 33, 37, 44], [1, 25, 46], [2, 11, 25, 39], [0, 1, 9, 23, 29, 44], [20, 36, 40, 41, 46], [5, 7, 16, 24, 29, 35, 48, 49], [2, 3, 10, 18, 19, 20, 21, 22, 33, 34, 43, 47, 49], [17, 28, 40, 48], [0, 10, 26, 30, 32], [12, 15, 21, 27, 38, 45], [8, 34, 37, 42], [4, 7, 15, 18, 22, 32, 35, 39, 41, 45], [28, 31], [3, 9, 11, 14, 19, 23, 27, 38], [4, 6, 13, 16, 30, 31, 36, 42]]

def get_score(model_name, base_url=None, message = []):
    if model_name == "gpt-4o-2024-11-20" or model_name == "gpt-4o-mini":
        api_key = os.environ.get('API_KEY')
        client = OpenAI(api_key=api_key)
    x = client.chat.completions.create(
        model=model_name,
        messages=message,
        temperature=0.0,
        max_tokens=8192
    )
    logging.info(x)
    result = x.choices[0].message.content
    
    pattern = r'```json(.*?)```'
    match = re.search(pattern, result, re.DOTALL)

    if match:

        json_str = match.group(1).strip()
        try:
            
            evaluation_results = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            raise Exception
    else:
        print("No JSON block found.")
        raise Exception
    return evaluation_results

        
def evaluate_personalization_and_proactivity(instruction, profile, profile_dict, result, evaluation_models, plan=False, model="gpt-4o"):
    openai.api_key = os.environ.get('API_KEY')
    timestamp = result['timestamp']

    
    profile = json.dumps(profile)
    query = result['query']
    output = []
    if plan:
        preoutput = result['output'][4:]
    else:
        preoutput = result['output'][2:]
    
    for item_data in preoutput:
        if item_data["role"] == "assistant" and "tool_calls" not in item_data:
            output.append(item_data)
        elif item_data["role"] == "assistant" and type(item_data["tool_calls"]) == dict:
            assert item_data["tool_calls"]["name"] == "finish"
            output.append({"role": "assistant", "content": item_data["tool_calls"]["arguments"], "tool_calls": None})
        elif item_data["role"] == "assistant" and item_data["tool_calls"] == None:
            output.append(item_data)
        elif item_data["role"] == "assistant" and len(item_data["tool_calls"]) == 0:
            output.append({"role": "assistant", "content": item_data["content"], "tool_calls": None})
        elif item_data["role"] == "assistant" and item_data["tool_calls"][0]["name"] == "finish":
            output.append({"role": "assistant", "content": item_data["content"] + "\n" + item_data["tool_calls"][0]["arguments"], "tool_calls": None})
        else:
            output.append(item_data)
    
    
    output = [str(item) for item in output]
    output = "\n".join(output)

    keypoint_for_personal = result['keypoint for personal']
    keypoint_for_proactive = result['keypoint for proactive']
    personal = {}
    proactive = {}
    for item in keypoint_for_personal:
        personal[item] = {"analysis": "<You need to analyze whether the key point is met in the model's output. If it is met, clearly indicate which tools or parameters were used, or which specific aspect of the final answer satisfied the key point. If it is not met, analyze whether the key point does not need to be considered due to objective reasons (for example, there is no need to consider bringing an umbrella on a sunny day). >", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"}
        
    for item in keypoint_for_proactive:
        proactive[item] = {"analysis": "<You need to analyze whether the key point is met in the model's output. If it is met, clearly indicate which tools or parameters were used, or which specific aspect of the final answer satisfied the key point. If it is not met, analyze whether the key point does not need to be considered due to objective reasons (for example, there is no need to consider bringing an umbrella on a sunny day).>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"}
        
    
    output_format = {
        "Procedure":{
            "Keypoints for Procedure": {
                "Completeness": {"analysis": "<Analyze whether the response fully addresses the query>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"},
                "Avoid Unneccessary Action": {"analysis": "<Evaluate if any redundant or irrelevant actions are present>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"},
                "Call the tool accurate": {"analysis": "<Check if each tool call is in the correct format and necessary>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"},
                "Summary the query clearly and comperhensive": {"analysis": "<Assess if the query has been summarized accurately and fully>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"}
            },
            "Final Assessment": {"anlysis": "Analyze the score of Procedure", "score": "<A score from 0 to 5 based on the Procedure criteria>"}
        },
        "Personalization": {
            "Keypoints for Personalization": personal,
            "Final Assessment": {"analysis": "Analyze the score of Personalization", "score": "<A score from 0 to 5 based on the Personalization criteria>"}
        },
        "Proactivity": {
            "Keypoints for Proactivity": proactive,
            "Final Assessment": {"Analysis": "Analyze the score of the Proative", "score": "<A score from 0 to 5 based on the Proactive criteria>"}
        }
    }
    output_format = json.dumps(output_format, indent=4)

    message = [{"role": "user", "content": EVALUATE_PERSONALIZATION_AND_PROACTIVITY.format(profile=profile + "\n" + str(profile_dict), query=query, personal=personal, proactive=proactive, output_format=output_format, keypoint_for_personal=keypoint_for_personal, keypoint_for_proactive=keypoint_for_proactive, output=str(output))}]
    
    logging.info(message[0]["content"])
    scores = {}
    for model_name in evaluation_models:
        error_time = 0
        while error_time < 3:
            try:
                evaluation_results = get_score(model_name=model_name, message=message)
                break
            except Exception as e:
                print(e)
                error_time += 1
                
        scores[model_name] = {}
        scores[model_name]["Procedure"] = int(evaluation_results["Procedure"]["Final Assessment"]["score"])
        scores[model_name]["Personalization"] = int(evaluation_results["Personalization"]["Final Assessment"]["score"])
        scores[model_name]["Proactivity"] = int(evaluation_results["Proactivity"]["Final Assessment"]["score"])
        scores[model_name]["evaluation_result"] = evaluation_results
        

        logging.info(model_name)
        logging.info(str(scores[model_name]["Procedure"]) + " " + str(scores[model_name]["Personalization"]) + " " + str(scores[model_name]["Proactivity"]))
    
    
    return [result], scores


def main(params):
    with open(params['profile_file'], 'r') as profile_f:
        profiles = json.load(profile_f)
    all_score = {}
    evaluation_models = ["gpt-4o-2024-11-20"]
    for model in evaluation_models:
        all_score[model] = {'Procedure': 0, 'Personalization': 0, 'Proactivity': 0}
    all_evaluate_result = []
    num = 0
    with open("../data/instruction/instruction.json", 'r') as f:
        all_instruction = json.load(f)  
    plan = True if "e-react" in params['result_file'] else False
    
    len_already = len(all_evaluate_result)
    for i, (person, profile) in enumerate(profiles.items()):
        if i * 50 < len_already:
            # len_already -= 50
            continue
        print(i)
        person_name = person.replace(" ", "_")
        file_name = f"{person_name}_instruction.json"
        result_file = os.path.join(params['result_file'], file_name)

        with open(result_file, 'r') as f:
            model_result = json.load(f)

        with open(os.path.join("../profile/concrete_profile", f"profile_{person_name}.json"), "r") as f:
            profile_dict = json.load(f)
        
        
        for idx, result in enumerate(model_result):
            if i*50 + idx < len_already:
                print(idx)
                continue
            print(f"processing {person}, instruction id {random_index[i][idx]}")
            instruction = all_instruction[random_index[i][idx]]
            assert result["query"] == instruction["query"]
            preferences = []
            available_tool_names = result["available_tools_name"]
            already_add = []
            for tool_name in available_tool_names:
                if transfor_dict[tool_name] != "" and transfor_dict[tool_name] not in already_add:
                    already_add.append(transfor_dict[tool_name])
                    preferences.append(profile_dict[transfor_dict[tool_name]])
            used_tools = []
            for item in result["output"]:
                if item["role"] == "assistant":
                    if "tool_calls" not in item:
                        continue
                    if type(item["tool_calls"]) == list:
                        for item_tool in item["tool_calls"]:
                            
                            if item_tool["name"] in list(transfor_dict.keys()):
                                used_tools.append(item_tool["name"])
                    elif type(item["tool_calls"]) == dict:
                        item_tool = item["tool_calls"]
                        assert item_tool["name"] == "finish"
                        if item_tool["name"] in list(transfor_dict.keys()):
                            used_tools.append(item_tool["name"])
                    else:
                        continue
            
            for tool_name in used_tools:
                if transfor_dict[tool_name] != "" and transfor_dict[tool_name] not in already_add:
                    already_add.append(transfor_dict[tool_name])
                    preferences.append(profile_dict[transfor_dict[tool_name]])
            evaluate_result, score = evaluate_personalization_and_proactivity(instruction, profile=profile, profile_dict=preferences, result=result, evaluation_models=evaluation_models, plan=plan)
            num += 1
            for model in evaluation_models:
                all_score[model]['Procedure'] += score[model]['Procedure']
                all_score[model]['Personalization'] += score[model]['Personalization']
                all_score[model]['Proactivity'] += score[model]['Proactivity']
            all_evaluate_result.append({"query": params["result_file"] + "_" + person_name + "_" + result["query"], "evaluation_result": score})

        
        with open(params["evaluate_output_dir"], "w") as file:
            json.dump(all_evaluate_result, file, ensure_ascii=False, indent=4)
    for model in evaluation_models:
        all_score[model]['Procedure'] = all_score[model]['Procedure'] / num
        all_score[model]['Personalization'] = all_score[model]['Personalization'] / num
        all_score[model]['Proactivity'] = all_score[model]['Proactivity'] / num
    
    print("evaluating data num: {}".format(num))
    print(all_score)

    logging.info("evaluating data num: {}".format(num))
    logging.info(all_score)


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_file", default="../profile/profiles.json")
    parser.add_argument("--result_file", default="../output/prompted_CHATGPT_gpt-4o_react")


    args = parser.parse_args()
    args.evaluate_output_dir = args.result_file + f"/evaluate_result.json"
        
    params = args.__dict__
    c_time = time.time()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=args.result_file + f"/evaluate_result_logging_{c_time}.log", 
        filemode='w' 
    )
    
    main(params)