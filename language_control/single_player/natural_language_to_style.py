import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import GPT
import json
import random
import time
from prompt_en import prompt_natural_language_to_style_en

def word_to_reward_factor(result):
    factors = {
        'active_area_x': [0 ,0 ,0],
        'active_area_y': [0 ,0 ,0],
        'shot': [0, 0],
        'move': [0, 0, 0]
    }
    
    choice_to_idx = {
        "Front": 0,
        "Midfield": 1,
        "Back": 2,
        "Left": 0,
        "Center": 1,
        "Right": 2,
        "Run": 0,
        "Dribble": 1,
        "Goal Area": 0,
        "Penalty Area": 1,
    }
    
    for k, v in result.items():
        if k == 'Horizontal Position':
            for choice in v:
                if choice in ['Front', 'Midfield', 'Back']:
                    idx = choice_to_idx[choice]
                    factors['active_area_x'][idx] = 1
        elif k == 'Vertical Position':
            for choice in v:
                if choice in ['Left', 'Center', 'Right']:
                    idx = choice_to_idx[choice]
                    factors['active_area_y'][idx] = 1
        elif k == 'Shooting Position':
            for choice in v:
                if choice in ['Goal Area', 'Penalty Area']:
                    idx = choice_to_idx[choice]
                    factors['shot'][idx] = 1
        elif k == 'Movement Action':
            for choice in v:
                if choice in ['Run', 'Dribble']:
                    idx = choice_to_idx[choice]
                    factors['move'][idx] = 1
            
    return factors

        
def natural_language_to_style(instruction=""):
    # llm = Hunyuan()
    # human_prompt = prompt_natural_language_to_style_cn.format(instruction=instruction)
    
    llm = GPT()
    human_prompt = prompt_natural_language_to_style_en.format(instruction=instruction)
    
    json_data = {}
    messages = [{"role": "user", "content": human_prompt.content}]
    begin_time = time.time()
    while True:
        try:
            response = llm.call(messages)
            break
        except Exception as e:
            print(f"SSL Error occurred: {e}. Retrying...")
    data = response['content']
    print(data)

    if data.startswith('```json') and data.endswith('```'):
        data = data[7:-3].strip()  
    else:
        data = data.strip()
    json_data = None
    try:
        json_data = json.loads(data)
        # print(json_data['result'])
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
    # print(json_data)
    end_time = time.time() - begin_time
    print('end_time:', end_time)
    
    if json_data:
        return json_data['result']
    else:
        return {}


def acc_check_en(nl_result, param):
    result = True
    param = param.split(',')
    instruction_type = 'shot' if param[2].strip(" ") == 'Shoot' else 'move'
    if "Horizontal Position" in nl_result or "Vertical Position" in nl_result and instruction_type == "move":
        move_instruction_keys = ['Horizontal Position', 'Vertical Position', 'Movement Action']
        if not all(element in nl_result.keys() for element in move_instruction_keys):
            return False
        move, area_x, area_y = param[0], param[1].strip(" "), param[2].strip(" ")
        if area_x != '' and nl_result['Horizontal Position'] != [area_x]:
            result = False
        if area_y != '' and nl_result['Vertical Position'] != [area_y]:
            result = False
        if move != '' and nl_result['Movement Action'] != [move]:
            result = False
        return result
    elif "Shooting Action" in nl_result and instruction_type == "shot":
        shot_instruction_keys = ['Shooting Position', 'Movement Action', 'Shooting Action']
        if not all(element in nl_result.keys() for element in shot_instruction_keys):
            return False
        move, shot_pos, shot_act = param[0], param[1].strip(" "), param[2].strip(" ")
        if shot_pos != '' and nl_result['Shooting Position'] != [shot_pos]:
            result = False
        if move != '' and nl_result['Movement Action'] != [move]:
            result = False
        if shot_act != '' and nl_result['Shooting Action'] != [shot_act]:
            result = False
        return result
    else:
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction_path', type=str, default='')
    args = parser.parse_args()
    instruction_path = args.instruction_path
    with open(instruction_path, 'rb') as f:
        nl_instruction = f.read()
    nl_instruction = json.loads(nl_instruction)

    
    def display_results(index, natural_language, nl_result):
        print(f"Index: {index}")
        print(f"Natural Language: {natural_language}")
        print(f"Natural Language Result: {nl_result}")
    
    acc_cnt = 0
    total_cnt = 0

    for instruction, v_dict in nl_instruction.items():
        for idx, nl in v_dict.items():
            nl_result = natural_language_to_style(nl)
            display_results(idx, nl, nl_result)
            acc_cnt += int(acc_check_en(nl_result, instruction))
            print('real insturction:', instruction)
            print("is acc:", int(acc_check_en(nl_result, instruction)))
            total_cnt += 1

    print("{}, acc_cnt: {}, total_cnt: {}, acc: {:.2f}".format(instruction_path, acc_cnt, total_cnt, acc_cnt/total_cnt))
