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
        'goal': 0,
        'hold_ball': 0,
        'pass': 0,
        'formation': 0,
        'shot': [0, 0],
        'move': [0, 0, 0]
    }
    
    choice_to_idx = {
        "Retreat": 0,
        "Balanced": 0.5,
        "Press": 1,
        "Run": 0,
        "Dribble": 1,
        "Goal Area": 0,
        "Penalty Area": 1,
    }
    
    for k, v in result.items():
        if k == 'Ball Control Action':
            if len(v) > 1:
                choice = random.choice(v)
            else:
                choice = v[0]
            if choice == "Pass Ball":
                factors['pass'] = 1
                factors['hold_ball'] = 0
            elif choice == "Carry Ball":
                factors['hold_ball'] = 1
        elif k == 'Formation':
            if len(v) > 1:
                choice = random.choice(v)
            else:
                choice = v[0]
            if choice in ['Retreat', 'Balanced', 'Press']:
                idx = choice_to_idx[choice]
                factors['formation'] = idx
        elif k == 'Shooting Position':
            if len(v) > 1:
                choice = random.choice(v)
            else:
                choice = v[0]
            if choice in ['Goal Area', 'Penalty Area']:
                idx = choice_to_idx[choice]
                factors['shot'][idx] = 1
                factors['goal'] = 1
        elif k == 'Movement Action':
            if len(v) > 1:
                choice = random.choice(v)
            else:
                choice = v[0]
            if choice in ['Run', 'Dribble']:
                idx = choice_to_idx[choice]
                factors['move'][idx] = 1
            
    return factors

def natural_language_to_style(instruction=""):
    llm = GPT()
    human_prompt = prompt_natural_language_to_style_en.format(instruction=instruction)
    messages = [{"role": "user", "content": human_prompt.content}]
    
    result_json = {}
    begin_time = time.time()
    while True:
        try:
            response = llm.call(messages)
            break
        except Exception as e:
            print(f"SSL Error occurred: {e}. Retrying...")
    text = response['content']
    print(text)

    try:
        result_index = text.find('"result":')
        if result_index == -1:
            raise ValueError("The 'result' field was not found.")

        result_text = text[result_index + len('"result":'):]

        start_brace_index = result_text.find('{')
        if start_brace_index == -1:
            raise ValueError("The '{' character was not found.")

        brace_count = 0
        end_index = None
        for i, char in enumerate(result_text[start_brace_index:], start=start_brace_index):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_index = i + 1
                    break

        if end_index is None:
            raise ValueError("A matching '}' character was not found.")

        result_json_str = result_text[start_brace_index:end_index]

        result_json = json.loads(result_json_str)

        print("The parsed 'result' field is: ")
        print(result_json)

    except Exception as e:
        print(f"Parsing error: {e}")
        
    return result_json


def acc_check_en(nl_result, param):
    result = True
    param = param.split(',')

    instruction_type = None
    if len(param) == 3:
        if param[2].strip(" ") == 'Shoot':
            instruction_type = 'shot' 
    elif len(param) == 2:
        instruction_type = 'ball control'
    if "Ball Control Action" in nl_result and instruction_type == "ball control":
        instruction_keys = ['Formation', 'Ball Control Action']
        if not all(element in nl_result.keys() for element in instruction_keys):
            return False
        formation, ball_control_action = param[0], param[1].strip(" ")
        if formation != '' and nl_result['Formation'] != [formation]:
            result = False
        if ball_control_action != '' and nl_result['Ball Control Action'] != [ball_control_action]:
            result = False
        return result
    elif "Shooting Action" in nl_result and instruction_type == "shot":
        instruction_keys = ['Movement Action', 'Shooting Position', 'Shooting Action']
        if not all(element in nl_result.keys() for element in instruction_keys):
            return False
        move, shot_pos, shot_act = param[0], param[1].strip(" "), param[2].strip(" ")
        if move != '' and nl_result['Movement Action'] != [move]:
            result = False
        if shot_pos != '' and nl_result['Shooting Position'] != [shot_pos]:
            result = False
        if shot_act != '' and nl_result['Shooting Action'] != [shot_act]:
            result = False
        return result
    else:
        return False


def acc_expectation_check_en(nl_result, param):
    result = 1
    param = param.split(',')

    instruction_type = None
    if len(param) == 3:
        if param[2].strip(" ") == 'Shoot':
            instruction_type = 'shot' 
    elif len(param) == 2:
        instruction_type = 'ball control'
    if "Ball Control Action" in nl_result and instruction_type == "ball control":
        instruction_keys = ['Formation', 'Ball Control Action']
        if not all(element in nl_result.keys() for element in instruction_keys):
            return 0
        formation, ball_control_action = param[0], param[1].strip(" ")
        if formation != '' and nl_result['Formation'] != [formation]:
            if formation in nl_result['Formation']:
                result *= 1 / len(nl_result['Formation'])
        if ball_control_action != '' and nl_result['Ball Control Action'] != [ball_control_action]:
            if ball_control_action in nl_result['Ball Control Action']:
                result *= 1 / len(nl_result['Ball Control Action'])
        return result
    elif "Shooting Action" in nl_result and instruction_type == "shot":
        instruction_keys = ['Movement Action', 'Shooting Position', 'Shooting Action']
        if not all(element in nl_result.keys() for element in instruction_keys):
            return False
        move, shot_pos, shot_act = param[0], param[1].strip(" "), param[2].strip(" ")
        if move != '' and nl_result['Movement Action'] != [move]:
            if move in nl_result['Movement Action']:
                result *= 1 / len(nl_result['Movement Action']) 
        if shot_pos != '' and nl_result['Shooting Position'] != [shot_pos]:
            if shot_pos in nl_result['Shooting Position']:
                result *= 1 / len(nl_result['Shooting Position'])
        if shot_act != '' and nl_result['Shooting Action'] != [shot_act]:
            if shot_act in nl_result['Shooting Action']:
                result *= 1 / len(nl_result['Shooting Action'])
        return result
    else:
        return 0

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
    instruction = list(nl_instruction.keys())[0]
    v_dict = nl_instruction[instruction]

    for idx, nl in v_dict.items():
        nl_result = natural_language_to_style(nl)
        display_results(idx, nl, nl_result)
        acc_cnt += int(acc_check_en(nl_result, instruction))
        print('real insturction:', instruction)
        print("is acc:", int(acc_check_en(nl_result, instruction)))
        total_cnt += 1

    print("{}, acc_cnt: {}, total_cnt: {}, acc: {:.2f}".format(instruction_path, acc_cnt, total_cnt, acc_cnt/total_cnt))