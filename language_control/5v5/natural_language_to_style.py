import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import GPT
import json
import time
import numpy as np
from  prompt_en import prompt_natural_language_to_style_en


def word_to_reward_factor(result):
    factors = {
        'win': 0.5,
        'goal': 0.5,
        'lose_goal': 0.5,
        'hold_ball': 0.5,
        'get_possession': 0.5,
        'pass': 0.5,
        'active_area_x': [0, 0, 0],
        'active_area_y': [0, 0, 0],
        'spacing': 0.5,
        'shot': [1, 1],
        'move': [1, 1, 1],
        'formation': 0.5,
    }
    
    id_2_factor = {
        "Style_1": 'win',
        "Style_2": 'goal',
        "Style_3": 'lose_goal',
        "Style_4": 'hold_ball',
        "Style_5": 'get_possession',
        "Style_6": 'pass',
        "Style_7": 'spacing',
        "Style_8": 'formation',
        "Style_9": 'shot',
        "Style_10": 'move'
    }
    
    for k, v in result.items():
        if k not in id_2_factor:
            continue
        if k in ['Style_9', 'Style_10']: 
            factor_name = id_2_factor[k]
            if v < len(factors[id_2_factor[k]]) and v >= 0:
                tmp_style = [0.0] * len(factors[id_2_factor[k]])
                tmp_style[v] = 1
                factors[factor_name] = tmp_style
        else:
            factor_name = id_2_factor[k]
            noise = 0 #  0.05 * np.random.randn()
            factor_value = np.clip(v / 10 + noise, 0, 1)
            factors[factor_name] = factor_value
       
    return factors

def natural_language_to_style(instruction=""):
    llm = GPT()
    json_data = {}
    human_prompt = prompt_natural_language_to_style_en.format(instruction=instruction)
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
    try:
        start_marker = "```json" 
        end_marker = "```"
        start_index = data.find(start_marker) + len(start_marker)
        end_index = data.find(end_marker, start_index)
        json_str = data[start_index:end_index].strip()
        json_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")

    end_time = time.time() - begin_time
    print('end_time:', end_time)
    
    if json_data:
        return json_data
    else:
        return {}

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
    
    for instruction, v_dict in nl_instruction.items():

        idx, nl = 0, v_dict['1']
        nl_result = natural_language_to_style(nl)
        factors = word_to_reward_factor(nl_result)
        display_results(idx, nl, nl_result)
        print('factors: ', factors)
        print('real insturction: ', instruction)
