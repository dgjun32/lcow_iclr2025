# SFT dataset
import numpy as np 
import pandas as pd
import json
import pandas as pd
import transformers
from transformers import AutoTokenizer

from prompt_v2 import hf_format_rephrase_prompt
from utils import add_action_semantic


def create_sft_dataset(data):
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
    df = {'text': [], 
          'reward': [], 
          'reward_spec':[], 
          'input': [], 
          'output': [], 
          'task_id': [], 
          'step': [], 
          'history':[], 
          'gt_action':[]}
    
    sample_loc = []
    for sample in data:
        obs = sample['observation']
        action = sample['ref_action']
        history = sample['history']
        sem_action = add_action_semantic(action, obs)
        
        # get rep observation with maximum reward
        candidates = sample['candidates']
        possible = any(item['action_matching'] > 0.0 for item in candidates)
        if not possible:
            pass
        else:
            key = max(
                sample['candidates'], 
                key=lambda x: x['action_matching']
                )
            reasoning, extraction = key['reasoning'], key['extraction']
            rephrase_prompt, rephrase_system_prompt = hf_format_rephrase_prompt(
                sample['domain_info'],
                sample['goal'], 
                sample['history'],
                '\n'.join(sample['observation'].split('\n')[1:]).strip())
            if extraction == None:
                pass
            else:
                sample_loc.append(f'{sample["task_id"]}_{sample["task_timestep"]}')
                rep_obs = f'## Reasoning\n{reasoning}\n\n## Focused Observation\n{extraction}[END]'
                # configure the prompt and output
                chat = [
                    {'role': 'system', 'content': rephrase_system_prompt},
                    {'role': 'user', 'content': rephrase_prompt},
                    {'role': 'assistant', 'content': rep_obs} 
                ]
                text = tokenizer.apply_chat_template(chat, tokenize=False)
                df['text'].append(text)
                df['reward'].append(key['action_matching'])
                df['reward_spec'].append(key['action_matching_list'])
                df['input'].append(rephrase_prompt)
                df['output'].append(rep_obs)
                df['task_id'].append(sample['task_id'])
                df['step'].append(sample['task_timestep'])
                df['gt_action'].append(sem_action)
                df['history'].append(history)
    df = pd.DataFrame(df)
    
    return df


if __name__ == '__main__':
    with open('./webarena_data/sampled_contextualization.json', 'r') as f:
        data = json.load(f)
    df = create_sft_dataset(data)
    df.to_csv(f'./webarena_data/sft_train.csv')
    
