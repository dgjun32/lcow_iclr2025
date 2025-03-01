# SFT dataset

import pandas as pd
import json
import transformers 
from transformers import AutoTokenizer
import sys
from utils import return_meta_prompt_repr_3_hf
sys.path.append('./')



if __name__ == '__main__':
    with open('collected_data/rephrase_data_iter_0.json', 'r') as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    df = {'text': [], 
            'obs': [], 
            'rephrase':[], 
            'action':[],
            'reward':[]}

    for sample in data:
        # get src observation
        obs = sample['obs']
        prev_actions = sample['prev_actions']
        goal = sample['goal']
        action = sample['action']
        src_obs = obs.replace('Instruction:\n'+goal+'\n', '')
        prompt, system_prompt = return_meta_prompt_repr_3_hf(goal, src_obs, prev_actions)
        
        # get rep observation with maximum reward
        candidates = sample['candidates']
        #possible = any(item['action_matching']+item['reasoning_action_alignment']+item['extraction_action_alignment'] == 3.0 for item in candidates)
        possible = any(item['action_matching'] > 0.0 for item in candidates)
        if not possible:
            pass
        else:
            #rep_obs = max(sample['candidates'], key=lambda x: x['action_matching']+x['reasoning_action_alignment']+x['extraction_action_alignment'])['rephrase']
            key = max(sample['candidates'], key=lambda x: x['action_matching'])
            rep_obs = key['rephrase']
            rep_obs = rep_obs + '\n[END]'
            chat = [
                        {'role': 'system', 'content':system_prompt},
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': rep_obs} 
                    ]
            text = tokenizer.apply_chat_template(chat, tokenize=False)
            # configure the prompt and output
            df['text'].append(text)
            df['action'].append(sample['action'])
            df['rephrase'].append(rep_obs)
            df['obs'].append(src_obs)
            df['reward'].append(key['action_matching'])

    df = pd.DataFrame(df)
    
    df.to_csv('collected_data/sft_iter_0.csv')
