# SFT dataset
import numpy as np 
import pandas as pd
import json
import pandas as pd
import transformers
from transformers import AutoTokenizer

from prompt_v2 import hf_format_rephrase_prompt
from utils import add_action_semantic, remove_all_error_logs

def create_sft_dataset(data):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    df = {'text': [], 
          'task_name':[], 
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
        task_name = sample['task_name']
        sem_action = add_action_semantic(action, obs)
        # get rep observation with maximum reward
    
        possible = max(sample['action_matching_rewards']) > 0
        if not possible:
            pass
        else:
            key = max(sample['candidates'], key=lambda x: x['action_matching'])
            repr = key['rephrase']+'[END]'
            rephrase_prompt, rephrase_system_prompt = hf_format_rephrase_prompt(sample['domain_info'],
                                                                        sample['goal'], 
                                                                        '\n'.join(remove_all_error_logs(sample['history']).split('\n')[1:]).strip(), 
                                                                        '\n'.join(sample['observation'].split('\n')[1:]).strip())
            if repr == None:
                pass
            else:
                sample_loc.append(f'{sample["task_id"]}_{sample["task_timestep"]}')
                # configure the prompt and output
                chat = [
                    {'role': 'system', 'content': rephrase_system_prompt},
                    {'role': 'user', 'content': rephrase_prompt},
                    {'role': 'assistant', 'content': repr} 
                ]
                text = tokenizer.apply_chat_template(chat, tokenize=False)
                df['text'].append(text)
                df['reward'].append(key['action_matching'])
                df['reward_spec'].append(key['action_matching_list'])
                df['input'].append(rephrase_prompt)
                df['output'].append(repr)
                df['task_id'].append(sample['task_id'])
                df['step'].append(sample['task_timestep'])
                df['gt_action'].append(sem_action)
                df['history'].append(history)
                df['task_name'].append(task_name)
    df = pd.DataFrame(df)
    
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arguments for training data aggregation')
    # Add arguments
    parser.add_argument('--iter', type=int, default = 0)
    args = parser.parse_args()
    with open(f'./workarena_data/rephrase_data_workarena_iter_{args.iter-1}.json', 'r') as f:
        data = json.load(f)

    df = create_sft_dataset(data)
    df.to_csv(f'workarena_data/sft_train_iter_{args.iter}.csv')
