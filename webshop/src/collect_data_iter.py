import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from constants import FEW_SHOT_EXAMPLES_3_REPR, FEW_SHOT_EXAMPLES_REPR
import json
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor
from utils import hf_convert_obs_in_history_inputs_v2, hf_llm_batch_rephrase, return_meta_prompt_repr_3_hf
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import concurrent


class TrainDataset(Dataset):
    def __init__(self, args):
        demo_df = pd.read_csv('collected_data/train_demo.csv')[['traj_id', 'obs', 'obs_history', 'action', 'prev_actions', 'goal', 'reward']]
        demo_df['type'] = ['expertdemo']*len(demo_df)
        traj_df = pd.read_csv(f'collected_data/collected_traj_traintask_iter_{args.iter}.csv')
        traj_df = traj_df.rename(columns = {'task_id':'traj_id'})
        traj_df = traj_df[['traj_id', 'obs', 'obs_history', 'action', 'prev_actions', 'goal', 'reward']]
        traj_df['type'] = ['collected_traj']*len(traj_df)
        self.df = pd.concat([demo_df, traj_df], axis=0)
        self.df = self.df[self.df.reward == 1.0]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        output = {
            'type': self.df.type.iloc[index],
            'traj_id': self.df.traj_id.iloc[index],
            'obs': self.df.obs.iloc[index],
            'obs_history': self.df.obs_history.iloc[index],
            'prev_actions': self.df.prev_actions.iloc[index],
            'action': self.df.action.iloc[index],
            'goal': self.df.goal.iloc[index]
            }
        return output 


def format_reward_prompt(
    model, 
    tokenizer, 
    examplars, 
    obs_history, 
    obs_repr, 
    system_prompt
    ):
    
    prompt = hf_convert_obs_in_history_inputs_v2(
        model, 
        tokenizer, 
        obs_history, 
        obs_repr, 
        system_prompt
        )
    prompt = f'''
{examplars}

{prompt}'''

    return prompt


def compute_action_matching_reward(
    model, 
    tokenizer, 
    obs_history, 
    obs_repr, 
    ref_action, 
    system_prompt
    ):
    # define action modules
    action_gpt = ChatOpenAI(
                model_name='gpt-4o-2024-08-06',
                temperature=0.0,
                max_tokens=50,
            )
    action_claude = ChatAnthropic(
                model_name='claude-3-5-sonnet-20240620',
                anthropic_api_key=os.environ['ANTHROPIC_API_KEY'],
                temperature=0.0,
                max_tokens=50,
            )
    action_gemini = ChatGoogleGenerativeAI(
                model='gemini-1.5-flash-latest', 
                temperature=0.0,
                max_tokens=50,
            )
    # 1. convert observations in prompt using current policy (prompt LM)
    pred_actions = []
    action_prompt = format_reward_prompt(model, tokenizer, FEW_SHOT_EXAMPLES_REPR, obs_history, obs_repr, system_prompt) 
    action_prompt = action_prompt[:-8]+'State the next action without providing any reasoning.\n\nAction: '
    # 2. get reward from each action models
    def process_action(model_invoke, chat_messages):
        try:
            action = model_invoke(chat_messages).content
        except:
            print('Retrying due to error')
            action = model_invoke(chat_messages).content
        action = action.lower().split('\n')[0].split('action:')[-1].strip()
        #action = action.lower().split('\n')[0].split('action:')[-1].strip()
        return action, len(action_prompt) / 4.0, len(action) / 4.0

    # Using ThreadPoolExecutor to parallelize the tasks
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Define your models and their corresponding invocations
        models = [action_gpt, action_gemini, action_claude]
        
        # Submit all tasks to the executor
        futures = [
            executor.submit(process_action, model.invoke, action_prompt)
            for model in models
        ]
        
        # Process the results as they complete
        for action_model, future in zip(['gpt', 'gemini', 'claude'], concurrent.futures.as_completed(futures)):
            action, input_tokens, output_tokens = future.result()
            pred_actions.append(action)

    # query api LLM & receive action prediction
    rewards = [int(pred_action.lower().strip() == ref_action) for pred_action in pred_actions]
    return rewards, pred_actions


def reward_model(model, tokenizer, obs_histories, obs_reprs, actions, system_prompt):
    '''
    model:
    tokenizer:
    obs_reprs: [List] rephrasd observation - online
    actions: [List] action label
    goals: [List] goal instruction
    prev_actions: [List] str indicating previously executed actions
    '''
    tokens = 0
    action_matching_rewards = []
    pred_actions_list = []
    # 1. compute action matching reward
    if obs_histories is None:
        pass
    else:
        for obs_history, obs_repr, action in zip(obs_histories, obs_reprs, actions):
            action_matching_reward_list, pred_actions = compute_action_matching_reward(model, tokenizer, obs_history, obs_repr, action, system_prompt)
            action_matching_reward = sum(action_matching_reward_list)
            action_matching_rewards.append(action_matching_reward)
            pred_actions_list.append(pred_actions)
     
    return action_matching_rewards, pred_actions_list 



def collect_data(args, model, tokenizer, num_candidates):
    trainset = TrainDataset(args)
    trainloader = DataLoader(trainset, batch_size = 1, shuffle=False)
    dataset = []

    for i, batch in enumerate(tqdm(trainloader)):
        if type(batch['action'][0]) == torch.Tensor:
            action = 'None'
        else:
            action = batch['action'][0]
        sample = {
                'type': batch['type'][0],
                'traj_id': batch['traj_id'][0].item(),
                'goal': batch['goal'][0],
                'obs': batch['obs'][0].replace('Instruction:\n'+batch['goal'][0]+'\n', ''),
                'action': action, 
                'prev_actions': batch['prev_actions'][0], 
                'candidates':[]
                }
        
        # divide batch
        obs_history = batch['obs_history'][0]
        goal = batch['goal'][0]
        obs = batch['obs'][0].replace('Instruction:\n'+goal+'\n', '')
        prev_actions = batch['prev_actions'][0]
        action = batch['action'][0]
        if type(action) == torch.Tensor:
            action = 'None'

        if type(prev_actions) != str:
            # pass initial action (as first observation is always [Search])
            pass 
        else:
            # sample multiple rephrase candidates
            lm_input, rephrase_system_prompt = return_meta_prompt_repr_3_hf(goal, obs, prev_actions)
            obs_reprs = hf_llm_batch_rephrase(model, tokenizer, [lm_input]*num_candidates, rephrase_system_prompt) 
            rewards, pred_actions_list = reward_model(
                                    model,
                                    tokenizer,
                                    [obs_history]*num_candidates,
                                    obs_reprs,
                                    [action]*num_candidates,
                                    rephrase_system_prompt
                                    )
            for obs_repr, r, pred_actions in zip(obs_reprs, rewards, pred_actions_list):
                sample['candidates'].append({'rephrase': obs_repr,
                                             'action_matching': r,
                                             'pred_actions': pred_actions,
                                             })
                
            print('Reference action: ', action)
            print('Predicted actions: ', pred_actions_list)
            print('total rewards:', rewards)
            dataset.append(sample)
            # incrementally save collected data
            with open(f"collected_data/sampled_contextualization_iter_{args.iter}.json", 'w') as f:
                json.dump(dataset, f, indent=4)



#######################################################

if __name__ == '__main__':
    import argparse
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description='arguments for Data collection for Expert iteration using api LLM')
    
    # Add arguments
    parser.add_argument('--num_samples', type=int, default=4, help='number of rephrases to be sampled by Rephraser LM')
    parser.add_argument('--iter', type=int, help='expert iteration', default=1)
    parser.add_argument('--ckpt_step', type=int, help='checkpoint step of current rephraser')
    args = parser.parse_args() 

    ckpt_path = f'ckpt/sft_iter_{args.iter-1}/checkpoint-{args.ckpt_step}'
    model = AutoModelForCausalLM.from_pretrained(ckpt_path,
                                                device_map='auto',
                                                torch_dtype = torch.bfloat16,
                                                attn_implementation='flash_attention_2',
                                                trust_remote_code=True,
                                                use_auth_token=True)
    model = model.type(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    tokenizer.padding_side='left'
    
    print('Start collecting data!')
    collect_data(args, model, tokenizer, num_candidates = args.num_samples)
