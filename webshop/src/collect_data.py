import os 
import time
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor

import openai
import torch
import langchain
import numpy as np
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from utils import convert_obs_in_history_inputs, return_self_ctx_prompt
from constants import FEW_SHOT_EXAMPLES_REPR

class TrainDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('collected_data/train_demo.csv')

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        output = {
                'traj_id': self.df.traj_id.iloc[index],
                'obs': self.df.obs.iloc[index],
                'obs_history': self.df.obs_history.iloc[index],
                'prev_actions': self.df.prev_actions.iloc[index],
                'action': self.df.action.iloc[index],
                'goal': self.df.goal.iloc[index]
                }
        return output 


def rephrase_model(prompt, num_candidates=4):
    api_key = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-002',
                generation_config=genai.GenerationConfig(
                    temperature=1.0
                    ),
                )
    responses = []
    for i in range(num_candidates):
        response = model.generate_content(prompt)
        responses.append(response.text)
    return responses


def format_reward_prompt(
    examplars, 
    obs_history, 
    obs_repr
    ):
    
    prompt = convert_obs_in_history_inputs(
        obs_history, 
        obs_repr, 
        backbone='gemini-pro'
        )
    
    prompt = f'''
{examplars}

{prompt}'''

    return prompt


def compute_action_matching_reward(
    obs_history, 
    obs_repr, 
    ref_action,
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
    action_prompt = format_reward_prompt(
        FEW_SHOT_EXAMPLES_REPR, 
        obs_history, 
        obs_repr
        ) 
    action_prompt = action_prompt[:-8]+'State the next action without providing any reasoning.\n\nAction: '

    # 2. get reward from each action models
    def process_action(model_invoke, chat_messages):
        try:
            action = model_invoke(chat_messages).content
        except:
            print('Retrying due to error')
            action = model_invoke(chat_messages).content
        action = action.lower().split('\n')[0].split('action:')[-1].strip()
        return action, len(action_prompt) / 4.0, len(action) / 4.0

    # Using ThreadPoolExecutor to parallelize the tasks
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Define your models and their corresponding invocations
        models = [action_gpt, action_gemini, action_claude]
        
        # Submit all tasks to the executor
        futures = [
            executor.submit(
                process_action, 
                model.invoke, 
                action_prompt
                )
            for model in models
        ]
        
        # Process the results as they complete
        for action_model, future in zip(['gpt', 'gemini', 'claude'], concurrent.futures.as_completed(futures)):
            action, input_tokens, output_tokens = future.result()
            pred_actions.append(action)

    # query api LLM & receive action prediction
    rewards = [int(pred_action.lower().strip() == ref_action) for pred_action in pred_actions]
    return rewards, pred_actions


def reward_model(obs_histories, obs_reprs, actions):
    action_matching_rewards = []
    pred_actions_list = []
    # 1. compute action matching reward
    if obs_histories is None:
        pass
    else:
        for obs_history, obs_repr, action in zip(obs_histories, obs_reprs, actions):
            action_matching_reward_list, pred_actions = compute_action_matching_reward(
                obs_history, 
                obs_repr, 
                action,
                )
            
            action_matching_reward = sum(action_matching_reward_list)
            action_matching_rewards.append(action_matching_reward)
            pred_actions_list.append(pred_actions)
     
    return action_matching_rewards, pred_actions_list 


def collect_data(num_candidates):
    trainset = TrainDataset()
    trainloader = DataLoader(trainset, batch_size = 1, shuffle=False)
    dataset = []

    for i, batch in enumerate(tqdm(trainloader)):
        sample = {
                'traj_id': batch['traj_id'][0].item(),
                'goal': batch['goal'][0],
                'obs': batch['obs'][0].replace('Instruction:\n'+batch['goal'][0]+'\n', ''),
                'action': batch['action'][0], 
                'prev_actions': batch['prev_actions'][0], 
                'candidates':[]
                }
        
        # divide batch
        obs_history = batch['obs_history'][0]
        goal = batch['goal'][0]
        obs = batch['obs'][0].replace('Instruction:\n'+goal+'\n', '')
        prev_actions = batch['prev_actions'][0]
        action = batch['action'][0]

        if type(prev_actions) != str:
            # pass initial action (as first observation is always [Search])
            pass 
        else:
            # sample multiple rephrase candidates
            lm_input = return_self_ctx_prompt(goal, obs, prev_actions)
            
            obs_reprs = rephrase_model(lm_input, num_candidates=num_candidates)
            obs_reprs = [obs_repr.split('**rephrased observation**:')[-1] for obs_repr in obs_reprs]
            
            # compute reward for every candidates in the candidate pool TODO: parallelize
            rewards, pred_actions_list = reward_model(
                                    [obs_history]*num_candidates,
                                    obs_reprs,
                                    [action]*num_candidates,
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
            with open(f"collected_data/sampled_contextualization_iter_0.json", 'w') as f:
                json.dump(dataset, f, indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arguments for sampling contextualization data')
    
    # Add arguments
    parser.add_argument('--num_samples', 
                        type=int, 
                        default=4, 
                        help='number of contextualized observations to be sampled by Contextualization LM')
    args = parser.parse_args()
    print('Start collecting contextualization data!')
    collect_data(num_candidates = args.num_samples)
    
