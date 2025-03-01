import google.generativeai as genai
import openai
import sys
import time
import gym
import pandas as pd
import numpy as np
from rich.markup import escape
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
sys.path.append('./')
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import RandomPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE

from src.constants import FEW_SHOT_EXAMPLES_REPR
from src.utils import clean_obs, api_llm_inference, hf_llm_rephrase, return_meta_prompt_repr_3_hf 
import json
import re
import os


class WebshopAgent:
    def __init__(self, few_shot_examples, backbone):
        self.few_shot_prompt = few_shot_examples
        self.previous_actions = []
        self.backbone = backbone
    
    def act(self, input):
        prompt = self.few_shot_prompt + '\n' + input
        action = api_llm_inference(prompt, self.backbone, max_new_tokens=50)
        action = action.lower().split('\n')[0].split('action:')[-1].strip()
        action = re.sub(r'\[\s+', '[', action)  # Removes space after [
        action = re.sub(r'\s+\]', ']', action)
        return action
    

def run_eval_multiturn_repr(args, 
                            num_tasks, 
                            agent,
                            env,
                            rephraser=None, 
                            rephraser_tok=None, 
                            validation=None):
    
    if (rephraser is None) and (rephraser_tok is None):
        # load fine-tuned checkpoint
        ckpt_path = f'ckpt/sft_iter_{args.iter-1}/checkpoint-{args.ckpt_step}'
        rephraser = AutoModelForCausalLM.from_pretrained(ckpt_path,
                                                        device_map='auto',
                                                        torch_dtype = torch.bfloat16,
                                                        attn_implementation='flash_attention_2',
                                                        trust_remote_code=True,
                                                        use_auth_token=True)
        rephraser = rephraser.type(torch.bfloat16)
        rephraser_tok = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        rephraser_tok.padding_side='right'
    
    rewards = []
    rollouts = []
    collected_traj = {'traj_id':[],
                      'obs':[],
                      'obs_history':[],
                      'action':[],
                      'prev_actions':[],
                      'goal':[],
                      'reward':[]}
    
    for task_idx in tqdm(range(num_tasks)):
        obs, info = env.reset(session=task_idx+1000) # for training tasks (seed = 1000 ~ 1500)
        goal = clean_obs(obs)
        cleaned_goal = goal.split('Instruction: \n')[-1].split('\n')[0]
        previous_actions = ''
        print('-'*50+'\n')
        prompt = goal + '\n\nAction: '
        done = False
        trial = 0

        # for trajectory collection
        task_id_li, obs_li, obs_history_li, action_li, prev_actions_li, goal_li = [[]]*6 

        # start multi-step decision making
        while not done and trial <= 10:
            # predict action
            action_prompt = prompt[:-8]+'State the next action without providing any reasoning.\n\nAction: '
            action = agent.act(action_prompt)
            previous_actions += f'{trial}. {action}\n'
            # take step 
            obs, reward, done, info = env.step(action)
            # clean observation
            obs = clean_obs(obs)
            obs = obs.replace('Instruction:\n'+cleaned_goal+'\n', '')
            # rephrase observation
            meta_prompt, system_prompt = return_meta_prompt_repr_3_hf(cleaned_goal, obs, previous_actions)
            obs_repr = hf_llm_rephrase(rephraser, rephraser_tok, meta_prompt, system_prompt)
            # parse the repharased observation
            obs_repr = obs_repr.split('[END]')[0]
            print(f'Action: {str(action)}\n\nObservation:\n{str(obs_repr)}\n\n')
            prompt += f'{action}\n\nObservation:\n{obs_repr}\n\nAction: '

            # gather trajectory segment
            if not done and trial < 10:
                task_id_li.append(task_idx)
                obs_li.append(obs)
                goal_li.append(cleaned_goal)
                obs_history_li.append(prompt)
                prev_actions_li.append(previous_actions)
            if trial > 0:
                action_li.append(action)
            ###########################
            if done:
                rewards.append(reward)
                break
            elif trial == 10:
                rewards.append(0.0)
            trial += 1
        collected_traj['traj_id'] += task_id_li
        collected_traj['obs'] += obs_li
        collected_traj['action'] += action_li
        collected_traj['prev_actions'] += prev_actions_li
        collected_traj['obs_history'] += obs_history_li
        collected_traj['goal'] += goal_li
        collected_traj['reward'] += [reward] * len(task_id_li)
    
        traj_df = pd.DataFrame(collected_traj)
        traj_df.to_csv(f'collected_data/collected_traj_traintask_iter_{args.iter}.csv')
        avg_reward = sum(rewards) / len(rewards)
        sr = (np.array(rewards) == 1.0).mean()
        print('Averaged rewards until now:', avg_reward)
        print('Success rate until now:', sr)

        if validation is None:
            rollouts.append({'rollout': prompt, 'reward': reward})
            print('Rewards: ', rewards)
            # save log

    # compute metric
    avg_reward = sum(rewards) / len(rewards)
    sr = (np.array(rewards) == 1.0).mean()
    print('Averaged rewards:', avg_reward)
    print('Success rate:', sr)
    return avg_reward, sr



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arguments for collecting successful trajectories in WebShop training environment')
    
    # Add arguments
    parser.add_argument('--num_tasks', type=int, help='number of the episodes to evaluate')
    parser.add_argument('--backbone', type=str, default='gemini-pro', help='type of backbone LLM api')
    parser.add_argument('--ckpt_step', type=int)
    parser.add_argument('--iter', type=int, help='iteration for data collection', default=1)
    args = parser.parse_args()

    env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=DEBUG_PROD_SIZE)

    agent = WebshopAgent(FEW_SHOT_EXAMPLES_REPR, args.backbone)
    run_eval_multiturn_repr(args, args.num_tasks, agent, env)

            
        

