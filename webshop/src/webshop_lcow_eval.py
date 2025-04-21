
import json
import re
import os
import sys
import time

import gym
import torch
import numpy as np
import google.generativeai as genai

from rich.markup import escape
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append('./')
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import RandomPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE

from src.constants import FEW_SHOT_EXAMPLES_REPR
from src.utils import clean_obs, api_llm_inference, hf_llm_rephrase, return_lcow_prompt

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

def load_hfmodel(path):
    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
        attn_implementation='flash_attention_2'
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1 
    base_model = base_model.float()

    tokenizer = AutoTokenizer.from_pretrained(path,
                                            trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_eos_token = True
    print('Loaded Model and Tokenizer')

    return base_model, tokenizer

def run_eval_lcow(
                args, 
                num_tasks, 
                agent, 
                env, 
                contextualizer=None, 
                tokenizer=None, 
                validation=None
                ):
    if (contextualizer is None) and (tokenizer is None):
        
        base_ckpt_path = f"ckpt/sft_iter_{args.iter}/checkpoint-{args.ckpt_step}"
        contextualizer, tokenizer = load_hfmodel(base_ckpt_path)
        contextualizer = contextualizer.type(torch.bfloat16) 
        tokenizer.padding_side='right'     
    
    rewards = []
    rollouts = []
    
    for task_idx in tqdm(range(num_tasks)):
        obs, info = env.reset(session=task_idx)
        goal = clean_obs(obs)
        cleaned_goal = goal.split('Instruction: \n')[-1].split('\n')[0]
        previous_actions = ''
        print('-'*50+'\n')
        prompt = goal + '\n\nAction: '
        done = False
        trial = 0

        # for trajectory collection
        task_id_li, obs_li, obs_history_li, action_li, prev_actions_li, goal_li = [], [], [], [], [], [] 

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
            meta_prompt, system_prompt = return_lcow_prompt(cleaned_goal, obs, previous_actions)
            obs_repr = hf_llm_rephrase(
                contextualizer, 
                tokenizer, 
                meta_prompt, 
                system_prompt
                )
            # parse the contextualized observation
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
            if done:
                rewards.append(reward)
                break
            elif trial == 10:
                rewards.append(0.0)
            trial += 1
    
        if validation is None:
            rollouts.append({'rollout': prompt, 'reward': reward})
            print('Rewards: ', rewards)
            
            with open(f'results/expert_iter_{args.iter}_{args.backbone.split("/")[-1]}.json', 'w') as f:
                json.dump(rollouts, f, indent=4)
        
        avg_reward = sum(rewards) / 100
        sr = (np.array(rewards) == 1.0).mean()
        print('current Averaged reward:', avg_reward)
        print('current Success rate:', sr)

    # compute metric
    avg_reward = sum(rewards) / len(rewards)
    sr = (np.array(rewards) == 1.0).mean()
    print('Averaged rewards:', avg_reward)
    print('Success rate:', sr)
    return avg_reward, sr



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arguments for Webshop Evaluation using Fine-tuned Contextualization module')
    
    # Add arguments
    parser.add_argument('--num_tasks', type=int, help='number of the episodes to evaluate')
    parser.add_argument('--backbone', type=str, default='gemini-pro', help='type of backbone LLM')
    parser.add_argument('--ckpt_step', type=int)
    parser.add_argument('--iter', type=int)
    args = parser.parse_args()

    env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=DEBUG_PROD_SIZE)

    agent = WebshopAgent(FEW_SHOT_EXAMPLES_REPR, args.backbone)
    run_eval_lcow(args, args.num_tasks, agent, env)
        

