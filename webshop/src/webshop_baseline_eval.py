
import sys
sys.path.append('./')
import time
import re 
import gym
from tqdm import tqdm
from rich.markup import escape
import numpy as np 
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import RandomPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE

from constants import FEW_SHOT_EXAMPLES, FEW_SHOT_EXAMPLES_REPR
from utils import clean_obs, api_llm_inference, return_self_ctx_prompt
import json
import os


# Webshop agent based on LLM api
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

# Baseline evaluation (Act Prompting)
def run_baseprompt(args, agent, env):
    rewards = []
    rollouts = []
    for task_idx in tqdm(range(args.num_tasks)):
        obs, info = env.reset(session=task_idx)
        goal = clean_obs(obs)
        cleaned_goal = goal.split('Instruction: \n')[-1].split('\n')[0]
        print('-'*50+'\n')
        print(goal)
     
        prompt = goal + '\n\nAction: '
        done = False
        trial = 0
        while not done and trial <= 10:
            # predict next action given current history of (obs, action)
            action_prompt = prompt[:-8]+'State the next action without providing any reasoning.\n\nAction: '
            action = agent.act(action_prompt)
            # take step
            obs, reward, done, info = env.step(action)
            # preprocessing observation
            obs = clean_obs(obs)
            obs = obs.replace('Instruction:\n'+cleaned_goal+'\n', '')
            print(f'Action: {str(action)}\n\nObservation:\n{str(obs)}\n\n')
            prompt += f'{action}\n\nObservation:\n{obs}\n\nAction: '
            if done:
                rewards.append(reward)
                break
            
            elif trial == 10:
                rewards.append(0)
            trial += 1

        # Save rollout result
        rollouts.append({'rollout': prompt, 'reward': reward})
        print('Rewards: ', rewards)
        with open(f'results/{args.backbone.split("/")[1]}_rollout_actonly.json', 'w') as f:
            json.dump(rollouts, f, indent=4)
    # Compute metric (Success rate & Average reward)
    avg_reward = sum(rewards) / len(rewards)
    sr = (np.array(rewards) == 1.0).mean()
    print('Averaged rewards:', avg_reward)
    print('Success rate:', sr)


# Evaluation of Agent LM + Rephraser LM
def run_selfctx(args, agent, env):
    rewards = []
    rollouts = []
    for task_idx in tqdm(range(args.num_tasks)):
        # reset environment
        obs, info = env.reset(session=task_idx)
        goal = clean_obs(obs)
        cleaned_goal = goal.split('Instruction: \n')[-1].split('\n')[0]
        previous_actions = ''
        print('-'*50+'\n')
        print(goal)
        prompt = goal + '\n\nAction: '
        done = False
        trial = 0
        # start multi-step decision making
        while not done and trial <= 10:
            # predict next action
            action_prompt = prompt[:-8]+'State the next action without providing any reasoning.\n\nAction: '
            #if trial == 1:
            #    breakpoint()
            action = agent.act(action_prompt)
            # incrementally update previous actions
            previous_actions += f'{trial}. {action}\n'
            # take step
            obs, reward, done, info = env.step(action)
            # preprocess observation
            obs = clean_obs(obs)
            obs = obs.replace('Instruction:\n'+cleaned_goal+'\n', '')
            # Rephrase raw observation based on llm rephraser
            meta_prompt = return_self_ctx_prompt(cleaned_goal, obs, previous_actions)
            obs_repr = api_llm_inference(meta_prompt, args.backbone, max_new_tokens=1024)
            obs_repr = obs_repr.split('**rephrased observation**:')[-1]
            print(f'Action: {action}\n\nObservation:\n{obs_repr}\n\n')
            # Update input prompt (history of observation and action)
            prompt += f'{action}\n\nObservation:\n{obs_repr}\n\nAction: '
            if done:
                rewards.append(reward)
                break
            elif trial == 10:
                rewards.append(0)
            trial += 1
        rollouts.append({'rollout': prompt, 'reward': reward})
        print('Rewards: ', rewards)
        # save log
        with open(f'results/{args.backbone.split("/")[1]}_rollout_repr.json', 'w') as f:
            json.dump(rollouts, f, indent=4)
    # Compute metric (Success rate / Average reward)
    avg_reward = sum(rewards) / len(rewards)
    sr = (np.array(rewards) == 1.0).mean()
    print('Averaged rewards:', avg_reward)
    print('Success rate:', sr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arguments for Webshop Baseline Evaluation')
    
    # Add arguments
    parser.add_argument('--num_tasks', type=int, help='number of the episodes to evaluate')
    parser.add_argument('--rephrase', action='store_true', help='whether or not apply observation rephrase module')
    parser.add_argument('--backbone', type=str, default='gemini-pro', help='type of backbone LLM api')
    args = parser.parse_args()

    env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=DEBUG_PROD_SIZE)

    if args.rephrase:
        agent = WebshopAgent(FEW_SHOT_EXAMPLES_REPR, args.backbone)
        run_selfctx(args, agent, env)

    if not args.rephrase:
        agent = WebshopAgent(FEW_SHOT_EXAMPLES, args.backbone)
        run_baseprompt(args, agent, env)

            
        

