import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from utils import clean_obs

import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split

def clean_obs(obs):
    return obs.replace('[button]', '[').replace('[button_]', ']')

def demo_to_step_level(root='data/il_trajs_finalized_images.jsonl', max_steps=20):
    df = pd.read_json(root, lines=True)
    df['seq_len'] = list(map(lambda x: len(x), df.actions.tolist()))
    
    # get reward information
    root_dir = 'data/all_trajs'
    files = os.listdir(root_dir)
    files = list(map(lambda x: os.path.join(root_dir, x), files))

    trajs = []
    for i, file in enumerate(files):
        try:
            traj = pd.read_json(file, lines=True)
            if 'goal' in traj.keys() and 'reward' in traj.keys():
                trajs.append(traj)
        except:
            pass

    rewards = []
    for i in range(len(df)):
        sample = df.iloc[i]
        goal = sample['states'][0].split('\n')[2]
        seq_len = sample['seq_len']
        reward = -1
        for traj in trajs:
            if goal == traj.goal[0]['instruction_text'] and seq_len == len(traj)-1:
                reward = traj.reward.iloc[-1]
                break

        rewards.append(reward)

    df['reward'] = rewards
    df = df[df.reward != -1]

    # drop the demonstration exceeds max_step 
    df = df[df.seq_len <= max_steps]

    train_df, test_df = train_test_split(df, 
                                            test_size=0.1, 
                                            random_state=42)
    
    # convert the aggregated demonstration into step level dataframe (train_demo)
    traj_ids, observations, observations_history, actions, goals, prev_actions_li, seq_lens, rewards = [], [], [], [], [], [], [], []
    for i, (obs, act, seq_len, rwd) in enumerate(zip(train_df['states'], train_df['actions'], train_df['seq_len'], train_df['reward'])):
        observations += list(map(lambda x: clean_obs(x), obs))
        actions += act
        traj_ids += [i]*len(act)
        seq_lens += [seq_len]*len(act)
        rewards += [rwd]*len(act)
        goal = obs[1].split('Instruction:\n')[-1].split('\n')[0]
        goals += [goal]*len(act)

        ep_obs = []
        ep_prev_actions = []
        obs_prompt = f"{clean_obs(obs[0])}\nAction: "
        prev_actions = ""
        ep_obs.append(obs_prompt)
        ep_prev_actions.append(prev_actions)
        for j, (ob, a) in enumerate(zip(obs[1:], act[:-1])):
            prev_actions += f"{j+1}. {a}\n" 
            obs_prompt += f'{a}\n\nObservation:\n{clean_obs(ob)}\nAction: '
            ep_obs.append(obs_prompt)
            ep_prev_actions.append(prev_actions)
        observations_history += ep_obs
        prev_actions_li += ep_prev_actions

    train_demo = pd.DataFrame({'traj_id':traj_ids, 
                            'seq_len':seq_lens,
                            'obs':observations, 
                            'obs_history':observations_history, 
                            'action':actions, 
                            'prev_actions': prev_actions_li, 
                            'goal': goals,
                            'reward': rewards})

    train_demo = train_demo.sort_values(by=['seq_len', 'traj_id'])

    traj_ids, observations, observations_history, actions, goals, prev_actions_li, seq_lens, rewards = [], [], [], [], [], [], [], []
    for i, (obs, act, seq_len, rwd) in enumerate(zip(test_df['states'], test_df['actions'], test_df['seq_len'], test_df['reward'])):
        observations += list(map(lambda x: clean_obs(x), obs))
        actions += act
        traj_ids += [i]*len(act)
        seq_lens += [seq_len]*len(act)
        rewards += [rwd]*len(act)
        goal = obs[1].split('Instruction:\n')[-1].split('\n')[0]
        goals += [goal]*len(act)

        ep_obs = []
        ep_prev_actions = []
        obs_prompt = f"{clean_obs(obs[0])}\nAction: "
        prev_actions = ""
        ep_obs.append(obs_prompt)
        ep_prev_actions.append(prev_actions)
        for j, (ob, a) in enumerate(zip(obs[1:], act[:-1])):
            prev_actions += f"{j+1}. {a}\n" 
            obs_prompt += f'{a}\n\nObservation:\n{clean_obs(ob)}\nAction: '
            ep_obs.append(obs_prompt)
            ep_prev_actions.append(prev_actions)
        observations_history += ep_obs
        prev_actions_li += ep_prev_actions

    test_demo = pd.DataFrame({'traj_id':traj_ids, 
                            'seq_len':seq_lens,
                            'obs':observations, 
                            'obs_history':observations_history, 
                            'action':actions, 
                            'prev_actions': prev_actions_li, 
                            'goal': goals,
                            'reward': rewards})

    test_demo = test_demo.sort_values(by=['seq_len', 'traj_id'])

    return train_df, test_df, train_demo, test_demo

if __name__ == '__main__':
    _, _, train_demo, test_demo = demo_to_step_level(max_steps=6)
    train_demo = train_demo[train_demo.reward == 1.0]
    test_demo = test_demo[test_demo.reward == 1.0]
    train_demo.to_csv('collected_data/train_demo.csv')
    test_demo.to_csv('collected_data/test_demo.csv') 
