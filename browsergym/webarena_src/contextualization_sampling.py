import sys
import re
import os 
import json
import random
import concurrent.futures

import pandas as pd
from tqdm import tqdm

from reward_model import action_alignment
from prompt import format_rephrase_prompt, format_rephrase_prompt_retry, format_action_prompt_for_datacollection
from utils import remove_all_error_logs

sys.path.append('./')
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs


def main(demo : pd.DataFrame, 
         num_candidates : int = 3):
    collected_data = []
    rep_model_args = ChatModelArgs(
                model_name='anthropic/claude-3-5-sonnet-20240620',
                max_total_tokens=128_000, 
                max_input_tokens=126_488, 
                max_new_tokens=2_000,
                temperature = 1.0,
                top_p = 1.0,
            )
    
    rep_claude = rep_model_args.make_chat_model()
    
    # define 3 action models
    action_model_args = ChatModelArgs(
                model_name='openai/gpt-4o-2024-08-06',
                max_total_tokens=128_000,
                max_input_tokens=126_488,
                max_new_tokens=2_000,
                temperature = 0.0,
            )
    action_gpt = action_model_args.make_chat_model()

    action_model_args = ChatModelArgs(
                model_name='googleai/gemini-1.5-flash-002',
                max_total_tokens=128_000,
                max_input_tokens=126_488,
                max_new_tokens=2_000,
                temperature = 0.0,
            )
    action_genai = action_model_args.make_chat_model()

    action_model_args = ChatModelArgs(
                model_name='anthropic/claude-3-5-sonnet-20240620',
                max_total_tokens=128_000,
                max_input_tokens=126_488,
                max_new_tokens=2_000,
                temperature = 0.0,
            )
    action_claude = action_model_args.make_chat_model()
    
    availables = 0
    gpt_input_tokens = 0
    gpt_output_tokens = 0
    genai_input_tokens = 0
    genai_output_tokens = 0
    ant_input_tokens = 0
    ant_output_tokens = 0
    collected_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i in tqdm(range(len(demo))):
        
        task_id = str(demo.task_id.iloc[i])
        step = str(demo.step.iloc[i])
        domain_info = demo.domain_info.iloc[i]
        goal = demo.goal.iloc[i]
        history = demo.action_history.iloc[i]
        observation = demo.observation.iloc[i]
        ref_action = demo.action.iloc[i]
        ref_semantic_action = demo.semantic_action.iloc[i]

        ################## 1. SAMPLE CANDIDATE CONTEXTUALIZATIONS ##################
        candidates = []
        # format rephrase prompt
        rephrase_prompt, rephrase_system_prompt = format_rephrase_prompt(
            domain_info, 
            goal, 
            history, 
            '\n'.join(observation.split('\n')[1:]).strip(),
                )
        ant_input_tokens += len(rephrase_prompt) / 4.0 * num_candidates
        chat_messages = [
            {'role':'system', 'content':rephrase_system_prompt},
            {'role':'user', 'content':rephrase_prompt}
        ]
        
        rephrase_outputs = []
        for rephraser in [rep_claude]*num_candidates:
            rephrase_output = rephraser.invoke(chat_messages).content
            rephrase_outputs.append(rephrase_output)
    
        
        for rephrase_output in rephrase_outputs:
            ant_output_tokens += len(rephrase_output) / 4.0
            # parsing
            plan_pattern = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
            plan_match = plan_pattern.search(rephrase_output)
            try:
                plan = plan_match.group(1).strip()
            except:
                plan = 'None'
            # get extraction
            extract_pattern = re.compile(r'<extraction>(.*?)</extraction>', re.DOTALL)
            extract_match = extract_pattern.search(rephrase_output)
            try:
                rep_observation = extract_match.group(1).strip()
            except:
                rep_observation = 'None'
            # add to candidates
            candidates.append({'think': plan, 'obs_repr': rep_observation})
        ############################################################################

        ################## 2. COMPUTE ACTION MATCHING REWARD ####################
        action_matching_reward_lists = []
        pred_action_lists = []
        pred_think_lists = []
        action_matching_rewards = []
        for j, candidate in enumerate(candidates):
            action_prompt, action_system_prompt = format_action_prompt_for_datacollection(
                goal,
                history, 
                candidate['think'], 
                candidate['obs_repr'])

            chat_messages = [
                {'role':'system', 'content': action_system_prompt},
                {'role':'user', 'content': action_prompt}
                ]
            
            action_outputs = []
            # verfiy actions with 3 action modules
            def process_action(model_invoke, chat_messages):
                try:
                    action = model_invoke(chat_messages).content
                except:
                    print('Retrying due to error')
                    action = model_invoke(chat_messages).content
                return action, len(action_prompt) / 4.0, len(action) / 4.0

            # Using ThreadPoolExecutor to parallelize the tasks
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Define your models and their corresponding invocations
                models = [action_gpt, action_genai, action_claude]
                
                # Submit all tasks to the executor
                futures = [
                    executor.submit(process_action, model.invoke, chat_messages)
                    for model in models
                ]
                
                # Process the results as they complete
                for action_model, future in zip(['gpt', 'gemini', 'claude'], 
                                                concurrent.futures.as_completed(futures)):
                    action, input_tokens, output_tokens = future.result()
                    action_outputs.append(action)
                    if action_model == 'gpt':
                        gpt_input_tokens += input_tokens
                        gpt_output_tokens += output_tokens
                    elif action_model == 'gemini':
                        genai_input_tokens += input_tokens
                        genai_output_tokens += output_tokens
                    elif action_model == 'claude':
                        ant_input_tokens += input_tokens
                        ant_output_tokens += output_tokens
                    

            ## compute action matching reward for every LLMs
            pred_action_list = []
            pred_think_list = []
            action_matching_reward_list = []
            action_matching_reward = 0
            for action_output in action_outputs:
                # extract action
                action_pattern = re.compile(r'<action>(.*?)</action>', re.DOTALL)
                action_match = action_pattern.search(action_output)
                try:
                    action = action_match.group(1).strip()
                except:
                    action = None 
                pred_action_list.append(action)
                # extract think
                think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
                think_match = think_pattern.search(action_output)
                try:
                    think = think_match.group(1).strip()
                except:
                    think = None 
                pred_think_list.append(think)

                # assign action matching reward
                if action_alignment(action, ref_action):
                    action_matching_reward += 1
                    action_matching_reward_list.append(1)
                else:
                    action_matching_reward += 0
                    action_matching_reward_list.append(0)
    
            ########################
            action_matching_rewards.append(action_matching_reward)
            action_matching_reward_lists.append(action_matching_reward_list)
            pred_action_lists.append(pred_action_list)
            pred_think_lists.append(pred_think_list)
        # for tracking
        max_reward = max(action_matching_rewards)


        if max_reward == 0:
            print('Retry due to lack of valid contextualization / current rewards :', 
                  action_matching_rewards)
            ######## Retry - 1. SAMPLE CANDIDATE REFINEMENT with Action Que  #######
            candidates = []
            # format rephrase prompt
            rephrase_prompt, rephrase_system_prompt = format_rephrase_prompt_retry(
                domain_info, 
                goal, 
                history, 
                '\n'.join(observation.split('\n')[1:]).strip(),
                ref_semantic_action
            )
            ant_input_tokens += len(rephrase_prompt) / 4.0 * num_candidates
            chat_messages = [
                {'role':'system', 'content':rephrase_system_prompt},
                {'role':'user', 'content':rephrase_prompt}
            ]
            
            rephrase_outputs = []
            for rephraser in [rep_claude]*num_candidates:
                rephrase_output = rephraser.invoke(chat_messages).content
                rephrase_outputs.append(rephrase_output)
        
            
            for rephrase_output in rephrase_outputs:
                ant_output_tokens += len(rephrase_output) / 4.0
                # parsing
                plan_pattern = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
                plan_match = plan_pattern.search(rephrase_output)
                try:
                    plan = plan_match.group(1).strip()
                except:
                    plan = 'None'
                # get extraction
                extract_pattern = re.compile(r'<extraction>(.*?)</extraction>', re.DOTALL)
                extract_match = extract_pattern.search(rephrase_output)
                try:
                    rep_observation = extract_match.group(1).strip()
                except:
                    rep_observation = 'None'
                # add to candidates
                candidates.append({'think': plan, 'obs_repr': rep_observation})
            
            ############################################################################
            ###################### Compute reward for each candidate ###################
            action_matching_reward_lists = []
            pred_action_lists = []
            pred_think_lists = []
            action_matching_rewards = []
            for j, candidate in enumerate(candidates):
                action_prompt, action_system_prompt = format_action_prompt_for_datacollection(
                    goal,
                    history, 
                    candidate['think'], 
                    candidate['obs_repr'])

                chat_messages = [
                    {'role':'system', 'content': action_system_prompt},
                    {'role':'user', 'content': action_prompt}
                    ]
            
                action_outputs = []
                # verfiy actions with 3 action modules
                def process_action(model_invoke, chat_messages):
                    try:
                        action = model_invoke(chat_messages).content
                    except:
                        print('Retrying due to error')
                        action = model_invoke(chat_messages).content
                    return action, len(action_prompt) / 4.0, len(action) / 4.0

                # Using ThreadPoolExecutor to parallelize the tasks
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Define your models and their corresponding invocations
                    models = [action_gpt, action_genai, action_claude]
                    
                    # Submit all tasks to the executor
                    futures = [
                        executor.submit(process_action, model.invoke, chat_messages)
                        for model in models
                    ]
                    
                    # Process the results as they complete
                    for action_model, future in zip(['gpt', 'gemini', 'claude'], concurrent.futures.as_completed(futures)):
                        action, input_tokens, output_tokens = future.result()
                        action_outputs.append(action)
                        if action_model == 'gpt':
                            gpt_input_tokens += input_tokens
                            gpt_output_tokens += output_tokens
                        elif action_model == 'gemini':
                            genai_input_tokens += input_tokens
                            genai_output_tokens += output_tokens
                        elif action_model == 'claude':
                            ant_input_tokens += input_tokens
                            ant_output_tokens += output_tokens
                        

                ## compute action matching reward for every LLMs
                pred_action_list = []
                pred_think_list = []
                action_matching_reward_list = []
                action_matching_reward = 0
                for action_output in action_outputs:
                    # extract action
                    action_pattern = re.compile(r'<action>(.*?)</action>', re.DOTALL)
                    action_match = action_pattern.search(action_output)
                    try:
                        action = action_match.group(1).strip()
                    except:
                        action = None 
                    pred_action_list.append(action)
                    # extract think
                    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
                    think_match = think_pattern.search(action_output)
                    try:
                        think = think_match.group(1).strip()
                    except:
                        think = None 
                    pred_think_list.append(think)

                    # assign action matching reward
                    if action_alignment(action, ref_action):
                        action_matching_reward += 1
                        action_matching_reward_list.append(1)
                    else:
                        action_matching_reward += 0
                        action_matching_reward_list.append(0)
        
                ########################
                action_matching_rewards.append(action_matching_reward)
                action_matching_reward_lists.append(action_matching_reward_list)
                pred_action_lists.append(pred_action_list)
                pred_think_lists.append(pred_think_list)
                
        
            # for tracking
            max_reward = max(action_matching_rewards)
        collected_stats[max_reward] += 1  
        
        print(f'Task: {task_id}\nGoal: {goal}')
        print('Action matching decompose: ', action_matching_reward_lists)
        print('Total reward: ', action_matching_rewards)
        print('Predicted actions: ', pred_action_lists)
        print('Reference action: ', ref_semantic_action)
        print('Collected data reward distribution', collected_stats)
        print('='*50)
        
        rephrase_sample = {
            'task_id': task_id,
            'task_timestep': int(step),
            'domain_info': domain_info,
            'goal': goal,
            'history': history,
            'observation': observation,
            'ref_action': ref_action, 
            'ref_semantic_action': ref_semantic_action,
            'predicted_actions': pred_action_lists,
            'action_matching_rewards': action_matching_rewards,
            'action_matching_reward_lists': action_matching_reward_lists,
            'candidates': [{'reasoning': rep['think'],
                            'extraction': rep['obs_repr'],
                            'action_matching_list': action_matching_list,
                            'action_matching': int(action_matching),
                            'predicted_action': pred_action_list,
                            'predicted_thought': pred_think_list
                            } 
                            for (rep, 
                                action_matching, 
                                action_matching_list, 
                                pred_action_list, 
                                pred_think_list
                                ) in zip(
                                    candidates, 
                                    action_matching_rewards, 
                                    action_matching_reward_lists, 
                                    pred_action_lists, 
                                    pred_think_lists)
                                ]
            }
        
        ####################
        gpt_cost = 2.5*gpt_input_tokens/1e+6 + 10*gpt_output_tokens/1e+6
        genai_cost = 0.15*genai_input_tokens/1e+6 + 1*genai_output_tokens/1e+6
        ant_cost = 3*ant_input_tokens/1e+6 + 15*ant_output_tokens/1e+6
        total_cost = gpt_cost + genai_cost + ant_cost
        print(f'Estimated GPT cost: ${gpt_cost}')
        print(f'Estimated GENINI cost: ${genai_cost}')
        print(f'Estimated CLAUDE cost: ${ant_cost}')
        print(f'Estimated total cost: ${total_cost}')
        ###########################################################################  
        collected_data.append(rephrase_sample)
        with open(f'webarena_data/sampled_contextualization.json', 'w') as f:
            json.dump(collected_data, f, indent=4)  
        
        
if __name__ == '__main__':
    import pandas as pd
    traj = pd.read_csv('webarena_data/seed_traj.csv')
    main(traj, num_candidates=3)



