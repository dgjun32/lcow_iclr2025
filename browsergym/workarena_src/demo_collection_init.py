import pandas as pd
import tqdm
from tqdm import tqdm
import dataclasses
import browsergym
import gymnasium as gym
import browsergym  # register webarena tasks as gym environments

import json
import os
import re
import pandas as pd
from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str
from dataclasses import asdict, dataclass, field
import traceback
from warnings import warn
from langchain.schema import HumanMessage, SystemMessage
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.experiments import Agent, AbstractAgentArgs
import sys
from utils import add_action_semantic
import base64
import openai
from openai import AzureOpenAI

sys.path.append('./')
from demo_agent.agents.legacy.dynamic_prompting import Flags
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs
from demo_agent.agents.legacy import dynamic_prompting
from demo_agent.agents.legacy.utils.llm_utils import ParseError, retry
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs

from workarena_src.utils import reformat_action_prompt
        
        
def forward_model(messages): 
    client = AzureOpenAI(
                azure_endpoint = "",
                api_key=base64.b64decode("").decode("utf-8"),
                api_version="2024-09-01-preview"
                )

    response = client.chat.completions.create(
                        model='o1-preview',
                        messages=messages,
                        max_completion_tokens = 4096,
                        )

    return response.choices[0].message.content

@dataclass
class GenericAgentArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    max_retry: int = 4

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, 
            flags=self.flags, 
            max_retry=self.max_retry, 
        )

class GenericAgent(Agent):

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Augment observations with text HTML and AXTree representations, which will be stored in
        the experiment traces.
        """

        obs = obs.copy()
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])

        return obs

    def __init__(
        self,
        chat_model_args: ChatModelArgs = None,
        flags: dynamic_prompting.Flags = None,
        max_retry: int = 4,
    ):
        self.chat_model_args = chat_model_args if chat_model_args is not None else ChatModelArgs()
        self.flags = flags #if flags is not None else dynamic_prompting.Flags()
        self.max_retry = max_retry

        self.chat_llm = chat_model_args.make_chat_model()
        self.action_set = dynamic_prompting._get_action_space(self.flags)

        # consistency check
        if self.flags.use_screenshot:
            if not self.chat_model_args.has_vision():
                warn(
                    """\

Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                self.flags.use_screenshot = False

        # reset episode memory
        self.obs_history = []
        self.actions = []
        self.memories = []
        self.thoughts = []

    def get_action(self, obs):
        obs = self.obs_preprocessor(obs)
        self.obs_history.append(obs)

        main_prompt = dynamic_prompting.MainPrompt(
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            flags=self.flags,
        )

        # Determine the minimum non-None token limit from prompt, total, and input tokens, or set to None if all are None.
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        
        prompt = dynamic_prompting.fit_tokens(
            main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            model=self.chat_llm
        )
        
        chat_messages = [
            SystemMessage(content=dynamic_prompting.SystemPrompt().prompt),
            HumanMessage(content=prompt),
        ]
        num_tokens = len(prompt) / 4
        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""
        
        
        """ temporary
        try:
            
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"action": None}
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry
        try:
            think = ans_dict['think']
        except:
            think = None
        """
        ######## modified ##############
        prompt = prompt[0]['text']
        prompt = reformat_action_prompt(prompt)
        splits = prompt.split('\n# ')
        action_history = '# ' + splits[2]
        domain_info = prompt.split('## AXTree')[1].split('\n')[1].split("'")[1]
        chat_messages = [{'role': 'user', 'content': prompt}]
        model_output = forward_model(chat_messages)
        think = model_output.split('<think>')[-1].split('</think>')[0].strip()
        action = model_output.split('<action>')[-1].split('</action>')[0].strip()
        #################################
        """
        semantic_action = add_action_semantic(ans_dict['action'], prompt)
        """
        semantic_action = add_action_semantic(action, prompt)
        think_action = f'think: {think}\naction: {semantic_action}'
        self.actions.append(semantic_action)
        #self.actions.append(ans_dict["action"])
        self.memories.append(None)
        self.thoughts.append(think)

        #ans_dict["chat_messages"] = [m.content for m in chat_messages]
        #ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        #action = ans_dict['action']
        return action, think, prompt, num_tokens, action_history, domain_info


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arguments for Webshop Evaluation using Fine-tuned Rephraser LM')
    # Add argumen
    parser.add_argument('--backbone', type=str, help='type of the backbone', default='googleai/gemini-1.5-flash-002')
    parser.add_argument('--max_steps', type=int, help='number of the maximum steps', default=10)
    args = parser.parse_args() 


    # Flags for environment features (Information w.r.t flags used in Webarena leaderboard is not provided.. )
    flags=Flags(
            action_space='bid',
            use_html=False,
            use_ax_tree=True,
            use_thinking=True,  # "Enable the agent with a memory (scratchpad)."
            use_error_logs=True,  # "Prompt the agent with the error logs."
            use_memory=False,  # "Enables the agent with a memory (scratchpad)."
            use_history=True,
            use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
            use_past_error_logs=True,  # "Prompt the agent with the past error logs."
            use_action_history=True,  # "Prompt the agent with the action history."
            multi_actions=False,
            use_abstract_example=True,  # "Prompt the agent with an abstract example."
            use_concrete_example=True,  # "Prompt the agent with a concrete example."
            use_screenshot=True,
            enable_chat=False,
            demo_mode="default",
            html_type='dom_txt',
        )

    chat_model_args = ChatModelArgs(
                model_name=args.backbone,#'openai/gpt-4o-2024-05-13',#'googleai/gemini-1.5-flash-latest',
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_488,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=2_000,
                temperature = 0.0,
                top_p = 1e-8  # "Maximum total tokens for the chat model."
            )

    ########### chosen tasks #####################
    from browsergym.workarena import ALL_WORKARENA_TASKS
    from browsergym.workarena import get_all_tasks_agents
    from browsergym.core.env import BrowserEnv
    
    L1_TASKS = get_all_tasks_agents(filter = 'l1')
    task_types = set([t[0] for t in L1_TASKS])
    ####################################################
    rollouts = []
    tokens = 0
    rewards = []
    tasks = []
    collected_data = {'task_id':[],
                      'domain_info':[],
                      'goal':[], 
                      'step':[],
                      'observation':[],
                      'action_history':[],
                      'think':[],
                      'action':[],
                      'semantic_action':[],
                      'success': []}
    
    for task in task_types:
        # seed for training tasks (5~20)
        task_id = task.get_task_id()
        #if 'filter' in task_id:
        for seed in range(5, 20):
            tasks.append((task, seed))

    for task_id, task_seed in tqdm(tasks):#(463, 678):
        print('task: ', task_id)
        print('seed: ',task_seed)
        agent = GenericAgentArgs(chat_model_args=chat_model_args, flags=flags).make_agent()
        
        #env = gym.make(os.path.join('browsergym', task_id), #task_id, 
        #            headless=True,
        #            )
        env = BrowserEnv(task_entrypoint = task_id, headless = True)
        ############## reset environment #############################
        for attempt in range(10):
            try:
                obs, info = env.reset(seed = task_seed)
                # If no error occurs, break out of the loop
                break
            except Exception as e:
                # If the max number of attempts is reached, raise the error
                if attempt == 9:
                    raise e
        #############################################################
        goal = obs['goal'] 
        terminated = False
        action_log = []
        trial = 0
        rollout = f'Goal: {goal}\nObs:\n'
        steps, task_ids, goals, domain_infos, observations, rephrase_observations, rephrase_thinks, action_histories, interaction_histories, successes, thinks, actions, semantic_actions = [], [], [], [], [], [], [], [], [], [], [], [], []
        print('-'*50)
        print('Goal: ', obs['goal'])
        while not terminated and trial <= args.max_steps:
            action, think, proc_prompt, num_tokens, action_history, domain_info = agent.get_action(obs)
            action_log.append(action)
            tokens += num_tokens
            axtree_obs = flatten_axtree_to_str(obs['axtree_object']) 
            rollout += f'{axtree_obs}\n\nThink: {think}\n\nAction: {action}\n\nObs:\n'
            
            ########### log the data incrementally ###########
            task_ids.append(f"{task_id}_{task_seed}")
            goals.append(goal)
            domain_infos.append(domain_info)
            steps.append(trial)
            observations.append(axtree_obs)
            action_histories.append(action_history)
            thinks.append(think)
            actions.append(action)
            semantic_action = add_action_semantic(action, proc_prompt)
            semantic_actions.append(semantic_action)
            
            ############# Transition to next state #############
            for attempt in range(10):
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    # If no error occurs, break out of the loop
                    break
                except Exception as e:
                    # If the max number of attempts is reached, raise the error
                    if attempt == 9:
                        raise e
            ###################################################
            try:
                print(f'Think {trial+1}: ', think)
                print(f'Action {trial+1}: ', action)
                semantic_action = add_action_semantic(action, proc_prompt)
                print(f'Action semantic {trial+1}: ', semantic_action)
                print(f"Exec result:\n{obs['last_action_error']}")
            except:
                pass
            trial += 1 

            # if consecutive 3 recent actions are the same, terminate the episode
            if len(action_log) >= 3 and (action_log[-1]==action_log[-2]==action_log[-3]):
                terminated = True
            ####################################################################
        
        collected_data['task_id'] += task_ids
        collected_data['goal'] += goals
        collected_data['domain_info'] += domain_infos
        collected_data['step'] += steps
        collected_data['observation'] += observations
        collected_data['think'] += thinks
        collected_data['action'] += actions
        collected_data['semantic_action'] += semantic_actions
        collected_data['action_history'] += action_histories
        collected_data['success'] += [reward]*len(task_ids)
        collected_df = pd.DataFrame(collected_data)
        collected_df.to_csv(f'workarena_data/workarena_seed_demo_o1.csv')
        rewards.append(reward)
        print('Rewards: ', rewards)
        print('tokens: ', tokens)
    success = [1 if r==1 else 0 for r in rewards]
    print('Success rate: ', sum(success)/len(success))
    print('Average rewards: ', sum(rewards)/len(rewards))
    print('Rewards: ', rewards)
    print('# tokens: ', tokens)
    
    

    