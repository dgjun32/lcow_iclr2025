
import os
import sys
import re
import json
import subprocess
import time
import traceback
from warnings import warn

import dataclasses
import browsergym
import gymnasium as gym
import browsergym
from tqdm import tqdm
from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str
from dataclasses import asdict, dataclass, field
from langchain.schema import HumanMessage, SystemMessage
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.experiments import Agent, AbstractAgentArgs

from utils import add_action_semantic, reformat_action_prompt

sys.path.append('./')
from demo_agent.agents.legacy.dynamic_prompting import Flags
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs
from demo_agent.agents.legacy import dynamic_prompting
from demo_agent.agents.legacy.utils.llm_utils import ParseError, retry
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs


@dataclass
class GenericAgentArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    max_retry: int = 4

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, 
            flags=self.flags, 
            max_retry=self.max_retry
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
        self.flags = flags if flags is not None else dynamic_prompting.Flags()
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
        ##### modification #####
        prompt = reformat_action_prompt(prompt)
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
        semantic_action = add_action_semantic(ans_dict['action'], prompt)
        self.actions.append(semantic_action)
        #self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        action = ans_dict['action']

        return action, ans_dict, prompt, num_tokens


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
            use_screenshot=False,
            enable_chat=False,
            demo_mode="default",
            html_type='dom_txt',
        )

    chat_model_args = ChatModelArgs(
                model_name=args.backbone,
                max_total_tokens=128_000,
                max_input_tokens=126_488,
                max_new_tokens=2_000,
                temperature = 0.0,
                )

    if not os.path.exists(f'workarena_results/{args.backbone.split("/")[-1]}_baseline/'):
        os.makedirs(f'workarena_results/{args.backbone.split("/")[-1]}_baseline/')
    
    rollouts = []
    tokens = 0
    rewards = []
    
    ## get Webarena-Lite tasks
    with open("webarena_data/webarena_lite_taskidx", "r") as f: 
        task_indices = json.load(f)

    for task_idx in tqdm(task_indices):
        try:
            print('task: ', task_idx)
            agent = GenericAgentArgs(chat_model_args=chat_model_args, flags=flags).make_agent()
            env = gym.make(f"browsergym/webarena.{task_idx}", headless=True)
            obs, info = env.reset()
            goal = obs['goal'] 
            terminated = False
            actions = []
            trial = 0
            rollout = f'Goal: {goal}\nObs:\n'
            print('-'*50)
            print('Goal: ', obs['goal'])
            while not terminated and trial <= args.max_steps:
                action, ans_dict, proc_prompt, num_tokens = agent.get_action(obs)
                actions.append(action)
                tokens += num_tokens
                try:
                    think = ans_dict['think']
                except:
                    think = None
                axtree_obs = flatten_axtree_to_str(obs['axtree_object']) 
                rollout += f'{axtree_obs}\n\nThink: {think}\n\nAction: {action}\n\nObs:\n'
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
                    print(f'Think {trial+1}: ', ans_dict['think'])
                    print(f'Action {trial+1}: ', action)
                    semantic_action = add_action_semantic(action, proc_prompt)
                    print(f'Action semantic {trial+1}: ', semantic_action)
                    print(f"Exec result:\n{obs['last_action_error']}")
                except:
                    pass
                trial += 1 

                # if consecutive 3 recent actions are the same, terminate the episode
                if len(actions) >= 3 and (actions[-1]==actions[-2]==actions[-3]):
                    terminated = True
                ####################################################################
            env.close()
            rollout_dict = {'task_idx': f"{task_idx}", 'reward': reward, 'rollout': rollout}
            with open(f'webarena_results/{args.backbone.split("/")[-1]}_baseline/webarena_{task_idx}.json', 'w') as f:
                json.dump(rollout_dict, f, indent=4)
            rewards.append(reward)
            print('Rewards: ', rewards)
            print('tokens: ', tokens)
            
        except Exception as e:
            print(e) 
            reward = 'N/A'
            rewards.append(reward)
            print('Rewards: ', rewards)
      
    success = [1 if r==1 else 0 for r in rewards]
    print('Success rate: ', sum(success)/len(success))
    print('# tokens: ', tokens)
    
    

    