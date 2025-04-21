import json
import os
import sys
import time
import traceback
from warnings import warn

import torch
from tqdm import tqdm
import gymnasium as gym
import browsergym.webarena  # register webarena tasks as gym environments
from transformers import AutoModelForCausalLM, AutoTokenizer

from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str
from dataclasses import asdict, dataclass, field
from langchain.schema import HumanMessage, SystemMessage
from browsergym.core.action.base import AbstractActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.experiments import Agent, AbstractAgentArgs

sys.path.append('./')
from webarena_src.prompt import hf_format_rephrase_prompt, hf_format_action_prompt
from webarena_src.utils import load_hfmodel, hf_llm_rephrase, add_action_semantic, remove_all_error_logs, reformat_action_prompt
from demo_agent.agents.legacy.dynamic_prompting import Flags
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs
from demo_agent.agents.legacy import dynamic_prompting
from demo_agent.agents.legacy.utils.llm_utils import ParseError, retry
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs


@dataclass
class GenericAgentArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    rephrase_lm: AutoModelForCausalLM = None
    rephrase_tok: AutoTokenizer = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    max_retry: int = 4

    def make_agent(self, goal):
        return GenericAgent(
            chat_model_args=self.chat_model_args, 
            rephrase_lm=self.rephrase_lm,
            rephrase_tok=self.rephrase_tok,
            flags=self.flags, 
            max_retry=self.max_retry,
            goal=goal,
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
        rephrase_lm: AutoModelForCausalLM = None,
        rephrase_tok: AutoTokenizer = None,
        flags: dynamic_prompting.Flags = None,
        max_retry: int = 4,
        goal: str = None,
    ):
        self.chat_model_args = chat_model_args if chat_model_args is not None else ChatModelArgs()
        self.flags = flags if flags is not None else dynamic_prompting.Flags()
        self.max_retry = max_retry

        self.chat_llm = chat_model_args.make_chat_model()
        self.rephrase_lm = rephrase_lm,
        self.rephrase_tok = rephrase_tok,
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
        self.interaction_history = f"Task: {goal}\n\n"

    def get_action(self, obs, trial):

        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""
        
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

        
        # divide prompt into subsections
        splits = prompt.split('\n# ')
        instruction = splits[0]
        goal = instruction.split('## Goal:\n')[1].split('\n')[0]
        observation = '# ' + splits[1]
        history = '# ' + remove_all_error_logs(splits[2])
        domain_info = prompt.split('## AXTree')[1].split('\n')[1].split("'")[1]
        
        ######## 2. Contextualize observation ################ 
        rephrase_prompt, rephrase_system_prompt = hf_format_rephrase_prompt(
            domain_info, 
            goal, 
            '\n'.join(remove_all_error_logs(history).split('\n')[1:]).strip(), 
            '\n'.join(observation.split('\n')[1:]).strip()
            )
        
        rep_observation = hf_llm_rephrase(self.rephrase_lm[0], self.rephrase_tok[0], rephrase_prompt, rephrase_system_prompt)
        rep_observation = rep_observation.split('[END]')[0]
        self.interaction_history += f"Rephraser:\n{rep_observation}\n\nAgent: "
        interaction_history_out = self.interaction_history.replace(f"{rep_observation}\n\nAction:\n", "")
        
               
        ######## 3. get action from the agent LM ##################
        action_prompt = hf_format_action_prompt(instruction, history, rep_observation)
        num_input_tokens += len(action_prompt) / 4
        chat_messages = [
            SystemMessage(content=dynamic_prompting.SystemPrompt().prompt),
            HumanMessage(content=action_prompt),
        ]
        
        try:
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            ans_dict = {"action": None}
            ans_dict['think'] = None
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry
    
        action = ans_dict['action']
        try:
            think = ans_dict['think']
        except:
            think = None
        semantic_action = add_action_semantic(action, prompt)
        self.actions.append(semantic_action)
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        self.interaction_history += f"{action}\n\n"
        ##########################################################
        return action, rep_observation, prompt, ans_dict, history, interaction_history_out, observation


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WorkArena LCoW evaluation')
    
    parser.add_argument('--backbone', type=str, help='type of the action model backbone', default='googleai/gemini-1.5-flash-002')
    parser.add_argument('--iter', type=int, help='path the rephraser_model_backbone', default=0)
    parser.add_argument('--ckpt_step', type=int, help='training step of the rephraser model')
    parser.add_argument('--max_steps', type=int, help='number of the maximum steps', default=20)
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

    # argument for action model
    chat_model_args = ChatModelArgs(
                model_name=args.backbone,
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_488,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
                temperature=0.0
            )
    
    # load rephrase LM and Tokenizer
    rephrase_lm, rephrase_tok = load_hfmodel(f"ckpt/webarena_sft_iter_{args.iter}/checkpoint-{args.ckpt_step}")
    rephrase_lm = rephrase_lm.type(torch.bfloat16)
    if not os.path.exists(f'webarena_results/{args.backbone.split("/")[-1]}_iter{args.iter}'):
        os.makedirs(f'webarena_results/{args.backbone.split("/")[-1]}_iter{args.iter}')
    
    rollouts, tokens, rewards = [], 0, []
    
    # Load WebArena-Lite task indices
    with open("webarena_data/webarena_lite_taskidx", "r") as f: 
        task_indices = json.load(f)

    for task_idx in tqdm(task_indices):
        try:
            print('task: ', task_idx)
            env = gym.make(f"browsergym/webarena.{task_idx}", headless=True)
            obs, info = env.reset()
            goal = obs['goal']
            # define agent corresponding to the goal
            agent = GenericAgentArgs(chat_model_args=chat_model_args,
                                    rephrase_lm = rephrase_lm,
                                    rephrase_tok = rephrase_tok, 
                                    flags=flags, 
                                    ).make_agent(goal=goal)
            terminated, actions, trial, rollout = False, [], 0, f'Goal: {goal}\n\nObs:\n'
            print('-'*50)
            print('Goal: ', obs['goal'])
            while (not terminated) and trial <= args.max_steps:
                action, rep_obs, prompt, ans_dict, action_history, interaction_history, raw_observation = agent.get_action(obs, trial)
                actions.append(action)
                ############# Transition to next state #############
                obs, reward, terminated, truncated, info = env.step(action)
                ###################################################
                try:
                    print(f'Rephrased Obs {trial+1}:\n {rep_obs}\n')
                    think = ans_dict['think']
                    print(f'Think {trial+1}:\n {think}\n')
                    semantic_action = add_action_semantic(action, prompt)
                    print(f'Action {trial+1}: {action}\n')
                    print(f'Action semantic {trial+1}: {semantic_action}\n')
                    print('-'*10)
                except:
                    print(f'Rephrased Obs {trial+1}:\n {rep_obs}\n')
                    semantic_action = add_action_semantic(action, prompt)
                    print(f'Action {trial+1}: {action}\n')
                    print(f'Action semantic {trial+1}: {semantic_action}\n')
                    print('-'*10)
                rollout += f'{rep_obs}\n\nThink: {think}\n\nAction: {action}\n\nObs:\n'               
                trial += 1 

                # if consecutive 3 recent actions are the same, terminate the episode
                if len(actions) >= 3 and (actions[-1]==actions[-2]==actions[-3]):
                    terminated = True
            env.close()
            # incrementally save log
            rewards.append(reward)
            print('Rewards: ', rewards)
            rollout_dict = {'task_idx': f"{task_idx}", 'reward': reward, 'rollout': rollout}
            with open(f'webarena_results/{args.backbone.split("/")[-1]}_iter{args.iter}/webarena_{task_idx}.json', 'w') as f:
                json.dump(rollout_dict, f, indent=4)
        except Exception as e:
            print(e)
            rewards.append('NA')
            print('Rewards: ', rewards)   
            
    success = [1 if r==1 else 0 for r in rewards]
    print('Success rate: ', sum(success)/len(success))
    print('Rewards: ', rewards)
    
    
    
    