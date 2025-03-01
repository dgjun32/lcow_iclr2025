
import os 
import re
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import torch
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import openai
from openai import OpenAI



def clean_obs(obs):
    return obs.replace('[button]', '[').replace('[button_]', ']')


################################# Util functions for meta prompt construction ###############################
def get_rephrase_system_prompt():
    prompt = '''You are an agent tasked with extracting and rephrasing a subset of the webpage's observations based on the content of the page and user instructions. 
Perform the following tasks based on the provided [Information source], including user instructions, interaction history, and the AXTree observation at the current time step. 
First, provide high-level reasoning for the next action by analyzing the provided information. 
Second, extract a few relevant elements based on your high-level reasoning.'''
    return prompt


# meta prompt for Rephraser LM
def return_meta_prompt_repr_3_hf(goal, obs, previous_actions):
    rephrase_system_prompt = f'''You are an agent tasked with extracting and rephrasing a subset of the webpage's observations based on the content of the page and user instructions. 
'''

    rephrase_prompt = f'''[General instructions]
You are currently on the online shopping website.
Your task is to generate a "Reasoning" and a "Rephrased observation" based on the provided inputs.

First, review the "User instruction" and "History of interactions" and, then, generate the "Reasoning".
Analyze the progress made so far, and provide a rationale for the next steps needed to efficiently accomplish the user instruction on the online shopping website.

Second, rephrase the "AXTree observation at the current time step" into a "Rephrased observation".
Select a subset of the AXTree observation that is necessary for completing the user instruction.

[Information source]
# User instruction
{goal}

# History of interactions
{previous_actions}

# AXTree observation at the current time step
{obs}
'''
    return rephrase_prompt, rephrase_system_prompt


def return_meta_prompt_repr_3(goal, obs, previous_actions):
    prompt = f'''
The current webpage on the web shopping site is described in the observation.
Evaluate the current progress based on previous actions and current observation.
Determine the next action by reasoning based on goal and progress.
Condense the observation into a concise format, highlighting clickable buttons indicated by [].
Ensure the summary includes only elements relevant to the goal and not already covered in previous actions.
Focus on clickable buttons indicated as [].

Here are a few examples. Make sure to follow the format exhibited in the examples.

**goal**: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
**previous actions**: 
1. search[3 ounce bright citrus deodorant sensitive skin]
**current observation**:
[ Back to Search ] 
Page 1 (Total results: 50) 
[ Next > ] 
[ B078GWRC1J ] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[ B078GTKVXY ] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[ B08KBVJ4XN ] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95 

**rephrased observation**:
Progress: I searched the keyword '3 ounce bright citrus deodorant sensitive skin' to see the relvant items, And now I am looking at the item list.
Reasoning: the next step is to choose an item satisfying the specification of bright citrus deodorant.
I can focus on:
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99


**goal**: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
**previous actions**:
1. search[3 ounce bright citrus deodorant sensitive skin]
2. click[B078GWRC1J]
**current observation**:
[ Back to Search ]
[ < Prev ] 
size
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ] 

**rephrased observation**:
Progress: I searched and and clicked the item seems to be most relevant to the goal specification. I am looking at the option list. 
Reasoning: As the goal requires 3-ounce bottle, I can focus on the size option.
I can focus on:
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]


**goal**: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
**previous actions**: 
1. search[3 ounce bright citrus deodorant sensitive skin]
2. click[B078GWRC1J]
3. click[3 ounce (pack of 1)]
**current observation**:
You have clicked 3 ounce (pack of 1).
[ Back to Search ] 
[ < Prev ] 
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ]
[ Features ]
[ Reviews ]
[ Buy Now ]

**rephrased observation**:
Progress: I searched and and clicked the item id. Among the option list, and I clicked size option.
Reasoning: According to the progress, I have to focus on the scent option as a next step.
I can focus on:
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]


**goal**: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
**previous actions**:
1. search[3 ounce bright citrus deodorant sensitive skin]
2. click[B078GWRC1J]
3. click[3 ounce (pack of 1)]
4. click[bright citrus]
**current observation**:
You have clicked 3 ounce (pack of 1).
You have clicked bright citrus.
[ Back to Search ]
[ < Prev ]
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ] 

**rephrased observation**:
Progress: Based on **observation** and **previous actions**, I clicked size option and scent option.
Reasoning: As there is no more options to select and I met all requirements specified in the goal, next step is to buy the item.
I can focus on:
[ Buy Now ]

Here is the task.


**goal**:
{goal}
**previous actions**:
{previous_actions}
**current observation**: 
{obs}

**rephrased observation**:
'''
    return prompt

##### Utils functions for rephrasing observations in the "interaction history of observation and action e.g., (o_1, a_1, o_2, a_2, ..., o_t)"

def convert_obs_in_history_examples_v2(few_shot, backbone):
    goal = few_shot.split('Instruction:\n')[-1].split('\n')[0]
    # extract all observations in history
    observations = re.findall(r'Observation:\s+(.*?)\n\nAction:', few_shot, re.DOTALL)
    actions = re.findall(r'Action:\s+(.*?)\n\nObservation:', few_shot, re.DOTALL)
    rephrased_observations = []

    # rephrase the observations using Rephrase LM
    previous_actions = ''
    for i, (obs, action) in enumerate(zip(observations, actions)):
        previous_actions += f'{i+1}. {action}\n'
        meta_prompt = return_meta_prompt_repr_3(goal, obs, previous_actions)
        obs_repr = api_llm_inference(meta_prompt, backbone, max_new_tokens=1000)
        rephrased_observations.append(obs_repr)

    # insert all rephrased observations to original history
    parts = re.split(r'\n\nObservation:\s+', few_shot)
    new_few_shot = parts[0]
    for i, obs_repr in enumerate(rephrased_observations, 1):
        new_few_shot += "\n\nObservation:\n" + obs_repr + "\n\nAction:" + parts[i].split("Action:", 1)[1]
    return new_few_shot


def convert_obs_in_history_inputs_v2(history, cur_obs_repr, backbone):
    goal = history.split('Instruction:\n')[-1].split('\n')[0]
    # extract all observations in history
    observations = re.findall(r'Observation:\s+(.*?)\n\nAction:', history, re.DOTALL)
    actions = re.findall(r'Action:\s+(.*?)\n\nObservation:', history, re.DOTALL)
    rephrased_observations = []

    # rephrase the observations using Rephrase LM
    previous_actions = ''
    for i, (obs, action) in enumerate(zip(observations, actions)):
        previous_actions += f'{i+1}. {action}\n'
        meta_prompt = return_meta_prompt_repr_3(goal, obs, previous_actions)
        obs_repr = api_llm_inference(meta_prompt, backbone, max_new_tokens=1000)
        rephrased_observations.append(obs_repr)
    if len(rephrased_observations) >= 1:
        rephrased_observations[-1] = cur_obs_repr

    # insert all rephrased observations to original history
    parts = re.split(r'\n\nObservation:\s+', history)
    new_few_shot = parts[0]
    for i, obs_repr in enumerate(rephrased_observations, 1):
        new_few_shot += "\n\nObservation:\n" + obs_repr + "\n\nAction:" + parts[i].split("Action:", 1)[1]
    return new_few_shot


def hf_convert_obs_in_history_examples_v2(base_model, tokenizer, few_shot):
    goal = few_shot.split('Instruction:\n')[-1].split('\n')[0]
    # extract all observations in history
    observations = re.findall(r'Observation:\s+(.*?)\n\nAction:', few_shot, re.DOTALL)
    actions = re.findall(r'Action:\s+(.*?)\n\nObservation:', few_shot, re.DOTALL)
    rephrased_observations = []

    # rephrase the observations using Rephrase LM
    previous_actions = ''
    for i, (obs, action) in enumerate(zip(observations, actions)):
        previous_actions += f'{i+1}. {action}\n'
        meta_prompt = return_meta_prompt_repr_3_hf(goal, obs, previous_actions)
        obs_repr = hf_llm_rephrase(base_model, tokenizer, meta_prompt)
        rephrased_observations.append(obs_repr)

    # insert all rephrased observations to original history
    parts = re.split(r'\n\nObservation:\s+', few_shot)
    new_few_shot = parts[0]
    for i, obs_repr in enumerate(rephrased_observations, 1):
        new_few_shot += "\n\nObservation:\n" + obs_repr + "\n\nAction:" + parts[i].split("Action:", 1)[1]
    return new_few_shot


def hf_convert_obs_in_history_inputs_v2(base_model, tokenizer, history, cur_obs_repr, system_prompt):
    '''
    history: interaction history where the observation is not rephrased
    cur_obs_repr: rephrased current observation
    '''
    goal = history.split('Instruction:\n')[-1].split('\n')[0]
    # extract all observations in history
    observations = re.findall(r'Observation:\s+(.*?)\n\nAction:', history, re.DOTALL)
    actions = re.findall(r'Action:\s+(.*?)\n\nObservation:', history, re.DOTALL)
    rephrased_observations = []

    # rephrase the observations using Rephrase LM
    previous_actions = ''
    for i, (obs, action) in enumerate(zip(observations, actions)):
        previous_actions += f'{i+1}. {action}\n'
        meta_prompt, system_prompt = return_meta_prompt_repr_3_hf(goal, obs, previous_actions)
        obs_repr = hf_llm_rephrase(base_model, tokenizer, meta_prompt, system_prompt)
        rephrased_observations.append(obs_repr)
    if len(rephrased_observations) >= 1:
        rephrased_observations[-1] = cur_obs_repr

    # insert all rephrased observations to original history
    parts = re.split(r'\n\nObservation:\s+', history)
    new_few_shot = parts[0]
    for i, obs_repr in enumerate(rephrased_observations, 1):
        new_few_shot += "\n\nObservation:\n" + obs_repr + "\n\nAction:" + parts[i].split("Action:", 1)[1]
    return new_few_shot

################################# Util functions for GEMINI inference ###################################

class LLaMAChatModel:
    def __init__(self, model_name, max_new_tokens):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(
            api_key = os.environ['LLAMA_API_KEY'],
            base_url = "https://api.deepinfra.com/v1/openai"
            )
    
    def invoke(self, chat_messages):
        chat_messages = [{'role':'user', 'content': chat_messages}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            temperature=0.0,
            max_tokens=self.max_new_tokens
        )
        output = response.choices[0].message
        return output
    
def make_chat_model(model_name, temperature=0.0, max_new_tokens=100):
        if model_name.startswith("openai"):
            _, model_name = model_name.split("/")
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
        
        elif model_name.startswith("anthropic"):
            _, model_name = model_name.split("/")
            return ChatAnthropic(
                model_name=model_name,
                anthropic_api_key=os.environ['ANTHROPIC_API_KEY'],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
        
        elif model_name.startswith("googleai"):
            _, model_name = model_name.split("/")
            return ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
        
        elif model_name.startswith("meta"):
            return LLaMAChatModel(model_name, max_new_tokens)
        

def api_llm_inference(prompt, model_name, max_new_tokens):
    model = make_chat_model(model_name, 
                            temperature=0.0, 
                            max_new_tokens=max_new_tokens)  

    output = model.invoke(prompt).content
    return output
    


    
############################# Utils functions for Huggingface LLM inference ###########################

def load_hfmodel(ckpt=None):
    if ckpt == None:
        path = 'microsoft/Phi-3-mini-128k-instruct'
    else:
        path = ckpt

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



class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [32007]):#[58, 4794, 60]): #32007 for phi-3 128009
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

# llm inference pipeline (state -> action)
def hf_llm_rephrase(base_model, tokenizer, user_input, system_prompt):
    '''
    base_model: AutoModelforCausalLM
    tokenizer: AutoTokenizer
    state: str
    '''
    chat = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_input},
        ]
    input_ids = tokenizer.apply_chat_template(
                            chat, 
                            tokenize=True, 
                            add_generation_prompt=True, 
                            return_tensors="pt",
                            add_special_tokens=False, 
                            ).to('cuda')
    output_ids = base_model.generate(
        input_ids,
        max_new_tokens=2048,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria = [EosListStoppingCriteria()],
        num_beams=1,
        use_cache=True,
        temperature=None,
        top_p = None,
    ).squeeze(0)

    input_lens = input_ids.shape[1]
    output_ids = output_ids[input_lens:]
    obs_repr = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


    return obs_repr

def hf_llm_batch_rephrase(base_model, tokenizer, states, system_prompt):
    '''
    base_model: AutoModelforCausalLM
    tokenizer: AutoTokenizer
    state: str

    '''
    chat = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': states[0]},
        ]
    input_ids = tokenizer.apply_chat_template(
                            chat, 
                            tokenize=True, 
                            add_generation_prompt=True, 
                            return_tensors="pt",
                            add_special_tokens=False, 
                            ).to('cuda')
    
    output_ids = base_model.generate(
        input_ids,
        max_new_tokens=2048,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=len(states),
        tokenizer = tokenizer,
        stop_strings='[END]',
        #stopping_criteria = [EosListStoppingCriteria()],
        use_cache=True,
    )
    outputs = []
    for i in range(output_ids.shape[0]):
        input_len = input_ids.shape[1]
        output_id = output_ids[i, input_len:]
        output = tokenizer.decode(output_id,
                                  skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=True)
        outputs.append(output.split('[END]')[0])
    return outputs



def hf_llm_batch_rephrase_sampling(base_model, tokenizer, states,
                                   temperature=1.0,
                                   top_k=50,
                                   top_p=1.0):
    '''
    base_model: AutoModelforCausalLM
    tokenizer: AutoTokenizer
    state: str
    '''
    
    model_input = tokenizer(states, 
                            return_tensors='pt', 
                            max_length = 8192,
                            padding=True,
                            truncation=True,
                            add_special_tokens=False, 
                            ).to('cuda')
    
    output_ids = base_model.generate(
        **model_input,
        max_new_tokens=400,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria = [EosListStoppingCriteria()],
        use_cache=True,
    )

    outputs = []
    for i in range(output_ids.shape[0]):
        input_len = model_input['input_ids'][i].shape[0]
        output_id = output_ids[i, input_len:]
        output = tokenizer.decode(output_id,
                                  skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=True)
        outputs.append(output)

    
    return outputs





