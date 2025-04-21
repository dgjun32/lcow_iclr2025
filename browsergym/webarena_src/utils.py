import re
import torch
import transformers
from transformers import StoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_eos_token = True
    print('Loaded Model and Tokenizer')

    return base_model, tokenizer


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [58, 4794, 60]): #32007 for phi-3 128009
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
    # if token size exceeds 60000:
    if input_ids.shape[1] > 60000:
        input_ids = input_ids[:,:60000]
        template_tokens = torch.tensor([[128009, 128006, 78191, 128007, 271]]).to('cuda')
        input_ids = torch.cat([input_ids, template_tokens], dim=1)
    
    print(input_ids.shape)

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



### Util functions for parsing action command ######################

# Function to find and extract link information by ID
def extract_action_semantics(data, action_id):
    # Regex to find the pattern [ID] link '...'
    pattern = r"\[" + str(action_id) + r"\] [^\']* '([^']*)'"
    match = re.search(pattern, data)
    if match:
        return match.group(0)  # Returns the full line that contains the matched ID
    return "Link not found"


def add_action_semantic(action, obs):
    #obs = flatten_axtree_to_str(obs)
    try:
        if action.startswith('noop'):
            sem_action = 'No operation'
        elif action.startswith('scroll'):
            horiz = float(action.split('(')[1].split(',')[0].strip())
            vertic = float(action.split(',')[1].split(')')[0].strip())
            if horiz == 0 and vertic > 0:
                sem_action = 'scroll down'
            elif horiz == 0 and vertic < 0: 
                sem_action = 'scroll up'
            elif horiz > 0 and vertic == 0:
                sem_action = 'scroll right'
            elif horiz < 0 and vertic == 0:
                sem_action = 'scroll left'
            elif horiz > 0 and vertic > 0:
                sem_action = 'scroll right and down'
            elif horiz > 0 and vertic < 0: 
                sem_action = 'scroll right and up'
            elif horiz < 0 and vertic > 0:
                sem_action = 'scroll left and down'
            elif horiz < 0 and vertic < 0:
                sem_action = 'scroll left and up'
        
        elif action.startswith('send_msg_to_user'):
            sem_action = action
        
        elif action.startswith('fill'):
            elem_id = action.split('(')[1].split(',')[0].strip("'")
            value = action.split(',')[1].split(')')[0]
            element = extract_action_semantics(obs, elem_id)
            sem_action = f'fill {value} in {element}'

        elif action.startswith('select_option'):
            elem_id = action.split('(')[1].split(',')[0].strip("'")
            option = action.split(',')[1].split(')')[0]
            element = extract_action_semantics(obs, elem_id)
            sem_action = f'select {option} from {element}'

        elif action.startswith('click'):
            pattern = r"click\('([^']*)'"
            #pattern = r"click\((?:.*?=\s*)?'([^']*)'"
            match = re.search(pattern, action)
            if match:
                elem_id = match.group(1)
            else:
                raise Exception()
            element = extract_action_semantics(obs, elem_id)
            sem_action = f'click {element}'

        elif action.startswith('dbclick'):
            elem_id = action.split('(')[1].split(')')[0].split(',')[0].strip("'")
            element = extract_action_semantics(obs, elem_id)
            sem_action = f'double click {element}'

        elif action.startswith('hover'):
            elem_id = action.split('(')[1].split(')')[0].split(',')[0].strip("'")
            element = extract_action_semantics(obs, elem_id)
            sem_action = f'hover the cursor on {element}'

        elif action.startswith('press'):
            elem_id = action.split('(')[1].split(')')[0].split(',')[0].strip("'")
            element = extract_action_semantics(obs, elem_id)
            key = action.split('(')[1].split(')')[0].split(',')[1].strip()
            sem_action = f'press {key} keys while focusing on {element}'

        elif action.startswith('focus'):
            elem_id = action.split('(')[1].split(')')[0].split(',')[0].strip("'")
            element = extract_action_semantics(obs, elem_id)
            sem_action = f'focus on {element}' 

        elif action.startswith('clear'):
            elem_id = action.split('(')[1].split(')')[0].split(',')[0].strip("'")
            element = extract_action_semantics(obs, elem_id)
            sem_action = f'clear contents in {element}'
        
        else:
            sem_action = action
    except:
        sem_action = action

    return sem_action    


def remove_all_error_logs(log_text):
    start_marker = "=========================== logs ==========================="
    end_marker = "============================================================"

    # Continue to remove all error logs while both markers are present
    while True:
        start_index = log_text.find(start_marker)
        end_index = log_text.find(end_marker)
        
        if start_index != -1 and end_index != -1:
            # Remove the log section
            log_text = log_text[:start_index] + log_text[end_index + len(end_marker):]
        else:
            # If no more logs are found, break the loop
            break
    return log_text


def reformat_action_prompt(prompt):
    prompt = prompt.replace("Description: Sends a message to the user.", "Description: Sends a message to the user. You should send a short answer as a message and do not ask questions through message.")
    prompt = prompt.replace("\n        send_msg_to_user(\'Based on the results of my search, the city was built in 1751.\')", 
                            "\n        send_msg_to_user(\'the city was built in 1751.\')\n        send_msg_to_user(\'Yes\')\n        send_msg_to_user(\'No\')\n        send_msg_to_user(\'31112\')\n        send_msg_to_user(\'Yoshua Bengio\')")
    return prompt