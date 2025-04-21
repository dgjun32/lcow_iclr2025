# WebShop Experiment

## 1. Setup
Setup the environment (encouraged to use Python 3.10)
```
cd webshop
conda create -n webshop python==3.10
conda activate webshop
pip install -r requirements.txt
```
Setup API keys for LLMs
```
export GEMINI_API_KEY = 'your googleai api key'
export OPENAI_API_KEY = 'your openai api key'
export ANTHROPIC_API_KEY = 'your anthropic api key'
export LLAMA_API_KEY = 'your llama api key (register in deepinfra.ai)'
```

Additionally, you have to login to wandb and huggingface.
```
wandb login
huggingface-cli login
```

## 2. Quick Walkthrough for Webshop
```python
import gym
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE

# intialize environment
env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=DEBUG_PROD_SIZE)

# reset environment (each task idx corresponds to different human instruction)
task_idx = 0 # (0 ~ 6510)
obs, info = env.reset(session=task_idx)
print(obs)
action = 'search[iphone 13 pro]'
# reward is given at the end of the episode (when you click[buy now]), based on the task information.
obs, reward, done, info = env.step(action)
print(obs)
```

## 3. Experiment 

### Evaluating baseline
As a backbone, you may choose backbone as one of `openai/gpt-4o-2024-08-06`, `googleai/gemini-1.5-flash-002`, `anthropic/claude-3.5-sonnet-20240620`.
* without contextualization (Raw observation)
```
python src/webshop_baseline_eval.py \
--num_tasks 500 \
--backbone [backbone llm]
```

* self-contextualization
```
python src/webshop_baseline_eval.py \
--num_tasks 500 \
--backbone [backbone llm] \
--rephrase 
```

### LCoW (iteraion = 1)

```
# Sampling contextualized observations from seed demonstration
python src/collect_data.py --num_samples 4

# Training contextualization module
python src/train.py --iter 0

# Evaluating LCoW (iter = 1)
python src/webshop_lcow_eval.py \
--num_tasks 500 \
--iter 0 \ 
--ckpt_step [checkpoint step] \
--backbone [backbone llm]
```

### LCoW (iteraion > 1)

```
# Collecting successful trajectories from training environment
python src/traj_collection.py \
--backbone googleai/gemini-1.5-flash-002 \
--iter [iteration] \
--ckpt_step [checkpoint_step (prev iteration)]

# Sampling contextualized observations
python src/collect_data_iter.py 
--iter [iteration] \
--num_samples 4 \
--ckpt_step [checkpoint_step (prev iteration)]

# Training contextualization module
python src/train.py 
--iter [iteration] \
--start_ckpt_step [checkpoint_step (prev iteration)]

# Evaluating LCoW (iteration > 1)
python src/webshop_lcow_eval.py 
--num_tasks 500 \ 
--iter [iteration] \
--ckpt_step [checkpoint_step (current iteration)] \
--backbone [bacbone llm]
```

### Trajectory log & Dataset
We open-source webshop contextualization [dataset](https://drive.google.com/drive/folders/1SWTpVkzB6z1yvYzjYwNS3UWkOvrPgClb?usp=share_link) we collected for training the model.
We also opensource webshop experiment log. You can download [logs](https://drive.google.com/drive/folders/1Y98YTMMkWMs4p9LsZQhf1rxYe6hwqfKj?usp=share_link) regarding our WebShop experiment.

