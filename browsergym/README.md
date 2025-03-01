# WorkArena / WebArena Experiment

[[Setup]](#setup) ♦ [[Experiments]](#experiments)


# Setup

## LLM api keys
```
export GEMINI_API_KEY = 'your googleai api key'
export OPENAI_API_KEY = 'your openai api key'
export ANTHROPIC_API_KEY = 'your anthropic api key'
export LLAMA_API_KEY = 'your llama api key (register to deepinfra.ai)'
```

## Dependencies
Note that we utilize BrowserGym interface for experiment in WorkArena and WebArena.
For more details about BrowserGym, click this [link](https://github.com/ServiceNow/BrowserGym).
```
conda create -n browsergym python==3.10
conda activate browsergym
pip install -r requirements.txt
playwright install
```
Finally, each benchmark comes with its own specific setup that requires to follow additional steps.
 - for webarena, see [webarena/README.md](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/webarena/README.md)
 - for workarena, see [WorkArena](https://github.com/ServiceNow/WorkArena)


## Dataset
We open-source web page contextualization dataset we collected for training the model.
You may download [webarena_data.zip](), [workarena_data.zip]() here.
After download, please unzip the file under the current directory.

## Trajectory log
We also open-source the experiment log containing rollout trajectories for ease of analysis.
You may download [webarena_results.zip](), [workarena_data.zip]() here.
After download, please unzip the file under the current directory.

# Experiments
We utilize 5 LLMs for evaluation in WorkArena. You may enter the following model alias in commandline arguments.
* GPT-4o: `openai/gpt-4o-2024-08-06`
* Claude-3.5-Sonnet: `anthropic/claude-3-5-sonnet-20240620`
* Gemini-1.5-flash: `googleai/gemini-1.5-flash-002`
* Llama-3.1-70B: `meta-llama/Meta-Llama-3.1-70B-Instruct`
* Llama-3.1-8B: `meta-llama/Meta-Llama-3.1-8B-Instruct`

## WorkArena
### Raw observation
```
python workarena_src/webarena_eval_baseline.py \
    --backbone <backbone name> \
    --max_steps 20
```

### Self-contextualization
```
python webarena_src/webarena_eval_selfctx.py \
    --backbone <backbone name> \
    --max_steps 20
```

### LCoW
```
# 1. Sampling contextualized observations
python workarena_src/contextualization_sampling.py

# 2. Train contextualization module
python workarena_src/aggregate_data_init.py
python workarena_src/train.py --iter 0

# 3. Evaluation
python webarena_src/workarena_eval_lcow.py \
    --backbone <backbone name> \
    --max_steps 20 \
    --ckpt_step [checkpoint step]
```


## WebArena-Lite 
### Raw observation
```
python workarena_src/webarena_eval_baseline.py \
    --backbone <backbone name> \
    --max_steps 20
```

### Self-contextualization
```
python webarena_src/webarena_eval_selfctx.py \
    --backbone <backbone name> \
    --max_steps 20
```

### LCoW
```
# 1. Sampling contextualized observations
python webarena_src/contextualization_sampling.py

# 2. Train contextualization module
python webarena_src/aggregate_data_init.py
python webarena_src/train.py --iter 0

# 3. Evaluation (GPT-4o)
python webarena_src/webarena_eval_lcow.py \
    --backbone openai/gpt-4o-2024-08-06 \
    --max_steps 20 \
    --ckpt_step [checkpoint step]
```

