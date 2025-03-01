import os
import time
import json
import warnings
import torch
import trl
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import wandb
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from tqdm import tqdm

import sys
sys.path.append('./')
from workarena_src.utils import load_hfmodel
#from hf_eval_rollout import run_eval_multiturn_repr, WebshopAgent

import peft 

#####################################################################

def main(cfg):
    # load dataset for training and validation
    train_df = pd.read_csv('workarena_data/bc_data.csv')
    train_dataset = Dataset(pa.Table.from_pandas(train_df))  
    val_df = pd.read_csv('workarena_data/validation_data.csv')
    val_dataset = Dataset(pa.Table.from_pandas(val_df)) 

    # load model and tokenizer

    base_model, tokenizer = load_hfmodel(cfg.model_name)
    tokenizer.padding_side = 'right'

    training_args = SFTConfig(
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        output_dir=cfg.ckpt_dir,
        per_device_train_batch_size=cfg.per_gpu_bsz,
        per_device_eval_batch_size=cfg.per_gpu_bsz,
        fp16=False,
        bf16=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=cfg.lr,
        logging_steps=cfg.logging_steps,
        num_train_epochs=cfg.n_epochs,
        warmup_ratio = cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        report_to="wandb",
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=cfg.seed,
        group_by_length=True
    )

    
    sep_tokens = tokenizer.encode(
        '<|start_header_id|>assistant<|end_header_id|>'
        )[1:]
    
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template = sep_tokens, 
        tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field='text',
        max_seq_length=30000,
        tokenizer=tokenizer,
        args=training_args,
        data_collator = data_collator,
    )

    print('Set Trainer')
    print('Start Training!')
    trainer.train()
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='arguments for Fine-tuning BC agent')
    
    # Add arguments
    args = parser.parse_args()

    run = wandb.init(
        project="LCoW",
        name=f"llama3.1-8B_BC_with_seed_demo",
    )

    # config
    cfg = wandb.config
    cfg.seed = 0
    cfg.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    cfg.ckpt_dir = 'ckpt/behavior_cloning'
    cfg.per_gpu_bsz = 1
    cfg.gradient_accumulation_steps = 16
    cfg.lr = 1e-5
    cfg.logging_steps = 1
    cfg.n_epochs = 2
    cfg.weight_decay=0.1
    cfg.warmup_ratio=0.1
    cfg.eval_steps=50
    cfg.save_steps=100
    cfg.iter=args.iter
    cfg.backbone=args.backbone
    cfg.start_ckpt_step = args.start_ckpt_step

    main(cfg)




    
