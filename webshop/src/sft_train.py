import os
import sys
import time
import json
import warnings
from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers import TrainingArguments, TrainerCallback
import peft 
import trl
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import pandas as pd
import pyarrow as pa
from datasets import Dataset

sys.path.append('./')
from utils import load_hfmodel


def main(cfg):
    # load dataset for training and validation
    train_df = pd.read_csv(f'collected_data/sft_iter_{cfg.iter}.csv')
    train_dataset = Dataset(pa.Table.from_pandas(train_df))  
    
    # define validation callback
    #validation_callback = ValidationCallback(eval_steps=cfg.save_steps)

    # load model and tokenizer
    if args.iter == 0:
        base_model, tokenizer = load_hfmodel(
            cfg.model_name
            )
    else:
        base_model, tokenizer = load_hfmodel(
            f'./ckpt/sft_iter_{args.iter-1}/checkpoint-{args.start_ckpt_step}'
            )
    tokenizer.padding_side = 'right'

    training_args = TrainingArguments(
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        output_dir=cfg.ckpt_dir,
        per_device_train_batch_size=cfg.per_gpu_bsz,
        per_device_eval_batch_size=cfg.per_gpu_bsz,
        fp16=True,
        bf16=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=cfg.lr,
        logging_steps=cfg.logging_steps,
        num_train_epochs=cfg.n_epochs,
        warmup_ratio = cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        report_to="wandb",
        save_strategy="epoch",
        seed=cfg.seed,
        group_by_length=True
    )

    #sep_tokens = tokenizer.encode('<|assistant|>')[:-1]
    sep_tokens = tokenizer.encode(
        '<|start_header_id|>assistant<|end_header_id|>'
        )[1:]
    
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template = sep_tokens, 
        tokenizer=tokenizer
        )
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        dataset_text_field='text',
        max_seq_length=16384,
        tokenizer=tokenizer,
        args=training_args,
        data_collator = data_collator,
    )

    print('Set Trainer')
    print('Start Training!')
    trainer.train()
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='arguments for training contextualization module for WebShop')
    
    # Add arguments
    parser.add_argument('--start_ckpt_step', type=int, default = None)
    parser.add_argument('--iter', type=int, help='iteration for data collection', default=1)
    args = parser.parse_args()

    run = wandb.init(
        project="LCoW-Webshop",
        name=f"SFT_Phi-3_expert_iteration_iter_{args.iter}",
    )

    # config
    cfg = wandb.config
    cfg.seed = 0
    cfg.model_name = "microsoft/Phi-3-mini-128k-instruct"
    cfg.ckpt_dir = f'ckpt/sft_iter_{args.iter}/'
    cfg.per_gpu_bsz = 1
    cfg.gradient_accumulation_steps = 4
    cfg.lr = 1e-5
    cfg.logging_steps = 1
    cfg.n_epochs = 1
    cfg.weight_decay=1.0
    cfg.warmup_ratio=0.01
    cfg.eval_steps=50
    cfg.save_steps=100
    cfg.iter=args.iter
    cfg.start_ckpt_step = args.start_ckpt_step

    main(cfg)