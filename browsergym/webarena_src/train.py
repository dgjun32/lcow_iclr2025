import os
import sys
import time
import json
import warnings

import trl
import torch
import wandb
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from tqdm import tqdm
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

sys.path.append('./')
from webarena_src.utils import load_hfmodel

#####################################################################
def main(cfg):
    # load dataset for training and validation
    train_df = pd.read_csv(f'./webarena_data/sft_train_iter_{cfg.iter}.csv')
    train_dataset = Dataset(pa.Table.from_pandas(train_df))  
    
    # load model and tokenizers
    base_model, tokenizer = load_hfmodel(cfg.model_name) 
    #base_model = base_model.to("cuda")
    tokenizer.padding_side = 'right'
    #base_model = base_model.to("cuda:0")

    training_args = SFTConfig(
        use_liger=True,
        do_train = True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        dataset_text_field='text',
        max_seq_length=30000,
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
        save_strategy="no",
        evaluation_strategy="no",
        seed=cfg.seed,
        group_by_length=True,
        packing=False,
        dataset_batch_size=128
    )
 

    sep_tokens = tokenizer.encode('<|start_header_id|>assistant<|end_header_id|>')[1:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template = sep_tokens, 
        tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        #eval_dataset=val_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator = data_collator,
    )

    print('Set Trainer')
    print('Start Training!')
    trainer.train()
    trainer.save_model(cfg.ckpt_dir)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Training contextualization module')
    
    # Add arguments
    parser.add_argument('--backbone', 
                        type=str, 
                        help='type of the action model backbone', 
                        default='googleai/gemini-1.5-flash-002'
                        )
    
    parser.add_argument('--iter', 
                        type=int, 
                        help='iteration for data collection', 
                        default=0
                        )
    
    args = parser.parse_args()

    run = wandb.init(
        project="LCoW",
        name=f"Webarena-llama-3.1-8B-training",
    )

    # config
    cfg = wandb.config
    cfg.seed = 0
    cfg.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    cfg.ckpt_dir = f'ckpt/webarena_sft_iter_{args.iter}'
    cfg.per_gpu_bsz = 1
    cfg.gradient_accumulation_steps = 32
    cfg.lr = 1e-5
    cfg.logging_steps = 1
    cfg.n_epochs = 3
    cfg.weight_decay=0.1
    cfg.warmup_ratio=0.1
    cfg.eval_steps=50
    cfg.save_steps=100
    cfg.iter=args.iter
    cfg.backbone=args.backbone
    cfg.iter = args.iter

    main(cfg)




    
