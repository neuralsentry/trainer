#!/bin/bash

export WANDB_PROJECT="hf-distilbert-imdb-mlm"


python ./training/hf_trainer.py \
--model_path 'distilbert-base-uncased' \
--tokenizer_path 'distilbert-base-uncased' \
--dataset_path './data/imdb-tokenized-mlm' \
--output_dir "./models/$WANDB_PROJECT" \
--overwrite_output_dir true \
--hub_model_id "$WANDB_PROJECT" \
--push_to_hub true \
--task_name 'mlm' \
--report_to 'wandb' \
--do_train \
--do_eval \
--num_train_epochs 20 \
--max_seq_length 256 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--seed=420 \
--data_seed=420 \
--optim 'adamw_torch_fused' \
--learning_rate 2e-5 \
--lr_scheduler_type 'linear' \
--weight_decay 1e-2 \
--mlm_probability 0.15 \
--fp16 true \
--train_split 0.5 \
--evaluation_strategy 'epoch' \
--logging_strategy 'epoch' \
--save_strategy 'epoch' \
--hub_strategy 'every_save' \
--save_total_limit 2
