#!/bin/bash

model_name="distilbert-git-commit-bugfix-classification"
export WANDB_PROJECT="git-commit-bugfix-classification"
report_to="wandb" # wandb | none

interval_NO="no"
interval_STEPS="steps"
interval_EPOCH="epoch"

hub_strategy_END="end"
hub_strategy_EVERY_SAVE="every_save"
hub_strategy_CHECKPOINT="checkpoint"
hub_strategy_ALL_CHECKPOINTS="all_checkpoints"

config_training_output_dir="models/$model_name"
config_training_overwrite_output_dir=true
config_training_do_train=true
config_training_do_eval=true
config_training_per_device_train_batch_size=128
config_training_per_device_eval_batch_size=128
config_training_fp16=true
config_training_learning_rate=1e-4
config_training_seed=420
config_training_data_seed=420
config_training_num_train_epochs=2
config_training_optim="adamw_hf"
config_training_weight_decay=1e-2
config_training_lr_scheduler_type="linear"
config_training_evaluation_strategy=$interval_STEPS
config_training_logging_strategy=$interval_STEPS
config_training_save_strategy=$interval_EPOCH
config_training_eval_steps=3
config_training_logging_steps=3
config_training_save_steps=3
config_training_save_total_limit=2
config_training_report_to=$report_to
config_training_hub_strategy=$hub_strategy_END
config_training_push_to_hub=true
config_training_hub_model_id="neuralsentry/$model_name"
config_model_model_name_or_path="neuralsentry/distilbert-git-commits-mlm"
config_model_model_revision="main"
config_model_cache_dir=".cache"
config_data_dataset_name="neuralsentry/git-commit-bugfixes"
config_data_max_seq_length=256
config_data_validation_split_percentage=10
config_data_preprocessing_num_workers=4
config_data_text_column_name="commit_msg"
# config_data_max_train_samples=1000
# config_data_max_eval_samples=200

python training/run_git_commits_bugfix_classification.py \
--output_dir $config_training_output_dir \
--overwrite_output_dir $config_training_overwrite_output_dir \
--do_train $config_training_do_train \
--do_eval $config_training_do_eval \
--per_device_train_batch_size $config_training_per_device_train_batch_size \
--per_device_eval_batch_size $config_training_per_device_eval_batch_size \
--fp16 $config_training_fp16 \
--learning_rate $config_training_learning_rate \
--seed $config_training_seed \
--data_seed $config_training_data_seed \
--num_train_epochs $config_training_num_train_epochs \
--optim $config_training_optim \
--weight_decay $config_training_weight_decay \
--lr_scheduler_type $config_training_lr_scheduler_type \
--evaluation_strategy $config_training_evaluation_strategy \
--logging_strategy $config_training_logging_strategy \
--save_strategy $config_training_save_strategy \
--eval_steps $config_training_eval_steps \
--logging_steps $config_training_logging_steps \
--save_steps $config_training_save_steps \
--save_total_limit $config_training_save_total_limit \
--report_to $config_training_report_to \
--hub_strategy $config_training_hub_strategy \
--push_to_hub $config_training_push_to_hub \
--hub_model_id $config_training_hub_model_id \
--model_name_or_path $config_model_model_name_or_path \
--model_revision $config_model_model_revision \
--cache_dir $config_model_cache_dir \
--dataset_name $config_data_dataset_name \
--max_seq_length $config_data_max_seq_length \
--validation_split_percentage $config_data_validation_split_percentage \
--preprocessing_num_workers $config_data_preprocessing_num_workers \
--text_column_name $config_data_text_column_name # \
# --max_train_samples $config_data_max_train_samples \
# --max_eval_samples $config_data_max_eval_samples \

