#!/bin/bash

model_name="starencoder-git-commits-mlm"
export WANDB_PROJECT="git-commits-mlm"
report_to="none"
if [ -n "$WANDB_PROJECT" ]; then
    ReportTo="wandb"
fi

interval_NO="no"
interval_STEPS="steps"
interval_EPOCH="epoch"

hub_strategy_END="end"
hub_strategy_EVERY_SAVE="every_save"
hub_strategy_CHECKPOINT="checkpoint"
hub_strategy_ALL_CHECKPOINTS="all_checkpoints"

config_training_output_dir="models/$ModelName"
config_training_overwrite_output_dir=true
config_training_do_train=true
config_training_do_eval=true
config_training_per_device_train_batch_size=32
config_training_per_device_eval_batch_size=32
config_training_fp16=true
config_training_learning_rate=1e-4
config_training_seed=420
config_training_data_seed=420
config_training_num_train_epochs=40
config_training_optim="adamw_hf"
config_training_weight_decay=1e-2
config_training_lr_scheduler_type="linear"
config_training_evaluation_strategy=$Interval_EPOCH
config_training_logging_strategy=$Interval_EPOCH
config_training_save_strategy=$Interval_EPOCH
config_training_eval_steps=500
config_training_logging_steps=500
config_training_save_steps=500
config_training_save_total_limit=2
config_training_report_to=$ReportTo
config_training_hub_strategy=$HubStrategy_EVERY_SAVE
config_training_push_to_hub=true
config_training_hub_model_id="neuralsentry/$ModelName"
config_model_model_name_or_path="bigcode/starencoder"
config_model_model_revision="main"
config_model_cache_dir=".cache"
config_model_mask_token="<mask>"
config_model_pad_token="<pad>"
config_model_sep_token="<sep>"
config_model_cls_token="<cls>"
config_data_dataset_name="neuralsentry/git-commits"
config_data_max_seq_length=256
config_data_validation_split_percentage=10
config_data_preprocessing_num_workers=4
config_data_mlm_probability=0.15
config_data_text_column_name="commit_msg"
# config_data_max_train_samples=1000
# config_data_max_eval_samples=200

python training/run_mlm.py \
  --output_dir $Config_training_output_dir \
  --overwrite_output_dir $Config_training_overwrite_output_dir \
  --do_train $Config_training_do_train \
  --do_eval $Config_training_do_eval \
  --per_device_train_batch_size $Config_training_per_device_train_batch_size \
  --per_device_eval_batch_size $Config_training_per_device_eval_batch_size \
  --fp16 $Config_training_fp16 \
  --learning_rate $Config_training_learning_rate \
  --seed $Config_training_seed \
  --data_seed $Config_training_data_seed \
  --num_train_epochs $Config_training_num_train_epochs \
  --optim $Config_training_optim \
  --weight_decay $Config_training_weight_decay \
  --lr_scheduler_type $Config_training_lr_scheduler_type \
  --evaluation_strategy $Config_training_evaluation_strategy \
  --logging_strategy $Config_training_logging_strategy \
  --save_strategy $Config_training_save_strategy \
  --eval_steps $Config_training_eval_steps \
  --logging_steps $Config_training_logging_steps \
  --save_steps $Config_training_save_steps \
  --save_total_limit $Config_training_save_total_limit \
  --report_to $Config_training_report_to \
  --hub_strategy $Config_training_hub_strategy \
  --push_to_hub $Config_training_push_to_hub \
  --hub_model_id $Config_training_hub_model_id \
  --model_name_or_path $Config_model_model_name_or_path \
  --model_revision $Config_model_model_revision \
  --cache_dir $Config_model_cache_dir \
  --dataset_name $Config_data_dataset_name \
  --max_seq_length $Config_data_max_seq_length \
  --validation_split_percentage $Config_data_validation_split_percentage \
  --preprocessing_num_workers $Config_data_preprocessing_num_workers \
  --mlm_probability $Config_data_mlm_probability \
  --text_column_name $Config_data_text_column_name \
  --mask_token $Config_model_mask_token \
  --pad_token $Config_model_pad_token \
  --sep_token $Config_model_sep_token \
  --cls_token $Config_model_cls_token # \
  # --max_train_samples $Config_data_max_train_samples \
  # --max_eval_samples $Config_data_max_eval_samples \

