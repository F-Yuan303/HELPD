#!/bin/bash

LOAD="" # mplug-owl2 model path
DATA_FILE="" # training data path
IMAGE_FOLDER="" # image folder path

deepspeed mplug_owl2/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --visual_abstractor_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder $IMAGE_FOLDER \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard