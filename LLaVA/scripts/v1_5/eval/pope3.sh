#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python -m llava.eval.model_vqa_loader \
    --model-path /data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.75-gate0.6-default0.5-2reward \
    --model-base /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
    --question-file /data/fyuan/hallucination/LLaVA/playground/data/pope/coco_pope_random.jsonl \
    --image-folder /data/fyuan/VCR_generation/data/VQA-X/val2014 \
    --answers-file /data/fyuan/hallucination/LLaVA/playground/data/pope/llava-v1.5-7b-lora-random-precision-2GPU-epoch0.75-gate0.6-default0.5-2reward.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=5 python llava/eval/eval_pope.py \
    --annotation-dir /data/fyuan/hallucination/LLaVA/playground/data/pope \
    --question-file /data/fyuan/hallucination/LLaVA/playground/data/pope/coco_pope_random.jsonl \
    --result-file /data/fyuan/hallucination/LLaVA/playground/data/pope/llava-v1.5-13b-random.jsonl