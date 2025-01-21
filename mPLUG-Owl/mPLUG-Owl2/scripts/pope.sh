#!/bin/bash
MODEL_PATH="" # model path
MODEL_BASE="" # model base path
GENERATE_PATH="" # generate path
OUTPUT_PATH="" # output path

CUDA_VISIBLE_DEVICES=0 python -m mplug_owl2.evaluate.model_vqa_loader \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --question-file pope/coco_pope_popular.jsonl \
    --image-folder VQA-X/val2014 \
    --answers-file $GENERATE_PATH \
    --temperature 0 \
    # --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python mPLUG-Owl/mPLUG-Owl2/mplug_owl2/evaluate/eval_pope.py \
    --annotation-dir pope \
    --question-file pope/coco_pope_popular.jsonl \
    --result-file $OUTPUT_PATH
