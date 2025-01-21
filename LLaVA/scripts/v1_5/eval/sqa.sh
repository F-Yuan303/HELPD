#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path /data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.85-gate0.6-default0.5-2reward \
    --model-base /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
    --question-file /data/fyuan/ScienceQA-main/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /data/fyuan/ScienceQA-main/data/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-helpd.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-helpd.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-helpd_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-helpd_result.json
