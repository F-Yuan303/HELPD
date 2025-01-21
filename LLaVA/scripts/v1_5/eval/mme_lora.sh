#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.85-gate0.6-default0.5-2reward \
#     --model-base /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#     --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b-helpd.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-helpd

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-helpd
