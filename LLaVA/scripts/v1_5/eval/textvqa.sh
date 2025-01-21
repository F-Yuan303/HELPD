#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl




ds="textvqa_val"
checkpoint=/data/fyuan/hallucination/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b
 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-4} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    mplug_owl2/evaluate/evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 4 \
    --num-workers 2

ds="vqav2_testdev"
checkpoint=/data/fyuan/hallucination/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-4} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    mplug_owl2/evaluate/evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 4 \
    --num-workers 2
