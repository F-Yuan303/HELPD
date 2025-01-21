# CUDA_VISIBLE_DEVICES=6 python /data/fyuan/hallucination/LLaVA/GAVIE_evaluate.py \
#     --model-path /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
#     --output origin/
# wait
CUDA_VISIBLE_DEVICES=6 python /data/fyuan/hallucination/LLaVA/GAVIE_evaluate.py \
    --model-path /data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.75-gate0.6-default0.5-2reward \
    --model-base /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
    --output 0.75/
# wait
# CUDA_VISIBLE_DEVICES=6 python /data/fyuan/hallucination/LLaVA/GAVIE_evaluate.py \
#     --model-path /data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.85-gate0.6-default0.5-2reward \
#     --model-base /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
#     --output 0.85/
# wait
# CUDA_VISIBLE_DEVICES=6 python /data/fyuan/hallucination/LLaVA/GAVIE_evaluate.py \
#     --model-path /data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.95-gate0.6-default0.5-2reward \
#     --model-base /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
#     --output 0.95/