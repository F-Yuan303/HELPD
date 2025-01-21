python -m llava.eval.model_vqa_loader \
    --model-path /data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.85-gate0.6-default0.5-2reward \
    --model-base /data/fyuan/hallucination/LLaVA/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-helpd.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-helpd.jsonl