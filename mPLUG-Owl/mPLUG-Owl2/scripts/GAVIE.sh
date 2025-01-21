MODEL_PATH="" # model path
OUTPUT_PATH="" # output path


CUDA_VISIBLE_DEVICES=0 python /data/fyuan/hallucination/mPLUG-Owl/mPLUG-Owl2/GAVIE_evaluate.py \
    --model-path $MODEL_PATH \
    --output $OUTPUT_PATH 
