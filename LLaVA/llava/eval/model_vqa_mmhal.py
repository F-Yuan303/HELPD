import argparse
import json

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
import re
import torch
import os

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def load_image(image_file):
    path = '/data/fyuan/hallucination/LLaVA/playground/data/MMhal/test_data/images/'
    real_name = image_file.split('/')[-1]
    if os.path.exists(path + real_name):
        image = Image.open(path + real_name).convert('RGB')
        return image
    else:
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=str, default='/data/fyuan/hallucination/LLaVA/playground/data/MMhal/test_data/response_template.json', help='template file containing images and questions')
    parser.add_argument('--output', type=str, default='/data/fyuan/hallucination/LLaVA/playground/data/MMhal/response_llava_0.95_beam.json', help='output file containing model responses')
    parser.add_argument('--mymodel', type=str)
    parser.add_argument("--model-path", type=str, default="/data/fyuan/hallucination/LLaVA/experiment/checkpoints/llava-v1.5-13b-lora-precision-2GPU-epoch0.95-gate0.6-default0.5-2reward")
    parser.add_argument("--model-base", type=str, default="/data/fyuan/hallucination/LLaVA/llava-v1.5-7b")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # build the model using your own code
    disable_torch_init()
    force_cudnn_initialization()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    json_data = json.load(open(args.input_json, 'r'))

    for idx, line in enumerate(tqdm(json_data)):
        image_src = line['image_src']
        image = load_image(image_src)
        qs = line['question']

        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                # output_attentions=True,
                # stopping_criteria=[stopping_criteria],
                # opera_decoding=True,
                # key_position=key_position,
                # scale_factor=args.scale_factor,
                # threshold=args.threshold,
                # num_attn_candidates=args.num_attn_candidates,
                # penalty_weights=args.penalty_weights,
            )
        # print(idx, response)
        output_ids = output_ids[0]
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        outputs = outputs.replace('\n','')
        line['model_answer'] = outputs

    with open(args.output, 'w') as f:
        json.dump(json_data, f, indent=2)