import argparse
import json

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

import shortuuid

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.utils import disable_torch_init
from mplug_owl2.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

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
    parser.add_argument('--output', type=str, default='/data/fyuan/hallucination/mPLUG-Owl/mPLUG-Owl2/experiments/MMhal/response_mplug_0.95_beam.json', help='output file containing model responses')
    parser.add_argument('--mymodel', type=str)
    parser.add_argument("--model-path", type=str, default="/data/fyuan/hallucination/mPLUG-Owl/mPLUG-Owl2/checkpoints/mplug-owl2-finetune-lora-E_0.95-reward_2")
    parser.add_argument("--model-base", type=str, default="/data/fyuan/hallucination/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # build the model using your own code
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    json_data = json.load(open(args.input_json, 'r'))

    for idx, line in enumerate(tqdm(json_data)):
        image_src = line['image_src']
        image = load_image(image_src)
        qs = line['question']

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        attention_mask = 1 - input_ids.eq(pad_token_id).long()
        attention_mask = attention_mask.to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
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
        # output_ids = output_ids[0]
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace('</s>','').replace('\n','').strip()
        line['model_answer'] = outputs

    with open(args.output, 'w') as f:
        json.dump(json_data, f, indent=2)