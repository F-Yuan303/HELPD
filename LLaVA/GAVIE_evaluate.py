import argparse
import os
import random
import requests
from PIL import Image
from io import BytesIO
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn
import json

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


vg_annotation_file = '/data/fyuan/hallucination/GAVIE/vg_annotation.json'
with open(vg_annotation_file, "rb") as f:
    names = json.load(f)

evaluation_set_file = '/data/fyuan/hallucination/GAVIE/evaluation_set.json'
with open(evaluation_set_file, "rb") as f:
    datas = json.load(f)

def eval(args):
    disable_torch_init()
    force_cudnn_initialization()

    model_name = get_model_name_from_path(args.model_path)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )



    ids = []
    num = 0
    for data in tqdm(datas):
        image_id = data['image_id']
        boxs = names[image_id]
        path_o = '/data/fyuan/hallucination/LLaVA/experiment/GAVIE/' + str(args.output) + '/' + str(image_id) + '.txt'
        file_o = open(path_o, "w")
        L = ["Given an image with following information: bounding box, positions that are the object left-top corner coordinates(X, Y), object sizes(Width, Height). Highly overlapping bounding boxes may refer to the same object.\n\n"]
        L.append('bounding box:\n')
        for box in boxs:
            L.append(box['caption'] + ' X:' + str(box['bbox'][0]) + ' Y:' + str(box['bbox'][1]) + ' Width:' + str(
                box['bbox'][2]) + ' Height:' + str(box['bbox'][2]) + '\n')
        L.append('\n\n')
        L.append('Here is the instruction for the image:\n')
        L.append(data['question'] +'\n\n')
        qs = data['question']

        #=======================
        image_path = "/data/fyuan/VCR_generation/data/e-SNLI-VE/flickr30k_images/3347798761.jpg"
        qs = "Describe the content of  the image in detail."
        #=======================

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

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        image_path = "/data/fyuan/VCR_generation/data/vg/VG_100K/" + image_id + ".jpg"                           
            
        images = load_images([image_path])
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        # print(input_ids)
        # print(input_ids.shape)
        # print(input_ids.index(IMAGE_TOKEN_INDEX))

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
                stopping_criteria=[stopping_criteria],
            )
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
        print(outputs)
        exit(0)

        #The following two are sample answers from different models, one answer from each model. I mean you can evaluate several answers for the same image and instruction in one prompt.
        answer_sample1 = outputs
        # print(answer_sample1)
        # answer_sample2 = 'answer sample2'
    
        L.append('Answer1: ' + answer_sample1 + '\n')
        # L.append('Answer2: ' + answer_sample2 + '\n\n')
        L.append('Suppose you are a smart teacher, after looking at the image information above, please score the above answers(0-10) according to the following criteria:\n')
        L.append('1: whether the response directly follows the instruction.\n')
        L.append('2: whether the response is accurate concerning the image content.\n\n')
        # L.append('Output format:\n\n'
        #         'Relevancy:\nscore of the answer1:\nreason:\nscore of the answer2:\nreason:\n\n')
        # L.append('Accuracy:\nscore of the answer1:\nreason:\nscore of the answer2:\nreason:\n\n')
        L.append('Output format:\n\n'
                'Relevancy:\nscore of the answer1:\n\n')
        L.append('Accuracy:\nscore of the answer1:\n\n')

        file_o.writelines(L)
        file_o.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/fyuan/hallucination/LLaVA/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    eval(args)