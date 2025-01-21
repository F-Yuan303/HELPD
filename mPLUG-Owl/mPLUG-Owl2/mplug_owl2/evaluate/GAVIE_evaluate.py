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

import json
import os

import torch
from PIL import Image
import argparse
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.utils import disable_torch_init
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

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


vg_annotation_file = 'GAVIE/vg_annotation.json'
with open(vg_annotation_file, "rb") as f:
    names = json.load(f)

evaluation_set_file = 'GAVIE/evaluation_set.json'
with open(evaluation_set_file, "rb") as f:
    datas = json.load(f)

def eval(args):
    disable_torch_init()
    force_cudnn_initialization()

    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id



    ids = []
    num = 0
    for data in tqdm(datas):
        image_id = data['image_id']
        boxs = names[image_id]
        path_o = 'GAVIE/' + str(args.output) + '/' + str(image_id) + '.txt'
        file_o = open(path_o, "w")
        L = ["Given an image with following information: bounding box, positions that are the object left-top corner coordinates(X, Y), object sizes(Width, Height). Highly overlapping bounding boxes may refer to the same object.\n\n"]
        L.append('bounding box:\n')
        for box in boxs:
            L.append(box['caption'] + ' X:' + str(box['bbox'][0]) + ' Y:' + str(box['bbox'][1]) + ' Width:' + str(
                box['bbox'][2]) + ' Height:' + str(box['bbox'][2]) + '\n')
        L.append('\n\n')
        L.append('Here is the instruction for the image:\n')
        L.append(data['question'] +'\n\n')
        prompts = data['question']

        conv = conv_templates["mplug_owl2"].copy()


        image_path = "vg/VG_100K/" + image_id + ".jpg"                           
        images = [Image.open(image_path)]
        image_tensor = process_images(images, image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + prompts
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        # print(input_ids.shape)
        # stop_str = conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,)
                # stopping_criteria=[stopping_criteria])
        res = output_ids[0]
        outputs = tokenizer.decode(res[0, input_ids.shape[1]:], skip_special_tokens=True).replace('</s>','').replace('\n','').strip()

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