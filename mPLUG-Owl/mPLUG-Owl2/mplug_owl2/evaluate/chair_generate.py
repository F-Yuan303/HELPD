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
from transformers import TextStreamer

# from pope_loader import POPEDataSet
# from minigpt4.common.dist_utils import get_rank
# from minigpt4.models import load_preprocess

# from minigpt4.common.config import Config
# from minigpt4.common.dist_utils import get_rank
# from minigpt4.common.registry import registry

# # imports modules for registration
# from minigpt4.datasets.builders import *
# from minigpt4.models import *
# from minigpt4.processors import *
# from minigpt4.runners import *
# from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn
import json

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.utils import disable_torch_init
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


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






def eval_model(args):
# # ========================================
# #             Model Initialization
# # ========================================
    # disable_torch_init()
    # force_cudnn_initialization()

    query = "USER: <|image|>Describe the content of the image. ASSISTANT:"
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # conv = conv_templates["mplug_owl2"].copy()

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)


    img_files = os.listdir(args.data_path + "/val2014")
    random.shuffle(img_files)

    with open(args.data_path + '/annotations/instances_val2014.json', 'r') as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])

    img_dict = {}

    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}

    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )


    # epoch = model_name.split('epoch')[1].split('-')[0]
    # gate = model_name.split('gate')[1].split('-')[0]
    # default = "0.5"
    # reward = model_name.split('reward')[0].split('-')[-1]

    base_dir  = "" # model directory
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)


    key_position = {
            "image_start": 5, 
            "image_end": 70, 
            "response_start": 83
        }
        
    pad_token_id = tokenizer.pad_token_id
    for img_id in tqdm(range(len(img_files))):
        if img_id == 500:
            break
        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id

        image_path = args.data_path + "/val2014/" + img_file   

        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # inp = DEFAULT_IMAGE_TOKEN + query
        # conv.append_message(conv.roles[0], inp)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        

        input_ids = tokenizer_image_token(query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        attention_mask = 1 - input_ids.eq(pad_token_id).long()
        attention_mask = attention_mask.to(model.device)



        # stop_str = conv.sep2
        # keywords = [stop_str]

        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)                        

        with torch.inference_mode():
            # output_ids, roll_back = model.generate(
            #     input_ids,
            #     images=images_tensor,
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     max_new_tokens=args.max_new_tokens,
            #     use_cache=True,
            #     output_attentions=True,
            #     # repetition_penalty = 1.2,
            #     stopping_criteria=[stopping_criteria],
            #     opera_decoding=True,
            #     key_position=key_position,
            #     scale_factor=args.scale_factor,
            #     threshold=args.threshold,
            #     num_attn_candidates=args.num_attn_candidates,
            #     penalty_weights=args.penalty_weights,
            # )

            # print(input_ids.shape)
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=1,
                use_cache=True,
                opera_decoding=True,
                key_position=key_position,
                output_attentions=True,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )
                # stopping_criteria=[stopping_criteria])

        output_ids = output_ids[0]

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace('</s>','').replace('\n','').strip()

        img_save["caption"] = outputs

        # dump metric file
        with open(os.path.join(base_dir, 'ours-reward_2-E_0.85-opera.jsonl'), "a") as f: #e_{}-g_{}-d_{}-reward_{}-opera_3-max_128_sum100.jsonl'.format(epoch, gate, default, reward)), "a") as f:
            json.dump(img_save, f)
            f.write('\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--data_path", type=str, default="COCO_2014/val2014/", help="data path")

    # parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--opera_decoding", type=bool, default=True)
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)

    parser.add_argument("--model-path", type=str, default="/data/fyuan/hallucination/LLaVA/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default="/data/fyuan/hallucination/mPLUG-Owl/mPLUG-Owl2/mplug-owl2-llama2-7b")

    parser.add_argument("--query", type=str, default='Describe the content of the image.')
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_known_args()[0]

    setup_seed(42)
    eval_model(args)

