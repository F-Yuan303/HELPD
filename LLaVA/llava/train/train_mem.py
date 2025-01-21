# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train import train

from transformers import StoppingCriteria
import torch
import tqdm
import openai
import socket
import nltk
from openai import OpenAI
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model

import os

if __name__ == "__main__":
    train()
