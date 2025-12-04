import torch
import time
import json
import pathlib
import argparse
import re
import os

import ctypes
import numpy as np
from transformers import AutoConfig, TextStreamer, AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.append(sys.path[0] + "/../../../")

import logging
import gc
import copy

model_id = "meta-llama/Llama-3.1-405B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    attn_implementation="eager",
)

num_orig_declayers = model.config.num_hidden_layers
num_new_declayers = 21

new_config = AutoConfig.from_pretrained(model_id)
new_config.num_hidden_layers = num_new_declayers

small_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=new_config,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
### Copy the weights from the original model to the smaller model
print("Copying the weights from the original model to the smaller model")
small_model.model.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())

for i in range(num_new_declayers):
    small_model.model.layers[i].load_state_dict(model.model.layers[i].state_dict())
    print(f"Copied weights for layer {i}")

small_model.model.norm.load_state_dict(model.model.norm.state_dict())
small_model.lm_head.load_state_dict(model.lm_head.state_dict())

print("\nWeights copied to the new smaller model successfully!")
print(small_model)

output_dir = "/shared_storage/Llama-3.1-405B-small/"
os.makedirs(output_dir, exist_ok=True)
small_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)