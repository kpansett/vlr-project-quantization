import logging
import numpy as np
import os
import random
import sys
import time
import torch

import onnx
import onnxruntime

import transformers


device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import sys
import yaml
sys.path.append("X-VLM/")
from models.xvlm import *
from models.model_retrieval import *

config_path = "/home/ubuntu/project/X-VLM/configs/Retrieval_coco.yaml"
from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer



with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)

config["text_config"] = os.path.join("X-VLM", config["text_config"])
config["vision_config"] = os.path.join("X-VLM", config["vision_config"])
config["text_encoder"] = os.path.join("X-VLM", config["text_encoder"])
config["train_file"][0] = os.path.join("X-VLM", config["train_file"][0])
config["val_file"] = os.path.join("X-VLM", config["val_file"])
config["test_file"] = os.path.join("X-VLM", config["test_file"])
config["image_root"] = os.path.join("X-VLM", config["image_root"])

print(config)
if config['use_roberta']:
    tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
else:
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])


model = XVLM(config=config)


ckpt_rpath = "/home/ubuntu/project/X-VLM/data/checkpoint_best.pth"
model.load_pretrained(ckpt_rpath, config)

from dataset import *

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# dist.init_process_group(backend="gloo", rank=1, world_size=1)
setup(rank=0, world_size=1)

config["prompt"] = ""

train_dataset, val_dataset, test_dataset = create_dataset('re', config)

samplers = [None, None, None]
train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                      batch_size=[4,4,4],
                                                      num_workers=[4, 4, 4],
                                                      is_trains=[True, False, False],
                                                      collate_fns=[None, None, None])

from Retrieval import *

model_without_ddp = quantized_model
device = torch.device("cpu")

model_without_ddp.to(device)

start_time = time.time()
print("Start evaluating", flush=True) 

score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
# score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
print(val_result)
# test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
# print(test_result)

log_stats = {**{f'val_{k}': v for k, v in val_result.items()}}

print(log_stats)




