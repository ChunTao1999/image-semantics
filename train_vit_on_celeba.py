# Author: Chun Tao
# Date: 2023-10-02
import argparse
from celebAHQ import dataloader_celebAHQ
import json
import os
import torch
from utils import *
# Huggingface
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import pdb


#%% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--celebahq_img_path', type=str, default="", help="path to celebahq images")
parser.add_argument('--celebahq_label_path', type=str, default="", help='path to celebahq seg labels')
parser.add_argument('--label_mapping_path', type=str, default="", help='path to label to class mapping json file')
parser.add_argument('--batch_size_train', type=int, default=128, help="train batch size")
args = parser.parse_args()


#%% Logger setup
logger = set_logger(os.path.join('logs', 'train_vit_on_celeba.log'))
logger.info(args)


#%% Label mapping
# id2label = {1:"skin", 2:"hair", 3:"l_brow", 4:"nose", 5:"mouth", 6:"neck"}
# json.dump(id2label, open("./celebAHQ/celebahq-id2label.json", 'w'))
id2label = json.load(open(args.label_mapping_path, 'r'))
label2id = {v:k for k,v in id2label.items()}


#%% Dataset and Dataloader
trainloader = dataloader_celebAHQ.Data_Loader(img_path=args.celebahq_img_path, 
                                              label_path=args.celebahq_label_path,
                                              image_size=256, 
                                              batch_size=args.batch_size_train, 
                                              mode=True).loader()


#%% Preprocess, Models
checkpoint = "nvidia/mit-b0"
image_processor = AutoImageProcessor.from_pretrained(checkpoint,
                                                     do_reduce_labels=True)
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, 
                                                         id2label=id2label, 
                                                         label2id=label2id)
model.cuda()
model.train()


#%% Train loop

pdb.set_trace()