import random
import torch
import time
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from torch import nn
import torch.nn.functional as F
import os
from shutil import copyfile
from Camel.utils.utils import evaluate_loss, evaluate_metrics, train_scst, train_xe, train_xe_2, evaluate_loss_fn
from torch.utils.data import Dataset, Sampler, DataLoader
from common.evaluation import PTBTokenizer, Cider
from common.data import DataLoader
from common.data.field import RawField
from torch.nn.utils.rnn import pad_sequence
from Camel.utils.lookahead import Lookahead
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)




def test(args, image_model, target_model, online_model, datasets, image_field, text_field, optim=None, scheduler=None):

    device = args.device
    output = args.output
    use_rl = args.use_rl

    date = time.strftime("%Y-%m-%d", time.localtime())

    train_dataset, val_dataset, test_dataset = datasets

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    train_batch_size = args.batch_size // 5 if use_rl else args.batch_size
    dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataloader_val = DataLoader(val_dataset, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True)

    
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    best_cider = .0
    patience = 0
    start_epoch = 0

    # online_data = torch.load('/home/zkx/ImgCaption/CaDReL/captioning/Camel/saved/126.11.pth')
    online_data = torch.load('/home/zkx/ImgCap/CaDReL/captioning/Camel/saved/base_119.pth')
    # target_model.load_state_dict(target_data['state_dict_t'])
    # online_model.load_state_dict(online_data['state_dict_t'])
    online_model.load_state_dict(online_data['state_dict_o'])
    # target_model.to(args.device)
    online_model.to(args.device)


    print("Testing starts")

    online_cider_list=[]
    target_cider_list=[]



    # Validation scores
    scores = evaluate_metrics(image_model, online_model, dict_dataloader_test, text_field, 1, device, args)
    print("Online test scores", scores)
    val_cider = scores['CIDEr']
    online_cider_list.append(val_cider)

    # Test scores
    # scores = evaluate_metrics(image_model, target_model, dict_dataloader_test, text_field, 1, device, args, reverse=False)
    # print("Target test scores", scores)
    # target_cider_list.append(scores['CIDEr'])

    



