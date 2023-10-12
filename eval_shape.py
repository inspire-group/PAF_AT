from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import json
from collections import OrderedDict
import importlib

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex (otherwise use DP accelerator.")
import models 
from models.wide_resnet import wrn_28_10
from models.vgg import vgg16_bn
from models.resnet_cifar import resnet18
from models.ResNet import ResNet18
import data
import trainers
import utils
from utils import *

def clean_acc(model, test_loader):
    num_correct = 0
    for img, target in test_loader:
        img, target = img.cuda(), target.cuda()
        logits_clean = model(img)
        corr_classified = (torch.max(logits_clean, 1)[1] == target)
        num_correct += corr_classified.sum()
    return num_correct.item() / len(test_loader.dataset)

def main():
    model_dir = '/scratch/gpfs/sihuid/results/std_param'
    f = open('params_std.out', 'w')
    for m in os.listdir(model_dir):
        f.write("Evaluating {}\n".format(m))
        name = m.split('_')
        print(name)
        act = name[2]
        if act not in ['prelu', 'pelu', 'psilu', 'psoftplus', 'pblu', 'pprelu', 'pssilu2']:
            continue
        activation = models.__dict__[act]() 
        if name[1] == 'resnet':
            model = resnet18(num_classes=10, activation=activation).cuda().eval()
        elif name[1] == 'wrn':
            model = wrn_28_10(num_classes=10, activation=activation).cuda().eval()
        #elif name[1] == 'cifar100':
        #    model = wrn_28_10(num_classes=100, activation=activation).cuda().eval()
        elif name[1] == 'vgg':
            model = vgg16_bn(num_classes=10, activation=activation).cuda().eval()
        #elif name[1] == 'imagenette': 
        #    model = ResNet18(num_classes=10, activation=activation).cuda().eval()
        else:
            continue
        checkpoint = torch.load(model_dir + '/{}/trial_0/checkpoint/model_best.pth.tar'.format(m), map_location="cpu")
        d = fix_legacy_dict(checkpoint["state_dict"])
        model.load_state_dict(d, strict=True)
        train_loader, train_sampler, val_loader, val_sampler, _, _, train_transform = data.__dict__['cifar10']('/scratch/gpfs/sihuid/data', batch_size=256)

        # evaluation (return a dictionary from this functions and print its key-val pairs in file)
        acc = clean_acc(model, val_loader)

        f.write('acc: {} \n'.format(acc))

        if name[1] == 'vgg':
            f.write('alpha: {} \n'.format(model.features[2].alpha.item()))
            if act == 'pssilu2':
                f.write('beta: {} \n'.format(model.features[2].beta.item()))
        else:
            f.write('alpha: {} \n'.format(model.activation.alpha.item()))
            if act == 'pssilu2':
                f.write('beta: {} \n'.format(model.activation.beta.item()))

    f.close()
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
