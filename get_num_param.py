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
import data
import trainers
import utils
from utils import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
     
    # create model + load checkpoint
    act = models.__dict__['relu'](beta=0)
    for arch in ['resnet18', 'wrn_28_10', 'vgg16_bn', 'ResNet18']:
        model = models.__dict__[arch](num_classes=10, activation=act).to('cpu').eval()
        num_param = count_parameters(model)
        print(arch, num_param)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
