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


def main():
    parser = argparse.ArgumentParser(description="Robust residual learning")
    
    parser.add_argument("--configs", type=str, default="./configs/configs_cifar10.yml")
    parser.add_argument(
        "--results-dir", type=str, default="eval_logs/",
    )
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument("--arch", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--activation", type=str)
    parser.add_argument("--pssilu-beta", type=float, default=0.3, help="value to set as beta hyperparameter for pssilu/ssilu")
    parser.add_argument("--train", action="store_true", help="perform evaluations on train set if set to true")


    parser.add_argument("--swa", action="store_true", help="Load swa checkpoint, if set to true.")
    
    # eval
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--val-method", type=str, default="baseline", choices=("baseline", "adv", "auto", 'auto2'))
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--step-size", type=float)
    
    # misc
    parser.add_argument("--ckpt", type=str, help="checkpoint path for pretrained classifier")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    
    args = update_args(parser.parse_args())
    device = "cuda:0"
    torch.cuda.set_device(0)
    ngpus = torch.cuda.device_count() # Control available gpus by CUDA_VISIBLE_DEVICES only 
    print(f"Using {ngpus} gpus")
    
    assert args.normalize == False, "Presumption for most code is that the pixel range is [0,1]"
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    assert args.ckpt, "Must provide a checkpint for evaluation"
    
    # seed cuda
    torch.manual_seed((args.local_rank+1)*args.seed)
    torch.cuda.manual_seed((args.local_rank+1)*args.seed)
    torch.cuda.manual_seed_all((args.local_rank+1)*args.seed)
    np.random.seed((args.local_rank+1)*args.seed)
    
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    results_file = os.path.join(args.results_dir, args.exp_name + ".txt")
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(results_file, "a"))
    logger.info(args)
    
    # create model + load checkpoint
    act = models.__dict__[args.activation](beta=args.pssilu_beta)
    model = models.__dict__[args.arch](num_classes=args.num_classes, activation=act).to(device).eval()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if args.swa:
        try:
            d = fix_legacy_dict(checkpoint["swadict"]) # remove module in front of keys
            print(f"Loaded stochastic weighted average (swadict) from checkpoint at {args.ckpt}")
        except:
            d = fix_legacy_dict(checkpoint["state_dict"])
            print(f"No swadict is available in the checkpoint, loading regular checkpoint from {args.ckpt}")
    else:
        d = fix_legacy_dict(checkpoint["state_dict"])
        print(f"Loaded state dict from {args.ckpt}")
    model.load_state_dict(d, strict=True)
    print(f"Mismatched keys {set(d.keys()) ^ set(model.state_dict().keys())}")
    print(f"model loaded from {args.ckpt}")

    # parallelization
    if ngpus > 1:
        print(f"Using multiple gpus")
        model = nn.DataParallel(model).to(device)
    
    # dataloaders
    train_loader, train_sampler, val_loader, val_sampler, _, _, train_transform = data.__dict__[args.dataset](args.data_dir, batch_size=args.batch_size, mode=args.mode, normalize=args.normalize, size=args.size, workers=args.workers, distributed=False)
    criterion = nn.CrossEntropyLoss()

    if args.train:
        results = getattr(utils, args.val_method)(model, device, train_loader, criterion, args, epoch=0)
    else:
        results = getattr(utils, args.val_method)(model, device, val_loader, criterion, args, epoch=0)
    if args.val_method == 'auto2':
        for key in results:
            logger.info(key + ':\n')
            logger.info(", ".join(["{}: {:.3f}".format(k+"_val", v) for (k,v) in results[key].items()]))
    else:
        logger.info(", ".join(["{}: {:.3f}".format(k+"_val", v) for (k,v) in results.items()]))


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
