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
import copy 

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


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

def main():
    parser = argparse.ArgumentParser(description="Robust residual learning")
    
    parser.add_argument("--configs", type=str, default="./configs/configs_cifar10.yml")
    parser.add_argument(
        "--results-dir", type=str, default="results",
    )
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument("--arch", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--swa", action="store_true", help="Use stochstic weight averaging (https://arxiv.org/pdf/2010.03593.pdf)")
    parser.add_argument("--tau", type=float, help="Tau value in swa")
    parser.add_argument("--clip", type=float, default=0,  help="gradient clipping parameter")
 
    # activation settings
    parser.add_argument("--activation", type=str, help='name of activation')
    parser.add_argument("--fix-act", action="store_true", default=False, help="make activation parameter not trainable")
    parser.add_argument("--fix-act-val", type=float, default=1, help="value to fix activation parameter to")
    parser.add_argument("--pssilu-beta", type=float, default=0.3, help="value to set as beta hyperparameter for pssilu/ssilu")
    parser.add_argument("--pssilu-reg-type", type=str, default='l1', choices=('l1', 'l2'))
    parser.add_argument("--pssilu-reg", type=float, default=10)

    # training
    parser.add_argument("--trainer", type=str, default="baseline", choices=("baseline", "adv", "fgsm", "madry"))
    parser.add_argument("--val-method", type=str, default="baseline", choices=("baseline", "adv"))
    parser.add_argument("--accelerator", type=str, default="dp", choices=("dp", "ddp"))
    parser.add_argument("--fp16", action="store_true", default=False, help="half precision training")
    parser.add_argument("--classes", type=list)
    parser.add_argument("--training-images", type=int)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--wamrup-epochs", type=int, default=5)
    
    # synthetic data
    parser.add_argument("--syn-data-list", nargs="+", default=None, help="list of different synthetic datasets") # eg., --syn-data-list ti500k_cifar10 diffusion_cifar10
    parser.add_argument("--batch-size-syn", type=int, help="batch-size for synthetic data")
    
    # misc
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--ckpt", type=str, help="checkpoint path for pretrained classifier")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    
    args = update_args(parser.parse_args())

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

    elif args.dataset == "cifar100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    elif args.dataset == "imagenette":
        mean =  [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]


    def normalize(X):
        return (X - mu.to(X.get_device()))/std.to(X.get_device())

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    ngpus = torch.cuda.device_count() # Control available gpus by CUDA_VISIBLE_DEVICES only 
    print(f"Using {ngpus} gpus")
    args.distributed = (args.accelerator == "ddp") and ngpus > 1 # Need special care with ddp distributed training
    
    if args.fp16 and ngpus > 1:
        assert args.accelerator == "ddp", "half precision on multiple gpus supported only ddp mode"
    assert args.normalize == False, "Presumption for most code is that the pixel range is [0,1]"
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    assert not ((args.swa is False) ^ (args.tau is None)), "if using swa then must set tau, if not using swa then must not set tau (to be safe)"
    print("Make sure to manually scale learning rate with 0.1*total-batch-size/128 rule")
    
    # seed cuda
    torch.backends.cudnn.benchmark=True # a few percentage speedup
    torch.manual_seed((args.local_rank+1)*args.seed)
    torch.cuda.manual_seed((args.local_rank+1)*args.seed)
    torch.cuda.manual_seed_all((args.local_rank+1)*args.seed)
    np.random.seed((args.local_rank+1)*args.seed)
    
    # create resutls dir (for logs, checkpoints, etc.)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    result_main_dir = os.path.join(args.results_dir, args.exp_name)
    result_sub_dir = os.path.join(result_main_dir, f"trial_{args.trial}")
    create_subdirs(result_sub_dir)
    
    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)
    
    # multi-gpu DDP
    if args.accelerator == "ddp":
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
        world_size = torch.distributed.get_world_size()
        logger.info(f"world_size = {world_size}")
        
        # Scale learning rate based on global batch size
        args.batch_size = args.batch_size // world_size
        args.workers = args.workers // world_size
        args.batch_size_syn = args.batch_size_syn // world_size
        logger.info(f"New per-gpu batch-size = {args.batch_size}, syn-batch-size = {args.batch_size_syn}, workers = {args.batch_size}")
    
    # create model + optimizer
    act = models.__dict__[args.activation](beta=args.pssilu_beta)
    if args.fix_act:
       act.alpha = nn.Parameter(torch.tensor([args.fix_act_val]))
       act.alpha.requires_grad=False
       
    model = models.__dict__[args.arch](num_classes=args.num_classes, activation=act).to(device).train()
    model = nn.Sequential(Normalize(mean=mean, std=std), model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    best_prec = 0
    start_epoch = 0
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        d = fix_legacy_dict(checkpoint)
        model.load_state_dict(d, strict=True)
        logger.info(f"Mismatched keys {set(d.keys()) ^ set(model.state_dict().keys())}")
        logger.info(f"model loaded from {args.ckpt}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_prec = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
    print(model)
    
    # half-precision support (Actually O1 in amp is mixed-precision)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1') # O1 opt-level by default (it keeps batch_norm in float32)
    
    # parallelization
    if ngpus > 1:
        logger.info(f"Using multiple gpus")
        if args.accelerator == "dp":
            model = nn.DataParallel(model).to(device)
        elif args.accelerator == "ddp":
            model = DDP(model, delay_allreduce=True)
        else:
            raise ValueError("accelerator not supported")
    
    # dataloaders
    train_loader, train_sampler, val_loader, val_sampler, _, _, train_transform = data.__dict__[args.dataset](args.data_dir, batch_size=args.batch_size, mode=args.mode, normalize=args.normalize, size=args.size, workers=args.workers, distributed=args.distributed, classes=args.classes, training_images=args.training_images)
    num_batches = len(train_loader)
    criterion = nn.CrossEntropyLoss()
    
    # Use synthetic data
    syn_sampler = None
    if args.syn_data_list:
        logger.info(f"Using following synthetic datasets: {args.syn_data_list}")
        syn_sampler = []
        for s in args.syn_data_list:
            syn_loader, ssampler = get_synthetic_dataloader(s, args.batch_size_syn, transform=train_transform, workers=args.workers, distributed=args.distributed)
            num_batches = min(num_batches, len(syn_loader))
            train_loader = combine_dataloaders(train_loader, syn_loader) # update training dataloader
            syn_sampler.append(ssampler)
        logger.info(f"Using {num_batches} batches per epoch")
        logger.info(f"Ratio of original to synthetic data per batch {1}:{(len(args.syn_data_list)*args.batch_size_syn)/args.batch_size}")

    # warmup training
    if args.warmup:
        logger.info(f"Warmup training for {args.wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=args.lr, step_size_up=args.wamrup_epochs*num_batches)
        for epoch in range(args.wamrup_epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
                if syn_sampler: 
                    for sampler in syn_sampler:
                        sampler.set_epoch(epoch)
            logger.info(f"Warmup epoch {epoch}")
            _ = getattr(trainers, args.trainer)(model, device, train_loader, criterion, optimizer, num_batches, warmup_lr_scheduler, epoch, args)
        # reset learning rate
        for p in optimizer.param_groups:
            p["lr"] = args.lr
            p["initial_lr"] = args.lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * num_batches, eta_min=0.001)
    
    if args.swa:
        args.swadict = copy.deepcopy(model.state_dict())
    else:
        args.swadict = {}
        
    # Let's roll
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if syn_sampler: 
                for sampler in syn_sampler:
                    sampler.set_epoch(epoch)
                
        results_train = getattr(trainers, args.trainer)(model, device, train_loader, criterion, optimizer, num_batches, lr_scheduler, epoch, args)
        results_val = getattr(utils, args.val_method)(model, device, val_loader, criterion, args, None, epoch=epoch)
        if args.local_rank == 0:
            # remember best prec@1 (only based on clean accuracy) and save checkpoint
            if args.trainer == "baseline":
                prec = results_val["top1"]
            elif args.trainer in ["adv", "madry", "fgsm"]:
                prec = results_val["top1_adv"]
            else:
                raise ValueError()
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)

            d = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec,
                "optimizer": optimizer.state_dict(),
                "swadict": args.swadict
            }

            save_checkpoint(
                d, is_best, result_dir=os.path.join(result_sub_dir, "checkpoint"),
            )
            
            logger.info(f"Epoch {epoch}, " + ", ".join(["{}: {:.3f}".format(k+"_train", v) for (k,v) in results_train.items()]+["{}: {:.3f}".format(k+"_val", v) for (k,v) in results_val.items()]))


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
