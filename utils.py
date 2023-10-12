import os
import numpy as np
import math
import glob
from PIL import Image
from collections import OrderedDict
from easydict import EasyDict
import time
import shutil, errno
import yaml
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import random
import pickle

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from autoattack import AutoAttack
from torch.utils.data.dataset import Dataset

import data
from utils_adv import pgd_whitebox


def update_args(args):
    with open(args.configs) as f:
        new_args = EasyDict(yaml.load(f))
    
    for k, v in vars(args).items():
        if k in list(new_args.keys()):
            if v:
                new_args[k] = v
        else:
            new_args[k] = v
    
    return new_args


def display_vectors(images):
    if len(images) > 64:
        images = images[:64]
    if torch.is_tensor(images):
        images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))

    d = int(math.sqrt(len(images)))
    plt.figure(figsize=(8, 8))
    image = np.concatenate(
        [
            np.concatenate([images[d * i + j] for j in range(d)], axis=0)
            for i in range(d)
        ],
        axis=1,
    )
    if image.shape[-1] == 1:
        plt.imshow(image[:, :, 0], cmap="gray")
    else:
        plt.imshow(image)
        

def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d
   
    
def remove_module(d):
    return OrderedDict({(k[len("module."):], v) for (k, v) in d.items()})
    

def save_checkpoint(state, is_best, result_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(os.path.join(sub_dir, "checkpoint"), exist_ok=True)


def write_to_file(file, data, option):
    with open(file, option) as f:
        f.write(data)


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)

    def write_avg_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.avg, global_step)
            
            
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def baseline(model, device, val_loader, criterion, args, normalize, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    if normalize is None:
        normalize = lambda x: x
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top2], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(normalize(images))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 and args.local_rank == 0:
                progress.display(i)

        if args.local_rank == 0:
            progress.display(i)  # print final results

    result = {"top1": top1.avg, "top2":  top2.avg}
    return result


def adv(model, device, val_loader, criterion, args, normalize, epoch=0):
    """
        Evaluate on adversarial validation set inputs.
    """
    if normalize is None:
        normalize = lambda x: x

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_top2 = AverageMeter("Adv-Acc_2", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top2, adv_top1, adv_top2],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(normalize(images))
            loss = criterion(output, target)

            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            # adversarial images
            images = pgd_whitebox(
                model,
                images,
                target,
                device,
                args.epsilon,
                args.num_steps,
                args.step_size,
                args.clip_min,
                args.clip_max,
                is_random=True,
                distance=args.distance,
                normalize=normalize,
            )

            # compute output
            output = model(normalize(images))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 and args.local_rank == 0:
                progress.display(i)

        if args.local_rank == 0:
            progress.display(i)  # print final results
    
    result = {"top1": top1.avg, "top2":  top2.avg, "top1_adv": adv_top1.avg, "top2_adv": adv_top2.avg}
    return result

def auto2(model, device, val_loader, criterion, args, epoch=0):
    """
        Evaluate on atuo-attack adversarial validation set inputs.
    """
    attacks = ['apgd-ce','apgd-t', 'fab', 'square']
    stats_per_atk = {}
    for attack in attacks:
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        adv_losses = AverageMeter("Adv_Loss", ":.4f")
        top1 = AverageMeter("Acc_1", ":6.2f")
        top2 = AverageMeter("Acc_2", ":6.2f")
        adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
        adv_top2 = AverageMeter("Adv-Acc_2", ":6.2f")
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, adv_losses, top1, top2, adv_top1, adv_top2],
            prefix="Test: ",
        )
        stats = {'batch_time': batch_time,
                'losses': losses,
                'adv_losses': adv_losses,
                'top1': top1,
                'top2': top2,
                'adv_top1': adv_top1,
                'adv_top2': adv_top2,
                'progress': progress}

        stats_per_atk[attack] = stats

    # switch to evaluation mode
    model.eval()
    assert args.distance in ["linf", "l2"]

#    print("USING ONLY APGD_CE & APGD-T attacks. Rest of them don't change robust accuracy much but take super long to finish.")
    adversary = AutoAttack(model, norm="Linf" if args.distance=="linf" else "L2", eps=args.epsilon)
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc2 = accuracy(output, target, topk=(1, 2))

            images_dict = adversary.run_standard_evaluation_individual(images, target, bs=len(images))

            for key in images_dict:
                # add clean image stats
                stats_per_atk[key]['losses'].update(loss.item(), images.size(0))
                stats_per_atk[key]['top1'].update(acc1[0], images.size(0))
                stats_per_atk[key]['top2'].update(acc2[0], images.size(0))
                images = images_dict[key]

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc2 = accuracy(output, target, topk=(1, 2))

                stats_per_atk[key]['adv_losses'].update(loss.item(), images.size(0))
                stats_per_atk[key]['adv_top1'].update(acc1[0], images.size(0))
                stats_per_atk[key]['adv_top2'].update(acc2[0], images.size(0))
                # measure elapsed time
                stats_per_atk[key]['batch_time'].update(time.time() - end)
                end = time.time()

            if (i + 1) % args.print_freq == 0 and args.local_rank == 0:
                for key in stats_per_atk:
                    print(key)
                    stats_per_atk['progress'].display(i)

        if args.local_rank == 0:
            for key in stats_per_atk:
                print(key)
                stats_per_atk['progress'].display(i)  # print final results
    result = {}
    for key in stats_per_atk:
        top1 = stats_per_atk[key]['top1'].avg
        top2 = stats_per_atk[key]['top2'].avg
        top1_adv = stats_per_atk[key]['adv_top1'].avg
        top2_adv = stats_per_atk[key]['adv_top2'].avg
        result[key] ={'top1': top1, 'top2': top2, 'top1_adv': top1_adv, 'top2_adv': top2_adv}
    return result

def auto(model, device, val_loader, criterion, args, epoch=0):
    """
        Evaluate on atuo-attack adversarial validation set inputs.
    """

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_top2 = AverageMeter("Adv-Acc_2", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top2, adv_top1, adv_top2],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()
    assert args.distance in ["linf", "l2"]

#    print("USING ONLY APGD_CE & APGD-T attacks. Rest of them don't change robust accuracy much but take super long to finish.")
    adversary = AutoAttack(model, norm="Linf" if args.distance=="linf" else "L2", eps=args.epsilon)
    #adversary.attacks_to_run = ['apgd-ce', 'apgd-t']

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            images = adversary.run_standard_evaluation(images, target, bs=len(images))
            
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 and args.local_rank == 0:
                progress.display(i)

        if args.local_rank == 0:
            progress.display(i)  # print final results

    result = {"top1": top1.avg, "top2":  top2.avg, "top1_adv": adv_top1.avg, "top2_adv": adv_top2.avg}
    return result

############################# synthetic dataset ############################# 
class combine_dataloaders:
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
    
    def __iter__(self):
        return self._iterator()
    
    def _iterator(self):
        for (img1, label1), (img2, label2) in zip(self.dataloader1, self.dataloader2):
            images = torch.cat([img1, img2])
            labels = torch.cat([label1, label2])
            indices = torch.randperm(len(images))
            yield images[indices], labels[indices]


class cifar10_custom_unconditional_dataset(torch.utils.data.Dataset):
    def __init__(self, datadir, transform=None):
        self.datadir = datadir
        self.transform = transform
        
        self.files = [sorted(glob.glob(os.path.join(d, "*.png"))) for d in glob.glob(os.path.join(datadir, "*"))]
        self.k = 50000
        print(f"Numbers of cleaned up images per class {[len(f) for f in self.files]}")
        print(f"Using {self.k} images per class")
        
        self.clean_files = []
        self.labels = []
        for c in range(10):
            self.clean_files += [f for f in self.files[c][:self.k]]
            self.labels += [c]*self.k

    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        img, label = Image.open(self.clean_files[idx]), self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    
def tinyimages500k_dataset():
    with open("/scratch/gpfs/sihuid/data/ti_500K_pseudo_labeled.pickle", "rb") as f:
        data = pickle.load(f)
    images, labels = torch.tensor(data["data"]).permute(0, 3, 1, 2).float() / 255.0, torch.tensor(data["extrapolated_targets"]).long()
    ti_dataset = torch.utils.data.TensorDataset(images, labels)
    return ti_dataset
    
    
def get_synthetic_dataloader(name, batch_size, transform=None, workers=4, distributed=False):
    if name == "styleganU_cifar10":
        dataset = cifar10_custom_unconditional_dataset("/data/data_vvikash/fall20/stylegan_ada/cifar10/unconditional/", transform)
        print("Using original transformation since we have only small number of images per class")
        print(f"Number of StyleGAN-ADA unconditional cifar-10 generated images {len(dataset)}")
    elif name == "styleganC_cifar10":
        dataset = torchvision.datasets.ImageFolder("/data/data_vvikash/fall20/stylegan_ada/cifar10/conditional/", transform=transforms.ToTensor())
        print("Not using any transformation since we have infinite amount of images")
        print(f"Number of StyleGAN-ADA conditional cifar-10 generated images {len(dataset)}")
    elif name == "diffusion_cifar10":
        dataset = torchvision.datasets.ImageFolder("/scratch/gpfs/sihuid/data/cifar10_ddpm", transform=transforms.ToTensor())
        print("Not using any transformation since we have infinite amount of images")
        print(f"Number of Denoising-Diffusion-Probabilistic-Model unconditional cifar-10 generated images {len(dataset)}")
    elif name == "diffusion_cifar100":
        dataset = torchvision.datasets.ImageFolder("/scratch/gpfs/sihuid/data/cifar100_ddpm", transform=transforms.ToTensor())
    elif name == "ti500k_cifar10":
        dataset = tinyimages500k_dataset()
        print("Using no transformations since we have 500k images")
    elif name == "styleganC_imagenette":
        print("Using standard data augmentation policies for imagenette synthetic dataset. Also hardcoding image size, normalization")
        dd = data.imagenette(datadir="/data/nvme/tinashe/datasets/synthetic/", batch_size=batch_size, mode="org", size=224, normalize=False, norm_layer=None, workers=workers, distributed=distributed)
        dataset = dd[0].dataset # only use training paritition form synthetic images
    else:
        raise ValueError(f"Synthetic data {name} not available")
    
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=workers, pin_memory=True)
    return loader, sampler


def update_swadict(dictold, dictnew, tau):
    for (k, v) in dictold.items():
        dictold[k] = tau * dictold[k] + (1 - tau) * dictnew[k]
    
        
