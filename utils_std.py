import torch
import torch.nn as nn
import os
import trainers
import utils
from utils import *
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F
from copy import deepcopy
from autoattack.square import SquareAttack

def square(model, dataloader, seed, queries):
    square = SquareAttack(model, n_queries=queries, eps=0.031, device='cuda')
    square.seed = seed
    success_rate = 0
    num_in = 0
    for i, (img, target) in enumerate(dataloader):
        with torch.no_grad():
            img, target = img.cuda(), target.cuda()
            # Note: we count the queries only across successful attacks
            # and initial classification must be correct
            logits_clean = model(img)
            corr_classified = (torch.max(logits_clean, 1)[1] == target)
            x_adv = square.perturb(img, target)
            logits_adv = model(x_adv)
            success = (torch.max(logits_adv, 1)[1] != target)
            mask = corr_classified * success
            success_rate += mask.sum().item()
            num_in += corr_classified.sum().item()
    return success_rate / num_in


def PGD(model, x, y, eps, steps=4, step_size = 0.0078):
    eps = eps.reshape(eps.size(0), 1, 1, 1)
    x_pgd = Variable(x.detach().data, requires_grad=True)
    for _ in range(steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_pgd), y)
            grad = torch.autograd.grad(loss, [x_pgd], create_graph=False)[0].detach()
            x_pgd.data = x_pgd.data + step_size * grad.data.sign()
            eta = torch.min(torch.max(x_pgd.data - x.data, -eps), eps)
            x_pgd.data = torch.clamp(x.data + eta, 0, 1)
    return x_pgd.detach()

def PGD2(model, x, y, eps, steps=4, step_size = 0.0078):
    x_pgd = Variable(x.detach().data, requires_grad=True)
    for _ in range(steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_pgd), y)
            grad = torch.autograd.grad(loss, [x_pgd], create_graph=False)[0].detach()
            x_pgd.data = x_pgd.data + step_size * grad.data.sign()
            eta = torch.clamp(x_pgd.data - x.data, min=-eps, max=eps)
            x_pgd.data = torch.clamp(x.data + eta, 0, 1)
    return x_pgd.detach()


def get_act_diff(model, x, y, eps, steps, step_size = 0.0078):
     x_pgd = PGD2(model, x, y, eps, steps, step_size)
     
     fm_clean = model.conv1(x)
     fm_clean = model.bn1(fm_clean)
     fm_clean = model.activation(fm_clean)
     fm_clean = model.layer1(fm_clean)
     fm_clean = model.layer2(fm_clean)

     fm_adv = model.conv1(x_pgd)
     fm_adv = model.bn1(fm_adv)
     fm_adv = model.activation(fm_adv)
     fm_adv = model.layer1(fm_adv)
     fm_adv = model.layer2(fm_adv)

     return torch.norm(fm_clean - fm_adv, 2)


def bin_rad_search(model, x, label):
    step_size= 0.0078
    eps = torch.ones(x.size(0)).cuda() * 0.05
    eps_prev = torch.zeros(x.size(0)).cuda()
    out_clean = model(x)
    originally_correct = (torch.max(out_clean, 1)[1] == label)
    while (eps - eps_prev).abs().sum() > 1e-7:
        temp = deepcopy(eps)

        pgd_x = PGD(model, x, label, eps, step_size=step_size)

        # is fgsm successful
        out = model(pgd_x)
        is_correct = (torch.max(out, 1)[1] == label)

        # if predicted correctly, increase eps
        eps[is_correct] += (eps_prev[is_correct] - eps[is_correct]).abs() / 2

        # if predicted incorrectly, decrease eps
        not_correct = torch.logical_not(is_correct)
        eps[not_correct] -= (eps_prev[not_correct] - eps[not_correct]).abs() / 2
        eps_prev = temp
        step_size /= 2
    return eps[originally_correct]

def avg_radius(model, dataloader):
    model.eval()
    sum_eps = 0
    num_imgs = 0
    for i, (data, label) in enumerate(dataloader):
        data, label = data.cuda(), label.cuda()
        eps = bin_rad_search(model, data, label)
        sum_eps += eps.sum().item()
        num_imgs += len(eps)

    return sum_eps / num_imgs

def get_act_diff_all(model, dataloader):
    model.eval()
    sum_diff = 0
    for i, (data, label) in enumerate(dataloader):
        data, label = data.cuda(), label.cuda()
        sum_diff += get_act_diff(model, data, label, 0.031, steps=10, step_size = 0.0078).item()
    return sum_diff / len(dataloader.dataset)

def empirical_lipschitz(model, dataloader):
    avg_L = 0
    for i, (data, label) in enumerate(dataloader):
        data, label = data.cuda(), label.cuda()
        adv = PGD2(model, data, label, 0.031, steps=10, step_size = 0.0078)
        with torch.no_grad():
            diff_out = torch.norm(model(data) - model(adv), p=1)
            diff_img = torch.norm(adv - data, float('inf'))
            avg_L += diff_out / diff_img
    return avg_L / len(dataloader.dataset)

