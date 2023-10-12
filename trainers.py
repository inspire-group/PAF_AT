import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from apex import amp
from torch.autograd import Variable
from utils import AverageMeter, ProgressMeter
from utils import accuracy, update_swadict
from utils_adv import pgd_whitebox, msd_imgs
from utils_adv import trades_loss
    
    
def baseline(model, device, dataloader, criterion, optimizer, num_batches=0, lr_scheduler=None, epoch=0, args=None, normalize=None, **kwargs):
    if normalize is None:
        normalize = lambda x: x
    if args.local_rank == 0:
        print(" ->->->->->->->->->-> One epoch with Baseline natural training <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, top1, top2],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    
    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)
        
        # basic properties of training
        if i == 0 and args.local_rank == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )
        data_time.update(time.time() - end)
        
        output = model(normalize(images))
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        lr_scheduler.step()
        
        if args.swa:
            update_swadict(args.swadict, model.state_dict(), args.tau) # swadict = tau * swadict + (1 - tau) * modeldict
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.display(i)
    
    result = {"top1": top1.avg, "top2":  top2.avg}
    return result


def fgsm(model, device, dataloader, criterion, optimizer, num_batches=0, lr_scheduler=None, epoch=0, args=None, normalize=None, **kwargs):
    if normalize is None:
        normalize = lambda x: x
    if args.local_rank == 0:
        print(" ->->->->->->->->->-> One epoch with Adversarial (FGSM) training (only support linf attack) <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    top1_adv = AverageMeter("Acc_1_adv", ":6.2f")
    top2_adv = AverageMeter("Acc_2_adv", ":6.2f")
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, top1, top2, top1_adv, top2_adv],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    assert args.distance == "linf"
    
    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0 and args.local_rank == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )
        data_time.update(time.time() - end)
        
        # generate adverarial examples
        model.eval()
        step = 1.25 * args.epsilon
        eps = Variable(
            torch.zeros_like(images).uniform_(-args.epsilon, args.epsilon),
            requires_grad=True,
        )
        optimizer.zero_grad()
        logits = model(normalize(torch.clamp(images + eps, args.clip_min, args.clip_max))) # approximately equal to clean image output
        loss = criterion(logits, target)
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        eps.data = torch.clamp(
            eps.data + step * eps.grad.data.sign(), -args.epsilon, args.epsilon
        )
        eps.data = torch.clamp(images + eps.data, args.clip_min, args.clip_max) - images
        eps = eps.detach()
        model.train()
        
        # adv training
        logits_adv = model(normalize(images + eps))
        loss = criterion(logits_adv, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(logits, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))
        acc1_adv, acc2_adv = accuracy(logits_adv, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1_adv.update(acc1_adv[0], images.size(0))
        top2_adv.update(acc2_adv[0], images.size(0))

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        lr_scheduler.step()
        
        if args.swa:
            update_swadict(args.swadict, model.state_dict(), args.tau) # swadict = tau * swadict + (1 - tau) * modeldict
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.display(i)
    result = {"top1": top1.avg, "top2":  top2.avg, "top1_adv": top1_adv.avg, "top2_adv": top2_adv.avg}
    return result

def madry(model, device, dataloader, criterion, optimizer, num_batches=0, lr_scheduler=None, epoch=0, args=None, normalize=None, **kwargs):
    if normalize is None:
        normalize = lambda x: x
    if args.local_rank == 0:
        print(" ->->->->->->->->->-> One epoch with PGD Adversarial training <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    top1_adv = AverageMeter("Acc_1_adv", ":6.2f")
    top2_adv = AverageMeter("Acc_2_adv", ":6.2f")
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, top1, top2, top1_adv, top2_adv],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    
    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)
        
        # basic properties of training
        if i == 0 and args.local_rank == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )
        data_time.update(time.time() - end)

        logits = model(images)
        
        model.eval()
        advImages = pgd_whitebox(model, images, target, device, args.epsilon, args.num_steps, args.step_size, args.clip_min, args.clip_max, 
                                  is_random=True, distance=args.distance, fp16=args.fp16, baseOptimizer=optimizer, normalize=normalize)
        model.train()
        
        logits_adv = model(normalize(advImages))
        loss = criterion(logits_adv, target)
        if args.activation == 'pssilu2':
            if args.arch in ['resnet18', 'ResNet18', 'wrn_28_10']:
                if args.pssilu_reg_type == 'l1':
                    loss += args.pssilu_reg * torch.abs( model.activation.beta.squeeze())
                elif args.pssilu_reg_type == 'l2':
                    loss += args.pssilu_reg * model.activation.beta.squeeze()**2
            elif args.arch == 'vgg16_bn':
                if args.pssilu_reg_type == 'l1':
                    loss += args.pssilu_reg * torch.abs( model.features[2].beta.squeeze())
                elif args.pssilu_reg_type == 'l2':
                    loss += args.pssilu_reg * model.features[2].beta.squeeze()**2
        
        # measure accuracy and record loss
        acc1, acc2 = accuracy(logits, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))
        acc1_adv, acc2_adv = accuracy(logits_adv, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1_adv.update(acc1_adv[0], images.size(0))
        top2_adv.update(acc2_adv[0], images.size(0))

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if args.activation == 'pssilu2':
            if args.arch in ['resnet18', 'ResNet18', 'wrn_28_10']:
                torch.nn.utils.clip_grad_norm_(model.activation.beta, 0.1)
            elif args.arch == 'vgg16_bn':
                torch.nn.utils.clip_grad_norm_(model.features[2].beta, 0.1)

        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.swa:
            update_swadict(args.swadict, model.state_dict(), args.tau) # swadict = tau * swadict + (1 - tau) * modeldict
        
        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.display(i)
    result = {"top1": top1.avg, "top2":  top2.avg, "top1_adv": top1_adv.avg, "top2_adv": top2_adv.avg}
    return result


def adv(model, device, dataloader, criterion, optimizer, num_batches=0, lr_scheduler=None, epoch=0, args=None, normalize=None, **kwargs):
    if normalize is None:
        normalize = lambda x: x
    if args.local_rank == 0:
        print(" ->->->->->->->->->-> One epoch with Adversarial (Trades) training <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    top1_adv = AverageMeter("Acc_1_adv", ":6.2f")
    top2_adv = AverageMeter("Acc_2_adv", ":6.2f")
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, top1, top2, top1_adv, top2_adv],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    
    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)
        
        # basic properties of training
        if i == 0 and args.local_rank == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )
        data_time.update(time.time() - end)

        # calculate robust loss
        loss, logits, logits_adv = trades_loss(
            model=model,
            x_natural=images,
            y=target,
            device=device,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            distance=args.distance,
            fp16=args.fp16,
            normalize=normalize
        )

        # measure accuracy and record loss
        acc1, acc2 = accuracy(logits, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))
        acc1_adv, acc2_adv = accuracy(logits_adv, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1_adv.update(acc1_adv[0], images.size(0))
        top2_adv.update(acc2_adv[0], images.size(0))
        if args.activation == 'pssilu2':
            if args.arch in ['resnet18', 'ResNet18', 'wrn_28_10']:
                if args.pssilu_reg_type == 'l1':
                    loss += args.pssilu_reg * torch.abs( model.activation.beta.squeeze())
                elif args.pssilu_reg_type == 'l2':
                    loss += args.pssilu_reg * model.activation.beta.squeeze()**2
            elif args.arch == 'vgg16_bn':
                if args.pssilu_reg_type == 'l1':
                    loss += args.pssilu_reg * torch.abs( model.features[2].beta.squeeze())
                elif args.pssilu_reg_type == 'l2':
                    loss += args.pssilu_reg * model.features[2].beta.squeeze()**2

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if args.activation == 'pssilu2':
            if args.arch in ['resnet18', 'ResNet18', 'wrn_28_10']:
                torch.nn.utils.clip_grad_norm_(model.activation.beta, 0.1)
            elif args.arch == 'vgg16_bn':
                torch.nn.utils.clip_grad_norm_(model.features[2].beta, 0.1)


        optimizer.step()
        lr_scheduler.step()
        
        if args.swa:
            update_swadict(args.swadict, model.state_dict(), args.tau) # swadict = tau * swadict + (1 - tau) * modeldict
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.display(i)
    result = {"top1": top1.avg, "top2":  top2.avg, "top1_adv": top1_adv.avg, "top2_adv": top2_adv.avg}
    return result

