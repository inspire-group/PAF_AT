import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from apex import amp

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def proj_l1ball(x, epsilon=10, device = "cuda:1"):
    assert epsilon > 0
#     ipdb.set_trace()
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y = y.view(-1,3,32,32)
    y *= x.sign()
    return y


def proj_simplex(v, s=1, device = "cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]

    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).half().to(device)
    comp = (vec > (cssv - s)).half()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.HalfTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.half() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def pgd_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
    distance="linf",
    fp16=False,
    baseOptimizer=None,
    perc=99,
    normalize=None
):
    assert distance in ["l1", "linf", "l2"]
    if normalize is None:
        normalize = lambda x: x
    if distance == 'l1':
        x_pgd = x
        r = torch.zeros_like(x_pgd, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(normalize(x_pgd + r)), y)
            if fp16:
                with amp.scale_loss(loss, baseOptimizer) as scaled_loss:
                    grad = torch.autograd.grad(scaled_loss, [r], create_graph=False)[0].detach()
                    grad = grad / (scaled_loss / loss)
            else:
                grad = torch.autograd.grad(loss, [r], create_graph=False)[0].detach()
            grad_mag = torch.abs(grad)
            grad_perc = np.percentile(grad_mag.detach().cpu(), perc)
            e = torch.where(grad_mag >= grad_perc, torch.sign(grad), 0)
            r = r + step_size * e / torch.norm(e, p=1, dim=1)
            r = proj_l1ball(r, epsilon, device)
        x_pgd = torch.clamp(x.data + r, clip_min, clip_max)

    if distance == "linf":
        if is_random:
            random_noise = (
                torch.FloatTensor(x.shape)
                .uniform_(-epsilon, epsilon)
                .to(device)
                .detach()
            )
        x_pgd = Variable(x.detach().data + random_noise, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(normalize(x_pgd)), y)
            if fp16:
                with amp.scale_loss(loss, baseOptimizer) as scaled_loss: 
                    grad = torch.autograd.grad(scaled_loss, [x_pgd], create_graph=False)[0].detach()
                    grad = grad / (scaled_loss / loss)
            else:
                grad = torch.autograd.grad(loss, [x_pgd], create_graph=False)[0].detach()
            x_pgd.data = x_pgd.data + step_size * grad.data.sign()
            eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)

    if distance == "l2":
        if is_random:
            random_noise = (
                torch.FloatTensor(x.shape).uniform_(-1, 1).to(device).detach()
            )
            random_noise.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_pgd = Variable(x.detach().data + random_noise, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(normalize(x_pgd)), y)
            if fp16:
                with amp.scale_loss(loss, baseOptimizer) as scaled_loss: 
                    grad = torch.autograd.grad(scaled_loss, [x_pgd], create_graph=False)[0].detach()
                    grad = grad / (scaled_loss / loss)
            else:
                grad = torch.autograd.grad(loss, [x_pgd], create_graph=False)[0].detach()
            
            # renorming gradient
            grad_norms = grad.view(len(x), -1).norm(p=2, dim=1)
            grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad[grad_norms == 0] = torch.randn_like(
                    grad[grad_norms == 0]
                )
            x_pgd.data += step_size * grad.data
            eta = x_pgd.data - x.data
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)
    return x_pgd

# ref: https://github.com/locuslab/robust_union
def msd_imgs(
    model,
    x,
    y,
    device,
    args,
    clip_min,
    clip_max,
    fp16=False,
    baseOptimizer=None,
    perc=99):
    delta = torch.zeros_like(x, requires_grad=True)
    max_delta = torch.zeros_like(x)
    max_max_delta = torch.zeros_like(x)
    max_max_loss = torch.zeros(y.shape[0]).to(y.device).half()

    for _ in range(args.num_steps):
        logits = model(x + delta)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        with torch.no_grad():
            delta_l2 = delta.data + args.step_size_l2 * delta.grad / torch.norm(delta.grad, p=2, dim=1)
            delta_l2 *= args.epsilon_l2 / torch.norm(delta_l2, p=2, dim=1).clamp(min=args.epsilon_l2)
            delta_l2 = torch.min(torch.max(delta_l2, clip_min-x), clip_max - x)

            # For L_inf
            delta_linf = (delta.data + args.step_size_linf * delta.grad.sign()).clamp(-args.epsilon_linf, args.epsilon_linf)
            delta_linf = torch.min(torch.max(delta_linf, clip_min-x), clip_max - x)  # clip X+delta to [0,1]

            # For L1
            grad_mag = torch.abs(delta.grad)
            grad_perc = np.percentile(grad_mag.detach().cpu(), perc)
            e = torch.where(grad_mag >= grad_perc, torch.sign(delta.grad), 0)
            delta_l1 = delta.data + args.step_size_l1 * e / torch.norm(e, p=1, dim=1)
            delta_l1 = proj_l1ball(delta_l1, args.epsilon_l1, device)
            delta_l1 = torch.min(torch.max(delta_l1, clip_min-x), clip_max-x)

            # Compare
            delta_tup = (delta_l1, delta_l2, delta_linf)
            max_loss = torch.zeros(y.shape[0]).to(y.device).half()
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction='none')(model(x + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp)
            delta.data = max_delta.data
            max_max_delta[max_loss > max_max_loss] = max_delta[max_loss > max_max_loss]
            max_max_loss[max_loss > max_max_loss] = max_loss[max_loss > max_max_loss]
        delta.grad.zero_()
    return x + delta


# ref: https://github.com/yaodongyu/TRADES
def trades_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    beta,
    clip_min,
    clip_max,
    distance="linf",
    natural_criterion=nn.CrossEntropyLoss(),
    fp16=False,
    normalize=None
):
    if normalize is None:
        normalize = lambda x: x
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
        x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
    optimizer.zero_grad()
    
    if distance == "linf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(normalize(x_adv)), dim=1),
                    F.softmax(model(normalize(x_natural)), dim=1),
                )
            if fp16:
                with amp.scale_loss(loss_kl, optimizer) as scaled_loss: 
                    grad = torch.autograd.grad(scaled_loss, [x_adv])[0]
                    grad = grad / (scaled_loss / loss_kl)
            else:
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            #grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(normalize(adv)), dim=1), F.softmax(model(normalize(x_natural)), dim=1)
                )
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss: 
                    scaled_loss.backward()
                    grad = delta.grad
                    grad = grad / (scaled_loss / loss)
            else:
                loss.backward()
                grad = delta.grad
            # renorming gradient
            grad_norms = grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits, logits_adv = model(normalize(x_natural)), model(normalize(x_adv))
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1)
    )
    loss = loss_natural + beta * loss_robust
    return loss, logits, logits_adv
