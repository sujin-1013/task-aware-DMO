"""Training and evaluation loops. Every run saves one checkpoint file per epoch."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_single(net, train_loader, test_loader, epochs, lr, ckpt_dir: Path, device,
                 weight_decay: float = 5e-4):
    """Single-task training. Returns (per-epoch mean train loss curve, final test accuracy)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    net = net.to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    curve = []
    for ep in range(epochs):
        net.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(x); n += len(x)
        sched.step()
        curve.append(total / n)
        torch.save({"model": net.state_dict(), "epoch": ep}, ckpt_dir / f"ckpt_epoch_{ep:03d}.pt")
    return np.array(curve), evaluate(net, test_loader, device)


@torch.no_grad()
def evaluate(net, loader, device, task: str | None = None) -> float:
    net.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x, task) if task is not None else net(x)
        correct += (logits.argmax(1) == y).sum().item(); n += len(y)
    return 100.0 * correct / n


@torch.no_grad()
def per_sample_losses(net, loader, device) -> np.ndarray:
    net.eval()
    out = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out.append(F.cross_entropy(net(x), y, reduction="none").cpu().numpy())
    return np.concatenate(out)


def train_group(net, mask, task_loaders, task_test_loaders, epochs, lr, ckpt_dir: Path, device,
                weight_decay: float = 5e-4):
    """Joint training of a group-specific model with a fixed sparsity mask on the backbone.

    Each step draws one batch per task and sums the cross-entropy losses (paper Eq. 5);
    masked weights are zeroed after every update so pruned positions stay inactive.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    net = net.to(device)
    mask_dev = {n: m.to(device) for n, m in mask.items()}
    backbone_params = dict(net.backbone.named_parameters())

    def clamp_():
        with torch.no_grad():
            for n, m in mask_dev.items():
                backbone_params[n].mul_(m)

    clamp_()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tasks = list(task_loaders.keys())
    steps = max(len(dl) for dl in task_loaders.values())
    for ep in range(epochs):
        net.train()
        iters = {t: itertools.cycle(task_loaders[t]) for t in tasks}
        for _ in range(steps):
            loss = 0.0
            for t in tasks:
                x, y = next(iters[t])
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss = loss + F.cross_entropy(net(x, t), y)
            opt.zero_grad(); loss.backward(); opt.step(); clamp_()
        sched.step()
        torch.save({"model": net.state_dict(), "epoch": ep}, ckpt_dir / f"ckpt_epoch_{ep:03d}.pt")
    return {t: evaluate(net, task_test_loaders[t], device, task=t) for t in tasks}
