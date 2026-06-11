"""ResNet-18 backbone with task-specific heads."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


def single_task_net(num_classes: int = 10) -> nn.Module:
    net = resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


class GroupNet(nn.Module):
    """Shared ResNet-18 backbone with one classification head per task in the group."""

    def __init__(self, task_names: list[str], num_classes: dict[str, int]):
        super().__init__()
        net = resnet18(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.heads = nn.ModuleDict({t: nn.Linear(feat_dim, num_classes[t]) for t in task_names})

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.backbone(x))


def backbone_param_names(net: nn.Module) -> list[str]:
    """Weight tensors subject to pruning: conv/linear weights of the backbone (no BN, no bias)."""
    names = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name != "fc" and not name.startswith("heads"):
            names.append(f"{name}.weight")
    return names
