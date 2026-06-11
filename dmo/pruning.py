"""Difficulty-aware magnitude pruning and group-wise parameter merging (paper Sec. III-D)."""
from __future__ import annotations

import torch

from .metrics import keep_ratio


def magnitude_mask(state: dict, prune_names: list[str], keep: float) -> dict:
    """Global unstructured magnitude mask over the backbone weights, keeping ``keep`` fraction."""
    all_mags = torch.cat([state[n].abs().flatten() for n in prune_names])
    k = max(1, int(round(keep * all_mags.numel())))
    threshold = torch.topk(all_mags, k, largest=True).values.min()
    return {n: (state[n].abs() >= threshold).float() for n in prune_names}


def merge_group(
    states: list[dict],
    masks: list[dict],
    difficulties: list[float],
    prune_names: list[str],
) -> tuple[dict, dict]:
    """Merge surviving task-specific parameters into group weights and a union mask.

    Where surviving positions overlap, the parameter of the higher-difficulty task
    is prioritized (paper Sec. III-D).
    """
    order = sorted(range(len(states)), key=lambda i: difficulties[i])  # low -> high difficulty
    merged_w, merged_m = {}, {}
    for n in prune_names:
        w = torch.zeros_like(states[0][n])
        m = torch.zeros_like(states[0][n])
        for i in order:  # higher-difficulty tasks written last -> win overlaps
            mask = masks[i][n]
            w = torch.where(mask.bool(), states[i][n], w)
            m = torch.maximum(m, mask)
        merged_w[n], merged_m[n] = w, m
    return merged_w, merged_m


def build_group_masks(states, difficulties, prune_names, tau: float = 1.0):
    masks = [magnitude_mask(s, prune_names, keep_ratio(d, tau)) for s, d in zip(states, difficulties)]
    return masks


def apply_mask_(module_state: dict, mask: dict) -> None:
    for n, m in mask.items():
        module_state[n].mul_(m)


def count_surviving(mask: dict) -> int:
    return int(sum(m.sum().item() for m in mask.values()))
