"""Task similarity (paper Eq. 1), difficulty (Eq. 2-3), and grouping (Eq. 4)."""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch


def _flat_backbone(state: dict, prune_names: list[str]) -> np.ndarray:
    return torch.cat([state[n].flatten() for n in prune_names]).cpu().numpy()


def weight_similarity(states: list[dict], prune_names: list[str]) -> np.ndarray:
    """S_W: Pearson correlation between flattened backbone weights of task pairs.

    All task networks share the same architecture, so weights align elementwise.
    """
    vecs = [_flat_backbone(s, prune_names) for s in states]
    return np.corrcoef(np.stack(vecs))


def loss_similarity(loss_curves: list[np.ndarray]) -> np.ndarray:
    """S_L: Pearson correlation between per-epoch training loss curves."""
    return np.corrcoef(np.stack(loss_curves))


def combined_similarity(s_w: np.ndarray, s_l: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return alpha * s_w + (1.0 - alpha) * s_l


def task_difficulty(sample_losses: list[np.ndarray]) -> np.ndarray:
    """d^t = fraction of samples whose loss exceeds the across-task mean sample loss (Eq. 2-3)."""
    mu = float(np.mean([losses.mean() for losses in sample_losses]))
    return np.array([float((losses >= mu).mean()) for losses in sample_losses])


def group_tasks(sim: np.ndarray, tau_sim: float) -> list[list[int]]:
    """Maximal-clique grouping (Bron-Kerbosch) over the similarity graph (Eq. 4).

    Cliques are visited largest-first; tasks join the first clique-derived group
    they appear in, and any unassigned task forms its own group.
    """
    n = len(sim)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > tau_sim:
                g.add_edge(i, j)
    def clique_key(c):
        if len(c) < 2:
            return (len(c), 0.0)
        pair_sims = [sim[i, j] for ii, i in enumerate(c) for j in c[ii + 1:]]
        return (len(c), float(np.mean(pair_sims)))

    cliques = sorted(nx.find_cliques(g), key=clique_key, reverse=True)
    assigned, groups = set(), []
    for c in cliques:
        fresh = [t for t in sorted(c) if t not in assigned]
        if fresh:
            groups.append(fresh)
            assigned.update(fresh)
    for t in range(n):
        if t not in assigned:
            groups.append([t])
    return groups


def keep_ratio(difficulty: float, tau: float = 1.0) -> float:
    """Keep fraction under the difficulty-aware pruning ratio 1/exp(d/tau) (paper Sec. III-E.1).

    The pruning ratio decreases with difficulty (paper Fig. 5): an easy task
    (d ~ 0) is pruned almost entirely, a hard task keeps more of its weights.
    """
    return float(1.0 - 1.0 / np.exp(difficulty / tau))
