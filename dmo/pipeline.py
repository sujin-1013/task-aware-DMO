"""Scenario-1 pipeline stages: single-task training, planning, group training, report.

Each stage reads/writes artifacts under ``results/scenario1`` so the stages can
run as separate processes (e.g. in parallel on different GPUs).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .datasets import SCENARIO1_TASKS, TASK_CLASSES, eval_train_loader, loaders
from .engine import per_sample_losses, train_group, train_single
from .metrics import (combined_similarity, group_tasks, keep_ratio,
                      loss_similarity, task_difficulty, weight_similarity)
from .model import GroupNet, backbone_param_names, single_task_net
from .pruning import build_group_masks, count_surviving, merge_group

PAPER_TARGETS = {"single_avg": 89.95, "dmo_avg": 89.73, "dmo_params_M": 6.8, "ratio": 0.15}


@dataclass
class Config:
    """Paper hyperparameters for Scenario 1 (Sec. IV-A)."""

    repo: Path
    device: str = "cuda:0"
    epochs: int = 25
    batch_size: int = 64
    lr: float = 0.1
    alpha: float = 0.5      # weight vs loss similarity balance (Eq. 1)
    tau_sim: float = 0.45   # similarity edge threshold (Eq. 4)
    tau: float = 1.0        # pruning temperature (Sec. III-E.1)
    seed: int = 0

    @property
    def out(self) -> Path:
        return self.repo / "results" / "scenario1"

    @property
    def data(self) -> Path:
        return self.repo / "data"

    def seed_everything(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


def run_single(cfg: Config, task: str) -> dict:
    """Train one task-specific ResNet-18 and record its loss curve and sample losses."""
    cfg.seed_everything()
    device = torch.device(cfg.device)
    cfg.out.mkdir(parents=True, exist_ok=True)

    train, test = loaders(task, str(cfg.data), cfg.batch_size)
    net = single_task_net(TASK_CLASSES[task])
    curve, acc = train_single(net, train, test, cfg.epochs, cfg.lr,
                              cfg.repo / "checkpoints" / "scenario1" / task, device)
    artifact = {
        "task": task,
        "state": {k: v.cpu() for k, v in net.state_dict().items()},
        "curve": curve,
        "acc": acc,
        "sample_losses": per_sample_losses(net, eval_train_loader(task, str(cfg.data)), device),
    }
    torch.save(artifact, cfg.out / f"single_{task}.pt")
    print(f"[single] {task}: acc={acc:.2f}", flush=True)
    return artifact


def _load_singles(cfg: Config) -> list[dict]:
    return [torch.load(cfg.out / f"single_{t}.pt", weights_only=False) for t in SCENARIO1_TASKS]


def run_plan(cfg: Config) -> dict:
    """Measure similarity and difficulty, then group tasks by maximal cliques."""
    arts = _load_singles(cfg)
    prune_names = backbone_param_names(single_task_net())

    s_w = weight_similarity([a["state"] for a in arts], prune_names)
    s_l = loss_similarity([a["curve"] for a in arts])
    sim = combined_similarity(s_w, s_l, cfg.alpha)
    diff = task_difficulty([a["sample_losses"] for a in arts])
    groups = group_tasks(sim, cfg.tau_sim)

    plan = {
        "tasks": SCENARIO1_TASKS,
        "S_W": s_w.tolist(), "S_L": s_l.tolist(), "S": sim.tolist(),
        "difficulty": diff.tolist(),
        "groups": [[int(i) for i in g] for g in groups],
        "alpha": cfg.alpha, "tau_sim": cfg.tau_sim, "tau": cfg.tau,
    }
    (cfg.out / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"[plan] S=\n{np.round(sim, 3)}", flush=True)
    print(f"[plan] difficulty={dict(zip(SCENARIO1_TASKS, np.round(diff, 4)))}", flush=True)
    print(f"[plan] groups={[[SCENARIO1_TASKS[i] for i in g] for g in groups]}", flush=True)
    return plan


def run_group(cfg: Config, gid: int) -> dict:
    """Prune, merge, and jointly train the gid-th task group from the plan."""
    cfg.seed_everything()
    device = torch.device(cfg.device)
    plan = json.loads((cfg.out / "plan.json").read_text())
    tasks, diff = plan["tasks"], np.array(plan["difficulty"])
    group = plan["groups"][gid]
    g_tasks = [tasks[i] for i in group]
    g_diff = [float(diff[i]) for i in group]
    g_states = [torch.load(cfg.out / f"single_{t}.pt", weights_only=False)["state"] for t in g_tasks]
    prune_names = backbone_param_names(single_task_net())

    masks = build_group_masks(g_states, g_diff, prune_names, plan["tau"])
    merged_w, union_mask = merge_group(g_states, masks, g_diff, prune_names)

    net = GroupNet(g_tasks, TASK_CLASSES)
    hardest = g_states[int(np.argmax(g_diff))]  # non-pruned params (BN, biases) from the hardest task
    backbone_state = {k: v.clone() for k, v in hardest.items() if not k.startswith("fc.")}
    backbone_state.update(merged_w)
    net.backbone.load_state_dict(backbone_state, strict=False)

    g_train, g_test = {}, {}
    for t in g_tasks:
        g_train[t], g_test[t] = loaders(t, str(cfg.data), cfg.batch_size)
    accs = train_group(net, union_mask, g_train, g_test, cfg.epochs, cfg.lr,
                       cfg.repo / "checkpoints" / "scenario1" / f"group{gid}", device)

    surviving = count_surviving(union_mask)
    dense_rest = sum(p.numel() for n, p in net.named_parameters()
                     if n.replace("backbone.", "") not in prune_names)
    result = {
        "gid": gid, "tasks": g_tasks, "acc": accs,
        "keep_ratios": [keep_ratio(d, plan["tau"]) for d in g_diff],
        "surviving": surviving, "dense_rest": dense_rest, "params": surviving + dense_rest,
    }
    (cfg.out / f"group_{gid}.json").write_text(json.dumps(result, indent=2))
    print(f"[group{gid}] tasks={g_tasks} acc={accs} params={(surviving + dense_rest) / 1e6:.2f}M",
          flush=True)
    return result


def run_report(cfg: Config) -> dict:
    """Aggregate single-task and group results into results/scenario1/metrics.json."""
    plan = json.loads((cfg.out / "plan.json").read_text())
    single_acc = {a["task"]: a["acc"] for a in _load_singles(cfg)}
    single_params = sum(p.numel() for p in single_task_net().parameters())
    g_results = [json.loads((cfg.out / f"group_{k}.json").read_text())
                 for k in range(len(plan["groups"]))]

    dmo_acc = {t: r["acc"][t] for r in g_results for t in r["acc"]}
    total = sum(r["params"] for r in g_results)
    n_tasks = len(plan["tasks"])
    report = {
        "tasks": plan["tasks"],
        "single_task": {"acc": single_acc,
                        "avg": float(np.mean(list(single_acc.values()))),
                        "params_M": round(n_tasks * single_params / 1e6, 1)},
        "similarity": {"S_W": plan["S_W"], "S_L": plan["S_L"], "S": plan["S"]},
        "difficulty": dict(zip(plan["tasks"], plan["difficulty"])),
        "groups": [[plan["tasks"][i] for i in g] for g in plan["groups"]],
        "dmo": {"acc": dmo_acc,
                "avg": float(np.mean(list(dmo_acc.values()))),
                "params_M": round(total / 1e6, 2),
                "ratio": round(total / (n_tasks * single_params), 3)},
        "paper_targets": PAPER_TARGETS,
        "config": {"alpha": plan["alpha"], "tau_sim": plan["tau_sim"], "tau": plan["tau"]},
    }
    (cfg.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["single_task", "difficulty", "groups", "dmo"]},
                     indent=1), flush=True)
    return report
