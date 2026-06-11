#!/usr/bin/env python3
"""Scenario 1 end-to-end: CIFAR-10, STL-10, USPS, MNIST with ResNet-18 (paper Table 1).

Staged for multi-GPU parallelism:
  single --task T --device D   train one task net + measure per-sample losses
  plan                         similarity + difficulty + maximal-clique grouping
  group --gid K --device D     train one group-specific model from the plan
  report                       aggregate final metrics.json

Targets (paper): single-task avg 89.95 @ 44.8M; DMO avg 89.73 @ 6.8M (0.15x).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dmo.datasets import SCENARIO1_TASKS, TASK_CLASSES, eval_train_loader, loaders
from dmo.engine import per_sample_losses, train_group, train_single
from dmo.metrics import (combined_similarity, group_tasks, keep_ratio,
                         loss_similarity, task_difficulty, weight_similarity)
from dmo.model import GroupNet, backbone_param_names, single_task_net
from dmo.pruning import build_group_masks, count_surviving, merge_group

REPO = Path(__file__).resolve().parent
OUT = REPO / "results" / "scenario1"
DATA = REPO / "data"


def cmd_single(args) -> None:
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    t = args.task
    tr, te = loaders(t, str(DATA), args.batch_size)
    net = single_task_net(TASK_CLASSES[t])
    curve, acc = train_single(net, tr, te, args.epochs, args.lr,
                              REPO / "checkpoints" / "scenario1" / t, device)
    losses = per_sample_losses(net, eval_train_loader(t, str(DATA)), device)
    torch.save({"task": t, "state": {k: v.cpu() for k, v in net.state_dict().items()},
                "curve": curve, "acc": acc, "sample_losses": losses},
               OUT / f"single_{t}.pt")
    print(f"[single] {t}: acc={acc:.2f} -> {OUT / f'single_{t}.pt'}", flush=True)


def _load_singles():
    arts = [torch.load(OUT / f"single_{t}.pt", weights_only=False) for t in SCENARIO1_TASKS]
    return arts


def cmd_plan(args) -> None:
    arts = _load_singles()
    tasks = SCENARIO1_TASKS
    prune_names = backbone_param_names(single_task_net())
    s_w = weight_similarity([a["state"] for a in arts], prune_names)
    s_l = loss_similarity([a["curve"] for a in arts])
    sim = combined_similarity(s_w, s_l, args.alpha)
    diff = task_difficulty([a["sample_losses"] for a in arts])
    groups = group_tasks(sim, args.tau_sim)
    plan = {"tasks": tasks,
            "S_W": s_w.tolist(), "S_L": s_l.tolist(), "S": sim.tolist(),
            "difficulty": diff.tolist(),
            "groups": [[int(i) for i in g] for g in groups],
            "alpha": args.alpha, "tau_sim": args.tau_sim, "tau": args.tau}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"[plan] S=\n{np.round(sim,3)}", flush=True)
    print(f"[plan] difficulty={dict(zip(tasks, np.round(diff,4)))}", flush=True)
    print(f"[plan] groups={[[tasks[i] for i in g] for g in groups]}", flush=True)


def cmd_group(args) -> None:
    device = torch.device(args.device)
    plan = json.loads((OUT / "plan.json").read_text())
    tasks, diff = plan["tasks"], np.array(plan["difficulty"])
    g = plan["groups"][args.gid]
    g_tasks = [tasks[i] for i in g]
    arts = {t: torch.load(OUT / f"single_{t}.pt", weights_only=False) for t in g_tasks}
    g_states = [arts[t]["state"] for t in g_tasks]
    g_diff = [float(diff[i]) for i in g]
    prune_names = backbone_param_names(single_task_net())

    masks = build_group_masks(g_states, g_diff, prune_names, plan["tau"])
    merged_w, union_mask = merge_group(g_states, masks, g_diff, prune_names)

    net = GroupNet(g_tasks, TASK_CLASSES)
    init = g_states[int(np.argmax(g_diff))]  # non-pruned params from the hardest task
    backbone_state = {k: v.clone() for k, v in init.items() if not k.startswith("fc.")}
    backbone_state.update(merged_w)
    net.backbone.load_state_dict(backbone_state, strict=False)

    g_train, g_test = {}, {}
    for t in g_tasks:
        g_train[t], g_test[t] = loaders(t, str(DATA), args.batch_size)
    accs = train_group(net, union_mask, g_train, g_test, args.epochs, args.lr,
                       REPO / "checkpoints" / "scenario1" / f"group{args.gid}", device)

    surv = count_surviving(union_mask)
    dense_rest = sum(p.numel() for n, p in net.named_parameters()
                     if n.replace("backbone.", "") not in prune_names)
    result = {"gid": args.gid, "tasks": g_tasks, "acc": accs,
              "keep_ratios": [keep_ratio(d, plan["tau"]) for d in g_diff],
              "surviving": surv, "dense_rest": dense_rest, "params": surv + dense_rest}
    (OUT / f"group_{args.gid}.json").write_text(json.dumps(result, indent=2))
    print(f"[group{args.gid}] tasks={g_tasks} acc={accs} params={(surv+dense_rest)/1e6:.2f}M", flush=True)


def cmd_report(args) -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    arts = _load_singles()
    single_acc = {a["task"]: a["acc"] for a in arts}
    single_params = sum(p.numel() for p in single_task_net().parameters())
    g_results = [json.loads((OUT / f"group_{k}.json").read_text())
                 for k in range(len(plan["groups"]))]
    dmo_acc = {t: r["acc"][t] for r in g_results for t in r["acc"]}
    total = sum(r["params"] for r in g_results)
    report = {
        "tasks": plan["tasks"],
        "single_task": {"acc": single_acc, "avg": float(np.mean(list(single_acc.values()))),
                        "params_M": round(4 * single_params / 1e6, 1)},
        "similarity": {"S_W": plan["S_W"], "S_L": plan["S_L"], "S": plan["S"]},
        "difficulty": dict(zip(plan["tasks"], plan["difficulty"])),
        "groups": [[plan["tasks"][i] for i in g] for g in plan["groups"]],
        "dmo": {"acc": dmo_acc, "avg": float(np.mean(list(dmo_acc.values()))),
                "params_M": round(total / 1e6, 2),
                "ratio": round(total / (4 * single_params), 3)},
        "paper_targets": {"single_avg": 89.95, "dmo_avg": 89.73, "dmo_params_M": 6.8, "ratio": 0.15},
        "config": {"alpha": plan["alpha"], "tau_sim": plan["tau_sim"], "tau": plan["tau"]},
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["single_task", "difficulty", "groups", "dmo"]}, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = {"--epochs": 25, "--batch-size": 64, "--lr": 0.1}

    p = sub.add_parser("single"); p.add_argument("--task", required=True, choices=SCENARIO1_TASKS)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("plan")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tau-sim", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=1.0)
    p = sub.add_parser("group"); p.add_argument("--gid", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    sub.add_parser("report")
    for name, sp in sub.choices.items():
        if name in ("single", "group"):
            for flag, default in common.items():
                sp.add_argument(flag, type=type(default), default=default)

    args = ap.parse_args()
    {"single": cmd_single, "plan": cmd_plan, "group": cmd_group, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
