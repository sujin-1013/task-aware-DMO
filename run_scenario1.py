#!/usr/bin/env python3
"""Scenario 1: CIFAR-10, STL-10, USPS, MNIST with ResNet-18 (paper Table 1).

The pipeline is staged so the stages can run in parallel on different GPUs
(see run_scenario1_parallel.sh for a two-GPU orchestration):

  single --task T     train one task-specific network
  plan                similarity + difficulty + maximal-clique grouping
  group --gid K       prune, merge, and train the K-th group model
  report              aggregate everything into results/scenario1/metrics.json

Targets (paper): single-task avg 89.95 @ 44.8M; DMO avg 89.73 @ 6.8M (0.15x).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dmo.datasets import SCENARIO1_TASKS
from dmo.pipeline import Config, run_group, run_plan, run_report, run_single

REPO = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    single = sub.add_parser("single", help="train one task-specific network")
    single.add_argument("--task", required=True, choices=SCENARIO1_TASKS)

    plan = sub.add_parser("plan", help="similarity, difficulty, and task grouping")
    plan.add_argument("--alpha", type=float, default=Config.alpha,
                      help="weight vs loss similarity balance (Eq. 1)")
    plan.add_argument("--tau-sim", type=float, default=Config.tau_sim,
                      help="similarity edge threshold (Eq. 4)")
    plan.add_argument("--tau", type=float, default=Config.tau,
                      help="pruning temperature (Sec. III-E.1)")

    group = sub.add_parser("group", help="train one group-specific model from the plan")
    group.add_argument("--gid", type=int, required=True)

    sub.add_parser("report", help="aggregate results into metrics.json")

    for p in (single, group):
        p.add_argument("--device", default=Config.device)
        p.add_argument("--epochs", type=int, default=Config.epochs)
        p.add_argument("--batch-size", type=int, default=Config.batch_size)
        p.add_argument("--lr", type=float, default=Config.lr)
        p.add_argument("--seed", type=int, default=Config.seed)

    args = parser.parse_args()
    cfg = Config(repo=REPO, **{k: v for k, v in vars(args).items()
                               if k in Config.__dataclass_fields__})

    if args.stage == "single":
        run_single(cfg, args.task)
    elif args.stage == "plan":
        run_plan(cfg)
    elif args.stage == "group":
        run_group(cfg, args.gid)
    else:
        run_report(cfg)


if __name__ == "__main__":
    main()
