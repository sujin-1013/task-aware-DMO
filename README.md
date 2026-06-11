# Task-Aware Dynamic Model Optimization for Multi-Task Learning

[![IEEE Access](https://img.shields.io/badge/IEEE%20Access-2023-00629B)](https://doi.org/10.1109/ACCESS.2023.3339793)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2023.3339793-blue)](https://doi.org/10.1109/ACCESS.2023.3339793)

**Dynamic Model Optimization (DMO)** is a memory-efficient multi-task learning
framework. It measures inter-task similarity from weight and loss statistics,
groups similar tasks by maximal-clique detection, and allocates parameters to
each group with difficulty-aware pruning — producing lightweight group-specific
subnetworks that cut parameters by over 80% on average at comparable accuracy.

<p align="center">
  <img src="assets/dmo_overview.png" width="700"/>
</p>

## Highlights

- **Task grouping from similarity** — Pearson correlation on task weights (S_W) and
  loss curves (S_L), combined as S = αS_W + (1−α)S_L; groups found as maximal
  cliques (Bron–Kerbosch) on the similarity graph.
- **Difficulty-aware parameter allocation** — task difficulty d is the fraction of
  hard samples (loss above the cross-task mean); each task is magnitude-pruned at
  ratio 1/exp(d/τ), so easy tasks keep few weights and hard tasks keep more.
- **Group-specific subnetworks** — surviving parameters are merged within each
  group (higher-difficulty tasks take precedence on overlaps) and trained jointly.
- **84% fewer parameters** on the four-task scenario and **0.11× parameters** on
  Visual Decathlon at accuracy comparable to single-task learning.

## Results

**Scenario 1 — four tasks** (CIFAR-10, STL-10, USPS, MNIST; ResNet-18, paper Table 1):

| Method | CIFAR-10 | STL-10 | USPS | MNIST | Avg | #P (Ratio) |
|---|---:|---:|---:|---:|---:|---:|
| Single-task | 92.10 | 71.87 | 96.51 | 99.32 | 89.95 | 44.8M (1.00) |
| Hard parameter sharing | 79.72 | 70.82 | 51.66 | 98.03 | 75.06 | 11.2M (0.25) |
| MTAN | 81.92 | 72.21 | 95.06 | 94.53 | 85.93 | 19.1M (0.42) |
| Soft parameter sharing | 92.30 | 71.83 | 96.86 | 99.34 | 90.08 | 44.8M (1.00) |
| Cross-stitch | 89.79 | 68.60 | 96.71 | 99.22 | 88.58 | 56.0M (1.25) |
| TAPS | 79.49 | 60.30 | 94.72 | 98.99 | 83.37 | 25.0M (0.55) |
| **DMO (ours)** | 91.50 | 71.56 | 96.56 | 99.31 | **89.73** | **6.8M (0.15)** |

**Scenario 2 — 20 tasks from CIFAR-100** (superclass tasks, ResNet-18, paper Table 2):

| Method | Avg | #P (Ratio) |
|---|---:|---:|
| Single-task | 80.47 | 223.5M (1.00) |
| Hard parameter sharing | 75.48 | 11.1M (0.04) |
| MTAN | 82.88 | 35.9M (0.16) |
| Soft parameter sharing | 82.16 | 223.5M (1.00) |
| TAPS | 69.84 | 130.5M (0.58) |
| **DMO (ours)** | **83.10** | **33.5M (0.14)** |

**Visual Decathlon Challenge** (10 tasks, ImageNet-pretrained ResNet-18, paper Table 3):

| Method | Avg | #P (Ratio) |
|---|---:|---:|
| Single-task | 74.69 | 101.68M (1.00) |
| MTAN | 77.25 | 19.64M (0.19) |
| Cross-stitch | 76.33 | 203.36M (2.00) |
| TAPS | 59.11 | 65.32M (0.64) |
| **DMO (ours)** | 74.40 | **11.48M (0.11)** |

Full per-task tables and analysis are in the [paper](paper/).

## Reproduction (Scenario 1)

This repository provides a PyTorch implementation of the full DMO pipeline for
Scenario 1 (single-task training → similarity & difficulty → clique grouping →
difficulty-aware pruning & merging → group training).

```bash
pip install torch torchvision networkx

# end-to-end on two GPUs (singles and groups run in parallel)
./run_scenario1_parallel.sh

# or stage by stage
python run_scenario1.py single --task cifar10 --device cuda:0
python run_scenario1.py single --task stl10 --device cuda:1   # ... usps, mnist
python run_scenario1.py plan
python run_scenario1.py group --gid 0 --device cuda:0
python run_scenario1.py report                                 # results/scenario1/metrics.json
```

Datasets (CIFAR-10, STL-10, USPS, MNIST) download automatically to `data/`.
Hyperparameters follow the paper: ResNet-18, SGD (momentum 0.9, lr 0.1), 25
epochs, batch 64, images resized to 72×72. α and τ_sim are not specified in the
paper; this implementation uses α = 0.5, τ_sim = 0.45 (the value at which the
paper's USPS+MNIST grouping emerges), and pruning temperature τ = 1.0.

**Reproduced results** (RTX A6000 ×2, this implementation):

| | Avg | #P (Ratio) |
|---|---:|---:|
| Paper DMO (Table 1) | 89.73 | 6.8M (0.15) |
| Reproduced — no grouping (τ_sim = 0.5) | 89.33 | 6.25M (0.14) |
| Reproduced — paper grouping [USPS+MNIST], [CIFAR-10+STL-10] | **90.99** | **5.74M (0.13)** |

With the paper's grouping, joint training lifts STL-10 from 59.1 (our
single-task) to 81.6, and the overall average exceeds single-task learning
(86.7) by +4.3 points at 0.13× parameters — confirming the paper's claim of
comparable-or-better accuracy at ~0.15× parameters. Full logs and per-task
numbers: `results/scenario1/metrics.json`.

## Repository structure

```
dmo/
  datasets.py    Scenario-1 datasets (72x72, grayscale -> 3ch)
  model.py       ResNet-18 backbone, group model with per-task heads
  metrics.py     similarity (S_W, S_L), difficulty, clique grouping
  pruning.py     difficulty-aware magnitude pruning and group merging
  engine.py      train/eval loops (per-epoch checkpoints)
run_scenario1.py          staged pipeline (single / plan / group / report)
run_scenario1_parallel.sh two-GPU orchestration
paper/                    published IEEE Access paper (PDF)
```

## Citation

```bibtex
@article{choi2023dmo,
  author  = {Choi, Sujin and Jin, Hyundong and Kim, Eunwoo},
  journal = {IEEE Access},
  title   = {Task-Aware Dynamic Model Optimization for Multi-Task Learning},
  year    = {2023},
  volume  = {11},
  pages   = {137709--137717},
  doi     = {10.1109/ACCESS.2023.3339793}
}
```

## License & contact

Code released under the [MIT License](LICENSE).
Sujin Choi — popo2419@naver.com
