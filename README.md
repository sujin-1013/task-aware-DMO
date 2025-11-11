# Task-Aware Dynamic Model Optimization for Multi-Task Learning
📄 IEEE Access 2023 — [DOI: 10.1109/ACCESS.2023.3339793](https://doi.org/10.1109/ACCESS.2023.3339793)

This repository contains the implementation and resources for
**Task-Aware Dynamic Model Optimization (DMO)** — a memory-efficient
multi-task learning framework that groups tasks by inter-task
similarity and allocates parameters dynamically.


---

## 🔍 Overview

Multi-task learning (MTL) often suffers from **task interference** and **inefficient resource usage**.  
**DMO (Dynamic Model Optimization)** addresses these issues by:

- Grouping tasks based on **inter-task similarity** (weight & loss correlations)
- Dynamically allocating parameters based on **task difficulty**
- Producing **lightweight subnetworks**, reducing parameters by over **80%** on average

<p align="center">
  <img src="https://raw.githubusercontent.com/sujin-1013/Task-Aware-DMO/main/assets/dmo_overview.png" width="700"/>
</p>
<sub><i>Figure: Overview of Task-Aware Dynamic Model Optimization (DMO)</i></sub>

---

## 🎯 Key Components

- **Task Similarity Measurement**
  - Pearson correlation on weights (`S_W`)
  - Pearson correlation on losses (`S_L`)
- **Task Grouping**
  - Graph-based maximal clique detection (Bron–Kerbosch)
- **Dynamic Parameter Allocation**
  - Pruning by task difficulty
  - Merging into group-specific subnetworks

---

## 🛠️ Code Structure

```
Task-Aware-DMO/
├── models/           # Backbone network, task-specific heads, pruning utilities
├── train.py          # Training logic
├── evaluate.py       # Evaluation and metrics
├── config/           # Experimental settings
├── results/          # Graphs and tables from the paper
├── assets/           # Images and visual materials for README
└── paper/            # Original PDF paper for reference
```


---

## 🧪 Datasets Used

- CIFAR-10, STL-10, MNIST, USPS
- CIFAR-100 (20-task version)
- Visual Decathlon Challenge (10 diverse tasks)

---

## 📊 Result Summary

| Dataset Group         | Param Reduction | Accuracy (Avg)   |
|----------------------|-----------------|------------------|
| CIFAR-10 + MNIST     | ~84%            | ⭤ Comparable     |
| CIFAR-100 (20 Tasks) | ~85%            | ⬆️ Improved       |
| Decathlon Challenge  | ~88%            | ⭤ Slight Drop     |

> 🔹 Up to **97% reduction** on USPS with minimal performance loss

---

### 🧪 Visual Decathlon Challenge (Summary Table)

| Method                   | Avg Accuracy (%) | # Parameters (M) | Param Ratio |
|--------------------------|------------------|------------------|-------------|
| Single-task              | **74.69**        | 101.68           | **1.00**    |
| Soft Param Sharing [7]   | 65.24            | 101.68           | 1.00        |
| Cross-stitch [8]         | 76.33            | 203.36           | 2.00        |
| NDDR [10]                | 59.76            | 252.86           | 2.49        |
| TAPS [6]                 | 59.11            | 130.50           | 1.28        |
| **DMO (Ours)**           | 74.40            | **11.48**        | **0.11**    |

---

### 📑 Visual Decathlon Table (From Paper)

<p align="center">
  <img src="https://raw.githubusercontent.com/sujin-1013/Task-Aware-DMO/main/assets/visual_decathlon_table.png" width="700"/>
</p>
<sub><i>Figure: Visual Decathlon Challenge results table (from IEEE Access paper)</i></sub>

---

## Citation

If you use this work, please cite:
```bibtex
@ARTICLE{10345562,
  author={Choi, Sujin and Jin, Hyundong and Kim, Eunwoo},
  journal={IEEE Access}, 
  title={Task-Aware Dynamic Model Optimization for Multi-Task Learning}, 
  year={2023},
  volume={11},
  number={},
  pages={137709-137717},
  keywords={Task analysis;Loss measurement;Complexity theory;Optimization;Multitasking;Weight measurement;Training;Resource management;Modeling;Multi-task learning;resource-efficient learning;model optimization},
  doi={10.1109/ACCESS.2023.3339793}}
```

---

## 🧑‍💻 Author

**Su-jin Choi (최수진)**  
Master of AI, Chung-Ang University  
📧 popo2419@naver.com  
📄 [IEEE Access Paper](https://doi.org/10.1109/ACCESS.2023.3339793)
