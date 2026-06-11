#!/bin/bash
# Scenario 1 on two GPUs: single-task runs and group runs execute in parallel.
# Usage: ./run_scenario1_parallel.sh [python]
set -euo pipefail
cd "$(dirname "$0")"

PY=${1:-python3}
LOG=results/scenario1
mkdir -p "$LOG"

echo "[1/4] single-task training (4 tasks, 2 per GPU)"
pids=()
$PY run_scenario1.py single --task cifar10 --device cuda:0 > "$LOG/single_cifar10.log" 2>&1 & pids+=($!)
$PY run_scenario1.py single --task stl10   --device cuda:1 > "$LOG/single_stl10.log"   2>&1 & pids+=($!)
$PY run_scenario1.py single --task usps    --device cuda:0 > "$LOG/single_usps.log"    2>&1 & pids+=($!)
$PY run_scenario1.py single --task mnist   --device cuda:1 > "$LOG/single_mnist.log"   2>&1 & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

echo "[2/4] plan: similarity, difficulty, grouping"
$PY run_scenario1.py plan | tee "$LOG/plan.log"

echo "[3/4] group training (one GPU per group)"
N_GROUPS=$($PY -c "import json; print(len(json.load(open('$LOG/plan.json'))['groups']))")
pids=()
for k in $(seq 0 $((N_GROUPS - 1))); do
  $PY run_scenario1.py group --gid "$k" --device cuda:$((k % 2)) > "$LOG/group_$k.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "[4/4] report"
$PY run_scenario1.py report
