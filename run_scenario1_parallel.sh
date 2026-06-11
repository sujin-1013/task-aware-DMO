#!/bin/bash
# Scenario 1 with 2-GPU parallelism.
# Singles: mnist+usps start immediately; cifar10/stl10 start once their archives pass md5.
# Groups: trained in parallel, alternating GPUs.
set -u
cd "$(dirname "$0")"
PY=/home/sujin/dcase2023_task2_baseline_ae/.venv/bin/python
LOG=results/scenario1
mkdir -p "$LOG"

CIFAR_MD5=c58f30108f718f92721af3b95e74349a
STL_MD5=91f7769df0f17e558f3565bffb0c7dfb

wait_md5() {  # $1 file, $2 md5
  for _ in $(seq 1 360); do
    [ -f "$1" ] && [ "$(md5sum "$1" | cut -d' ' -f1)" = "$2" ] && return 0
    sleep 10
  done
  echo "[orchestrator] TIMEOUT waiting for $1" >&2; return 1
}

echo "[orchestrator] start mnist(cuda:0) usps(cuda:1)"
$PY run_scenario1.py single --task mnist --device cuda:0 > "$LOG/single_mnist.log" 2>&1 &
P_MNIST=$!
$PY run_scenario1.py single --task usps --device cuda:1 > "$LOG/single_usps.log" 2>&1 &
P_USPS=$!

wait_md5 data/cifar-10-python.tar.gz $CIFAR_MD5 || exit 1
echo "[orchestrator] cifar archive ready -> cifar10(cuda:0)"
$PY run_scenario1.py single --task cifar10 --device cuda:0 > "$LOG/single_cifar10.log" 2>&1 &
P_CIFAR=$!

wait_md5 data/stl10_binary.tar.gz $STL_MD5 || exit 1
echo "[orchestrator] stl archive ready -> stl10(cuda:1)"
$PY run_scenario1.py single --task stl10 --device cuda:1 > "$LOG/single_stl10.log" 2>&1 &
P_STL=$!

FAIL=0
for p in $P_MNIST $P_USPS $P_CIFAR $P_STL; do wait $p || FAIL=1; done
[ $FAIL -ne 0 ] && { echo "[orchestrator] a single-task run FAILED"; exit 1; }
echo "[orchestrator] singles done"

$PY run_scenario1.py plan > "$LOG/plan.log" 2>&1 || { echo "[orchestrator] plan FAILED"; exit 1; }
cat "$LOG/plan.log"

N_GROUPS=$($PY -c "import json; print(len(json.load(open('results/scenario1/plan.json'))['groups']))")
echo "[orchestrator] training $N_GROUPS groups in parallel"
PIDS=()
for k in $(seq 0 $((N_GROUPS - 1))); do
  $PY run_scenario1.py group --gid "$k" --device cuda:$((k % 2)) > "$LOG/group_$k.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
[ $FAIL -ne 0 ] && { echo "[orchestrator] a group run FAILED"; exit 1; }

$PY run_scenario1.py report
echo "[orchestrator] DONE"
