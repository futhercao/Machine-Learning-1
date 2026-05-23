#!/bin/bash
# GPU 0: pointnet2_ssg (2 seeds) + pointnet2_msg
set -e
cd "$(dirname "$0")"
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/bin/python

declare -a TASKS=(
  'pointnet2_ssg 2026 200 32'
  'pointnet2_ssg 42   200 32'
  'pointnet2_msg 2026 200 24'
)

for t in "${TASKS[@]}"; do
  read m s ep bs <<< "$t"
  log="logs/${m}_s${s}.log"
  ckpt="checkpoints/${m}_s${s}.pt"
  if [ -f "$ckpt" ]; then
    echo "[g0 skip] $ckpt exists"; continue
  fi
  echo "[g0] $m s=$s ep=$ep bs=$bs"
  $PY train.py --model $m --seed $s --epochs $ep --batch_size $bs \
    --num_workers 4 --device cuda \
    > "$log" 2>&1
  tail -3 "$log"
done
echo "[GPU0 DONE]"
