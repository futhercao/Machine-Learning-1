#!/bin/bash
# GPU 0 secondary: dgcnn + msg (run alongside the main g0 ssg pipeline)
set -e
cd "$(dirname "$0")"
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/bin/python

declare -a TASKS=(
  'dgcnn         2026 200 24'
  'dgcnn         42   200 24'
)

for t in "${TASKS[@]}"; do
  read m s ep bs <<< "$t"
  log="logs/${m}_s${s}.log"
  ckpt="checkpoints/${m}_s${s}.pt"
  if [ -f "$ckpt" ]; then
    echo "[g0b skip] $ckpt"; continue
  fi
  echo "[g0b] $m s=$s ep=$ep bs=$bs"
  $PY -u train.py --model $m --seed $s --epochs $ep --batch_size $bs \
    --num_workers 3 --device cuda \
    > "$log" 2>&1
  tail -3 "$log"
done
echo "[GPU0-B DONE]"
