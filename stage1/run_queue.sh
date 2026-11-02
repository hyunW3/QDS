#!/bin/bash
# usage: run_queue.sh GPU "arm:seed arm:seed ..."   (sequential runs on one GPU)
source "$(dirname "$0")/../ptq8b/env.sh"
cd "$(dirname "$0")"
GPU=$1; shift
for spec in $1; do
  arm=${spec%%:*}; seed=${spec#*:}; out=runs/${arm}_s${seed}
  if [ -f $out/final.pt ]; then echo "skip $out"; continue; fi
  echo "[$(date +%m-%d\ %H:%M)] GPU$GPU start $arm seed $seed"
  CUDA_VISIBLE_DEVICES=$GPU ${PY:-python} train.py --arm $arm --seed $seed --out $out --steps 6000 > $out.log 2>&1
  echo "[$(date +%m-%d\ %H:%M)] GPU$GPU end $arm seed $seed: $(grep -v -i 'pynvml\|warn' $out.log | tail -1)"
done
