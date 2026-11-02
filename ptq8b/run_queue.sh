#!/bin/bash
# usage: run_queue.sh GPU "model:tag:method ..."
cd "$(dirname "$0")/.."
source ptq8b/env.sh
for job in $2; do
  IFS=: read hf tag m <<< "$job"
  [ -f ptq8b/${tag}_${m}.json ] && grep -q avg5 ptq8b/${tag}_${m}.json && continue
  CUDA_VISIBLE_DEVICES=$1 ${PY:-python} ptq8b/ptq8b.py --hf $hf --tag $tag --method $m --out ptq8b/${tag}_${m}.json --bs 8 > ptq8b/${tag}_${m}.log 2>&1
done
