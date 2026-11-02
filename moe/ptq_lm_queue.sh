#!/bin/bash
# usage: ptq_queue.sh GPU "method method ..."   OLMoE W4A4 NVFP4 PTQ, sequential on one GPU
source "$(dirname "$0")/../ptq8b/env.sh"
cd "$(dirname "$0")/.."
for m in $2; do
  out=moe/ptq_olmoe-1b-7b_${m}_lm.json
  [ -f $out ] && continue
  CUDA_VISIBLE_DEVICES=$1 ${PY:-python} moe/moe_ptq.py --method $m --out $out --lm > moe/ptq_olmoe-1b-7b_${m}_lm.log 2>&1
  echo "$(date +%H:%M) $m: $(grep -E "Avg5|Error" moe/ptq_olmoe-1b-7b_${m}_lm.log | tail -1)"
done
