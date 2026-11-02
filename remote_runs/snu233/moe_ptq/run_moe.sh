#!/bin/bash
# usage: run_moe.sh GPU METHOD [--rht]   waits for setup, re-checks the GPU is free, then runs OLMoE W4A4 PTQ + Avg5
cd ~/hyun2/moe_ptq
GPU=$1; M=$2; shift 2; EXTRA="$*"; SUF=""; [[ "$EXTRA" == *--rht* ]] && SUF="_rht"
until grep -q SETUP_DONE setup.log; do sleep 30; done
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then
  echo "$(date) GPU$GPU busy (another process), not starting $M$SUF"; exit 1
fi
o=out/ptq_olmoe-1b-7b_${M}${SUF}_lm
echo "$(date) GPU$GPU start $M$SUF"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python moe_ptq.py --method $M $EXTRA --cal dclm_val_head2000.jsonl --out $o.json --lm > $o.log 2>&1
echo "$(date) GPU$GPU end $M$SUF: $(grep -E 'ppl|Avg5|Error' $o.log | tail -2 | tr '\n' ' ')"
