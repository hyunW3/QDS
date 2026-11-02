#!/bin/bash
# OLMoE probes, sharded over GPU0+1 (one GPU OOMs at 4x1024 tokens with backward)
source "$(dirname "$0")/../ptq8b/env.sh"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
$PY rotation/dfp_test.py --hf allenai/OLMoE-1B-7B-0125 --tag olmoe-1b-7b --out moe/dfp_olmoe-1b-7b.jsonl --moe_expert_stride 8 --auto > moe/dfp_olmoe-1b-7b.log 2>&1
$PY scaling/placement_test.py --hf allenai/OLMoE-1B-7B-0125 --tag olmoe-1b-7b --out moe/place_olmoe-1b-7b.jsonl --moe_expert_stride 8 --auto > moe/place_olmoe-1b-7b.log 2>&1
