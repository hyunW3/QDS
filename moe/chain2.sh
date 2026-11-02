#!/bin/bash
# dense placement reruns with QDS rows (GPU0/GPU1), then OLMoE placement sharded over GPU0+1
source "$(dirname "$0")/../ptq8b/env.sh"
cd "$(dirname "$0")/.."
PY=${PY:-python}
( CUDA_VISIBLE_DEVICES=0 $PY scaling/placement_test.py --hf Qwen/Qwen2.5-0.5B --tag qwen2.5-0.5b --out hf/place_qwen2.5-0.5b.jsonl > hf/place_qwen2.5-0.5b.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY scaling/placement_test.py --hf Qwen/Qwen3-1.7B --tag qwen3-1.7b --out hf/place_qwen3-1.7b.jsonl > hf/place_qwen3-1.7b.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=1 $PY scaling/placement_test.py --hf allenai/OLMo-2-0425-1B --tag olmo2-1b --out hf/place_olmo2-1b.jsonl > hf/place_olmo2-1b.log 2>&1
wait
echo "dense done $(date +%H:%M)"
export CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=pydeps
$PY scaling/placement_test.py --hf allenai/OLMoE-1B-7B-0125 --tag olmoe-1b-7b --out moe/place_olmoe-1b-7b.jsonl --moe_expert_stride 8 --auto > moe/place_olmoe-1b-7b.log 2>&1
echo "olmoe done $(date +%H:%M)"
