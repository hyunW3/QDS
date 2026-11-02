#!/bin/bash
source "$(dirname "$0")/../ptq8b/env.sh"
cd "$(dirname "$0")/.."
PY=${PY:-python}
( CUDA_VISIBLE_DEVICES=0 $PY scaling/scale46_test.py --hf Qwen/Qwen2.5-0.5B --tag qwen2.5-0.5b --out hf/s46_qwen2.5-0.5b.jsonl > hf/s46_qwen2.5-0.5b.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY scaling/scale46_test.py --hf Qwen/Qwen3-1.7B --tag qwen3-1.7b --out hf/s46_qwen3-1.7b.jsonl > hf/s46_qwen3-1.7b.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=1 $PY scaling/scale46_test.py --hf allenai/OLMo-2-0425-1B --tag olmo2-1b --out hf/s46_olmo2-1b.jsonl > hf/s46_olmo2-1b.log 2>&1
wait
