#!/bin/bash
source "$(dirname "$0")/../ptq8b/env.sh"
cd "$(dirname "$0")/.."
PY=${PY:-python}
CUDA_VISIBLE_DEVICES="" $PY rotation/klt_test.py --hf Qwen/Qwen2.5-0.5B --tag qwen2.5-0.5b --out hf/klt_qwen2.5-0.5b.jsonl --layers 0,4,8,12,16,20 > hf/klt_qwen2.5-0.5b.log 2>&1 &
CUDA_VISIBLE_DEVICES="" $PY rotation/klt_test.py --hf allenai/OLMo-2-0425-1B --tag olmo2-1b --out hf/klt_olmo2-1b.jsonl --layers 0,3,6,9,12,15 > hf/klt_olmo2-1b.log 2>&1 &
CUDA_VISIBLE_DEVICES="" $PY rotation/klt_test.py --hf Qwen/Qwen3-1.7B --tag qwen3-1.7b --out hf/klt_qwen3-1.7b.jsonl --layers 0,5,10,15,20,25 > hf/klt_qwen3-1.7b.log 2>&1 &
wait
