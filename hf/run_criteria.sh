#!/bin/bash
source "$(dirname "$0")/../ptq8b/env.sh"
cd "$(dirname "$0")/.."
PY=${PY:-python}
for spec in Qwen/Qwen2.5-0.5B:qwen2.5-0.5b allenai/OLMo-2-0425-1B:olmo2-1b Qwen/Qwen3-1.7B:qwen3-1.7b; do
  hf=${spec%%:*}; tag=${spec##*:}
  CUDA_VISIBLE_DEVICES="" $PY scaling/criteria_stats.py --cpu --hf $hf --tag $tag --out hf/crit_$tag.jsonl > hf/crit_$tag.log 2>&1
  echo "$tag $(wc -l < hf/crit_$tag.jsonl) $(grep -E 'loss|Error' hf/crit_$tag.log | tail -1)"
done
