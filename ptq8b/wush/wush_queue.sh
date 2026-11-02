#!/bin/bash
# usage: wush_queue.sh GPU "tag:hf:fmt:gptq ..."   (runs jobs sequentially on one GPU; skips finished ones)
cd ~/hyun2/wush
for job in $2; do
  IFS=: read tag hf fmt g <<< "$job"
  m=$([ "$g" = 1 ] && echo gptq || echo rtn)
  out=out/${tag}_${fmt}_wush_${m}.json
  [ -f $out ] && grep -q avg5 $out && continue
  CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 OMP_NUM_THREADS=16 \
    ~/hyun2/mrgptq/.venv/bin/python wush_run.py --tag $tag --out $out -- --model_name_or_path=$hf --format=$fmt \
    --gptq=$g --transform_type=wush --num_sequences=128 --sequence_length=2048 --seed=42 > logs/${tag}_${fmt}_wush_${m}.log 2>&1
done
