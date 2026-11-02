#!/bin/bash
# usage: rq.sh GPU "hf|tag|method ..."   (runs sequentially on one GPU, skips finished)
cd ~/hyun2/fp4_ptq
for job in $2; do
  IFS='|' read hf tag m <<< "$job"
  [ -f out/${tag}_${m}.json ] && grep -q avg5 out/${tag}_${m}.json && continue
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python ptq8b.py --hf $hf --tag $tag --method $m --out out/${tag}_${m}.json --bs 8 --cal dclm_val_head2000.jsonl > out/${tag}_${m}.log 2>&1
done
