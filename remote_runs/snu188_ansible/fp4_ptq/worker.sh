#!/bin/bash
# usage: worker.sh GPU   — pops lines "hf tag method fmt gptq" from queue.txt (flock) and runs them
cd ~/hyun2/fp4_ptq
while true; do
  job=$(flock queue.lock bash -c 'l=$(head -n 1 queue.txt); [ -n "$l" ] && sed -i 1d queue.txt; echo "$l"')
  [ -z "$job" ] && break
  read hf tag m fmt gq <<< "$job"
  suf=""; flag=""; [ "$gq" = 1 ] && suf="_gptq" && flag="--gptq"
  o=out/${tag}_${fmt}_${m}${suf}
  [ -f $o.json ] && grep -q avg5 $o.json && continue
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 start $job"
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python ptq8b.py --hf $hf --tag $tag --method $m --fmt $fmt $flag --out $o.json --bs 8 --cal dclm_val_head2000.jsonl > $o.log 2>&1
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 end $job: $(grep -o 'avg [0-9.]*' $o.log | tail -n 1)"
done
