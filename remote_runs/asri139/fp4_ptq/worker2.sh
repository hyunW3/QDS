#!/bin/bash
# usage: worker2.sh GPU [QUEUE]  — pops "hf tag method fmt gptq suffix [extra args...]" lines from QUEUE (default queue2.txt, flock)
cd ~/hyun2/fp4_ptq; Q=${2:-queue2.txt}
while true; do
  job=$(flock $Q.lock bash -c "l=\$(head -n 1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\"")
  [ -z "$job" ] && break
  read hf tag m fmt gq suf extra <<< "$job"
  [ "$suf" = "-" ] && suf=""
  flag=""; g=""; [ "$gq" = 1 ] && flag="--gptq" && g="_gptq"
  o=out/${tag}_${fmt}_${m}${g}${suf}
  [ -f $o.json ] && grep -q avg5 $o.json && continue
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 start $job"
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python ptq8b.py --hf $hf --tag $tag --method $m --fmt $fmt $flag --out $o.json --save_scales $o.scales.pt --bs 8 --cal dclm_val_head2000.jsonl $extra > $o.log 2>&1
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 end $job: $(grep -o 'avg [0-9.]*' $o.log | tail -n 1)"
done
