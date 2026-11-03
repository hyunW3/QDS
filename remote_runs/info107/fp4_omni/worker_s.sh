#!/bin/bash
# usage: worker_s.sh GPU [QUEUE]  — QDS search arms (ptq8b_s.py); pops "hf tag method fmt gptq suffix [extra args...]" lines
# from QUEUE (default queue_s.txt, flock). Before each job it re-checks the GPU: if another compute process holds it, the job
# goes back to the head of the queue and this worker stops (never co-tenant a GPU).
cd ~/hyun2/fp4_omni; Q=${2:-queue_s.txt}
while true; do
  job=$(flock $Q.lock bash -c "l=\$(head -n 1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\"")
  [ -z "$job" ] && break
  read hf tag m fmt gq suf extra <<< "$job"
  [ "$suf" = "-" ] && suf=""
  flag=""; g=""; [ "$gq" = 1 ] && flag="--gptq" && g="_gptq"
  o=out/${tag}_${fmt}_${m}${g}${suf}
  [ -f $o.json ] && grep -q avg5 $o.json && continue
  busy=$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$busy" != 0 ]; then
    echo "[$(date +%m-%d\ %H:%M)] GPU$1 busy ($busy procs), requeue and stop: $job"
    flock $Q.lock bash -c "sed -i '1i $job' $Q"; break
  fi
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 start $job"
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python ptq8b_s.py --hf $hf --tag $tag --method $m --fmt $fmt $flag --out $o.json --save_scales $o.scales.pt --bs 8 --cal dclm_val_head2000.jsonl $extra > $o.log 2>&1
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 end $job: $(grep -o 'avg [0-9.]*' $o.log | tail -n 1)"
done
