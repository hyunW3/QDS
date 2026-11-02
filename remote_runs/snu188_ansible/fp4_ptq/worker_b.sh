#!/bin/bash
# usage: worker_b.sh GPU [QUEUE] — pops "hf tag method gptq suffix [extra args...]" lines from QUEUE (default queue_b.txt, flock);
# re-checks right before every job that no other compute process is on the GPU (except an allowed PID list in ALLOW_PIDS)
cd "$(dirname "$0")"; Q=${2:-queue_b.txt}; PY=${PY:-.venv/bin/python}
while true; do
  UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1)
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$UUID" | awk -F', ' '{print $2}' | grep -vwE "${ALLOW_PIDS:-none}")
  if [ -n "$busy" ]; then echo "[$(date +%m-%d\ %H:%M)] GPU$1 busy (pid $busy), worker stops"; break; fi
  job=$(flock $Q.lock bash -c "l=\$(head -n 1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \"\$l\"")
  [ -z "$job" ] && break
  read hf tag m gq suf extra <<< "$job"
  [ "$suf" = "-" ] && suf=""
  flag=""; g=""; [ "$gq" = 1 ] && flag="--gptq" && g="_gptq"
  o=out/${tag}_nvfp4_${m}${g}${suf}
  [ -f $o.json ] && grep -q avg5 $o.json && continue
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 start $job"
  CUDA_VISIBLE_DEVICES=$1 $PY ptq8b_b.py --hf $hf --tag $tag --method $m --fmt nvfp4 $flag --out $o.json --bs 8 --cal ${CAL:-dclm_val_head2000.jsonl} $extra > $o.log 2>&1
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 end $job: $(grep -oE 'ppl [0-9.]*|avg [0-9.]*|Error.*' $o.log | tail -n 2 | tr '\n' ' ')"
done
