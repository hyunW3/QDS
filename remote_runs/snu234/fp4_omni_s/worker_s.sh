#!/bin/bash
# usage: worker_s.sh GPU [QUEUE]  — snu234 copy of remote_runs/info107/fp4_omni/worker_s.sh (QDS search arms, ptq8b_s.py).
# Differences: runs in ~/hyun2/fp4_omni_s, and HF_HUB_OFFLINE=0 for non-Llama jobs so they can fetch the public WikiText-2 /
# lm-eval datasets. Llama jobs stay offline: their weights were rsynced in and this node's token has no Llama access, so an
# online load 401s instead of falling back to the cache (run a non-Llama job first to cache the datasets).
cd ~/hyun2/fp4_omni_s; Q=${2:-queue_s.txt}
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
  off=0; case $hf in meta-llama/*) off=1;; esac
  HF_HUB_OFFLINE=$off CUDA_VISIBLE_DEVICES=$1 .venv/bin/python ptq8b_s.py --hf $hf --tag $tag --method $m --fmt $fmt $flag --out $o.json --save_scales $o.scales.pt --bs 8 --cal dclm_val_head2000.jsonl $extra > $o.log 2>&1
  echo "[$(date +%m-%d\ %H:%M)] GPU$1 end $job: $(grep -o 'avg [0-9.]*' $o.log | tail -n 1)"
done
