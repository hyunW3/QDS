#!/bin/bash
# Launch diag_prop.py: the three no-clamp fmt-STE runs + qdsres Llama-3.1 as a healthy control. One job per GPU (4-7);
# each GPU is re-checked for other compute processes right before its launch.
cd ~/hyun2/fp4_omni; mkdir -p diag
launch() {   # GPU HF TAG SUFFIX
  if [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then echo "GPU$1 busy, skip $3$4"; return; fi
  CUDA_VISIBLE_DEVICES=$1 nohup .venv/bin/python diag_prop.py --hf $2 --scales out/$3_nvfp4_$4.scales.pt --out diag/prop_$3_$4.json > diag/prop_$3_$4.log 2>&1 < /dev/null &
  echo "GPU$1 started $3 $4"
}
launch 4 meta-llama/Meta-Llama-3-8B llama3-8b diagomni_46_wt_fmt_noclamp
launch 5 meta-llama/Llama-3.1-8B llama31-8b diagomni_46_wt_fmt_noclamp
launch 6 Qwen/Qwen3-8B qwen3-8b diagomni_46_wt_fmt_noclamp
launch 7 meta-llama/Llama-3.1-8B llama31-8b qdsres_46_wt
