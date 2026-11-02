#!/bin/bash
# G128 RHT with the layers whose tensor-level rotation gain is negative left unrotated (massive-activation layers)
cd ~/hyun2/fp4_ptq; export HF_HUB_OFFLINE=1
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0); nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID" && { echo "GPU0 busy"; exit 1; }
for x in "meta-llama/Meta-Llama-3-8B llama3-8b 1.down_proj,31.down_proj" "meta-llama/Llama-3.1-8B llama31-8b 1.down_proj" "Qwen/Qwen3-8B qwen3-8b 6.down_proj,16.down_proj,6.up_proj,7.up_proj"; do set -- $x
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python ptq8b_b.py --hf $1 --tag $2 --method e2m1_h --fmt nvfp4_g128 --rht --rot_skip $3 \
    --out out/$2_g128rht_skipneg.json --bs 8 --cal dclm_val_head2000.jsonl 2>&1 | grep -E "ppl|avg|Error" | sed "s/^/$2 skip[$3] /"
done
echo DONE
