#!/bin/bash
# G128 diagnosis on snu233: GPU0 = tensor-level E4M3 probe (all layers); GPU1 = Llama-3 G128 RHT restricted to layer groups
cd ~/hyun2/fp4_ptq; export HF_HUB_OFFLINE=1
for G in 0 1; do UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G); nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID" && { echo "GPU$G busy, abort"; exit 1; }; done
( for x in "meta-llama/Meta-Llama-3-8B llama3-8b" "meta-llama/Llama-3.1-8B llama31-8b" "Qwen/Qwen3-8B qwen3-8b"; do set -- $x
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python gsize_8b.py --hf $1 --tag $2 --out out/gsize_e4m3_$2.jsonl --scale e4m3 --stride 1 --gs 16,128 2>&1 | grep "^\["; done > g128_probe.log ) &
( for r in "q_proj,k_proj,v_proj,o_proj attn" "gate_proj,up_proj,down_proj mlp" "down_proj down" "q_proj,k_proj,v_proj,gate_proj,up_proj in"; do set -- $r
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python ptq8b_b.py --hf meta-llama/Meta-Llama-3-8B --tag llama3-8b --method e2m1_h --fmt nvfp4_g128 --rht --rot_only $1 \
      --out out/llama3-8b_g128rht_only_$2.json --tasks piqa --limit 0.01 --bs 8 --cal dclm_val_head2000.jsonl 2>&1 | grep -E "ppl|Error" | sed "s/^/$2 /"; done > g128_rotonly.log ) &
wait; echo DONE >> g128_probe.log
