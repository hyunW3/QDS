#!/bin/bash
# usage: queue_gpu.sh <gpu> <tag:hf_id> ; runs NVFP4 (Had16), MXFP4 (Had128), MXFP4 (Had32) MR-GPTQ sequentially on one GPU
GPU=$1; TAG=${2%%:*}; HF=${2#*:}
cd ~/hyun2/mrgptq
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 HF_HUB_OFFLINE=1
COMMON="--model_name_or_path=$HF --w_bits=4 --a_bits=4 --gptq --w_observer=mse --quantization_order=activation --transform_class=hadamard --dataset_name_or_path=fineweb-edu --num_sequences=128 --sequence_length=2048 --dtype=auto --cpu_offload_activations --cpu_offload_modules --fuse_global_scale --amp"
run() { # name extra
  echo "[$(date)] start $1" >> out/queue_gpu$GPU.log
  .venv/bin/python mrgptq_run.py --tag $TAG --out out/${TAG}_$1_mrgptq.json -- $COMMON $2 > out/${TAG}_$1_mrgptq.log 2>&1
  echo "[$(date)] end $1 rc=$?" >> out/queue_gpu$GPU.log
}
run nvfp4 "--format=nvfp --w_group_size=16 --a_group_size=16 --hadamard_group_size=16"
run mxfp4 "--format=mxfp --w_group_size=32 --a_group_size=32 --hadamard_group_size=128"
run mxfp4had32 "--format=mxfp --w_group_size=32 --a_group_size=32 --hadamard_group_size=32"
