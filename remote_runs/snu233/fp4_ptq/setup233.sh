#!/bin/bash
# one-time setup for 8B W4A4 PTQ on snu233 (same package versions as snu188 ~/hyun2/fp4_ptq)
set -e
mkdir -p ~/hyun2/fp4_ptq/out && cd ~/hyun2/fp4_ptq
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -x .venv/bin/python ] || $UV venv -q --python 3.12 .venv
$UV pip install -q --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$UV pip install -q --python .venv/bin/python transformers==4.57.6 accelerate==1.15.0 datasets==5.0.1 lm_eval==0.4.9 "numpy<2.6"
.venv/bin/python -c "import torch,transformers,lm_eval;print('env', torch.__version__, transformers.__version__, torch.cuda.is_available())"
for m in Qwen/Qwen3-8B meta-llama/Meta-Llama-3-8B meta-llama/Llama-3.1-8B; do .venv/bin/hf download $m --exclude "original/*" > dl_$(basename $m).log 2>&1; echo "dl $m $?"; done
echo SETUP_DONE
