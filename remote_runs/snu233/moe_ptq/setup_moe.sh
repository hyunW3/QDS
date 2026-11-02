#!/bin/bash
# one-time setup for OLMoE W4A4 PTQ (moe_ptq.py needs transformers 5 for the 3D-expert MoE layout)
set -e
mkdir -p ~/hyun2/moe_ptq/out && cd ~/hyun2/moe_ptq
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -x .venv/bin/python ] || $UV venv -q --python 3.12 .venv
$UV pip install -q --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$UV pip install -q --python .venv/bin/python transformers==5.14.1 accelerate datasets lm_eval==0.4.12 numpy
.venv/bin/python -c "import torch,transformers,lm_eval;print('env', torch.__version__, transformers.__version__, torch.cuda.is_available())"
.venv/bin/hf download allenai/OLMoE-1B-7B-0125 > dl.log 2>&1
.venv/bin/python -c "from datasets import load_dataset; load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test'); print('wikitext ok')"
echo SETUP_DONE
