# Put the pip-installed NVIDIA libraries of the venv on LD_LIBRARY_PATH (torch 2.7 cu128 wheels; set VENV to your venv).
for d in "${VENV:-.venv}"/lib/python3*/site-packages/nvidia/*/lib; do [ -d "$d" ] && LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"; done
export LD_LIBRARY_PATH
