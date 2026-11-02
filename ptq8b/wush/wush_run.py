"""Official WUSH (IST-DASLab/WUSH, ICML 2026, arXiv 2512.00956) quantization + our eval protocol.

Quantization = WUSH src/main.py main() quantize path verbatim (fix_seed, model load, get_data, quantize_model),
with these wrapper-only differences (identical to ptq8b mrgptq_run.py, so the two baselines share calibration + eval):
  * fineweb-edu calibration stream is read from the local jsonl dump of the SAME stream used for MR-GPTQ
    (load_dataset("HuggingFaceFW/fineweb-edu","sample-10BT",streaming=True).shuffle(seed=42,buffer_size=1000), first 6000 docs),
    so the official get_fineweb_edu() selection/cropping logic runs unchanged with HF_HUB_OFFLINE=1.
  * QuantLinear is forced to the torch (dequant -> matmul, fake-quant) reference backend (A5000 has no FP4 MMA; the
    triton kernels need a newer triton than this venv's 3.4).
  * no export, WUSH's own eval is skipped.
Eval = our protocol (wikitext_ppl from ~/hyun2/fp4_ptq/ptq8b.py; lm-eval 0.4.9 0-shot, bs 8).
Usage: python wush_run.py --tag X --out Y.json [--ppl_chunks N] [--skip_lm_eval] [--lm_limit L] -- <WUSH main.py args>
"""
import os, sys, json, math, time, shlex
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
WSRC = os.path.expanduser("~/hyun2/wush/WUSH/src")
sys.path.insert(0, WSRC)

wp = argparse.ArgumentParser()
wp.add_argument("--tag", required=True); wp.add_argument("--out", required=True)
wp.add_argument("--ppl_chunks", type=int, default=None)
wp.add_argument("--skip_lm_eval", action="store_true"); wp.add_argument("--lm_limit", type=float, default=None)
wp.add_argument("--lm_bs", type=int, default=8)
wp.add_argument("--calib_jsonl", default=os.path.expanduser("~/hyun2/mrgptq/calib/fineweb_edu_sample10BT_shuf42_buf1000_first6000.jsonl"))
argv = sys.argv[1:]
split = argv.index("--")
wargs = wp.parse_args(argv[:split])
w_argv = argv[split + 1:]
full_cmd = "python wush_run.py " + " ".join(shlex.quote(a) for a in argv)

import torch, torch.nn.functional as F
import main as WM                      # module import only: applies WUSH's TF32-off / matmul-precision settings
import data_utils as DU
from quantize_model import quantize_model
from quant_module import QuantLinear
from common_utils import fix_seed
# torch reference backend everywhere, including the block-output propagation inside quantize_model: the triton
# kernels pass enable_reflect_ftz, which triton 3.4 (this venv) does not accept, and A5000 has no FP4 MMA anyway.
QuantLinear._resolve = lambda self, role, x: "torch"

_orig_load = DU.load_dataset
class _LocalStream:
    def __init__(self, path): self.path = path
    def shuffle(self, seed=None, buffer_size=None):
        assert seed == 42 and buffer_size == 1000, (seed, buffer_size)   # dump was made with exactly these
        return self
    def __iter__(self):
        with open(self.path) as f:
            for line in f: yield json.loads(line)
def _patched_load(name, *a, **k):
    if name == "HuggingFaceFW/fineweb-edu":
        assert a == ("sample-10BT",) and k.get("streaming") and k.get("split") == "train", (a, k)
        return _LocalStream(wargs.calib_jsonl)
    return _orig_load(name, *a, **k)
DU.load_dataset = _patched_load

sys.argv = ["main.py"] + w_argv
args = WM.parse_args()
assert args.format in ("nvfp4", "mxfp4") and not args.load_path
t0 = time.time()
# ---- WUSH main() quantize path, up to (excluding) save / their eval ----
fix_seed(args.seed)
device = torch.device(torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda")
model = WM.transformers.AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, torch_dtype="auto" if args.dtype == "auto" else getattr(torch, args.dtype),
    device_map=None if args.cpu_offload_modules else device, low_cpu_mem_usage=True)
tokenizer = WM.transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
model.config.use_cache = False
model.requires_grad_(False)
print("attn_implementation:", model.config._attn_implementation, "dtype:", model.dtype, flush=True)
calibration_data = DU.get_data(dataset_name=args.dataset_name_or_path, tokenizer=tokenizer,
                               max_sequence_length=args.sequence_length, num_calibration_samples=args.num_sequences, seed=args.seed)
calib_desc = (f"{args.dataset_name_or_path}: {len(calibration_data)} seqs x {args.sequence_length} tokens "
              f"(WUSH get_data, seed {args.seed}; fineweb-edu sample-10BT streaming shuffle(seed=42,buffer=1000), "
              f"docs >= {args.sequence_length} tokens, random crop)")
print(calib_desc, flush=True)
quantize_model(model=model, calibration_data=calibration_data, quantize_format_name=args.format,
               transform_type=args.transform_type, dampen_ratio=args.dampen_ratio, activation_quantile=args.activation_quantile,
               force_rtn=not args.gptq, device=device,
               cpu_offload_modules=args.cpu_offload_modules, cpu_offload_activations=args.cpu_offload_activations)
del calibration_data
t_quant = time.time() - t0
nq = 0
for m in model.modules():
    if isinstance(m, QuantLinear):
        m.quant_backend = m.matmul_backend = "torch"; nq += 1
print("QuantLinear modules:", nq, flush=True)
model = model.to(device).eval()
torch.cuda.empty_cache()
dev = device
tok = tokenizer

# ---- our eval protocol (copied from ~/hyun2/fp4_ptq/ptq8b.py, same as mrgptq_run.py) ----
@torch.no_grad()
def wikitext_ppl(T=2048, max_chunks=None):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // T; nll = 0.0
    if max_chunks: n = min(n, max_chunks)
    for j in range(n):
        s = ids[j * T:(j + 1) * T].to(dev)[None]
        nll += F.cross_entropy(model(input_ids=s, use_cache=False).logits[0, :-1].float(), s[0, 1:], reduction="sum").item()
    return math.exp(nll / (n * (T - 1)))

mname = f"WUSH-{'GPTQ' if args.gptq else 'RTN'} (official, transform={args.transform_type})"
res = dict(model=wargs.tag, hf=args.model_name_or_path, fmt=args.format, method=mname, flags=full_cmd,
           wush_args={k: (str(v) if not isinstance(v, (int, float, str, bool, type(None), list)) else v) for k, v in vars(args).items()},
           calibration=calib_desc, ppl_chunks=wargs.ppl_chunks, quant_sec=t_quant, n_quantlinear=nq)
t1 = time.time()
res["wikitext2_ppl"] = wikitext_ppl(max_chunks=wargs.ppl_chunks)
res["ppl_sec"] = time.time() - t1
print(f"[{wargs.tag}/{args.format}] wikitext2 ppl {res['wikitext2_ppl']:.3f}  (quant {t_quant:.0f}s, ppl {res['ppl_sec']:.0f}s)", flush=True)
json.dump(res, open(wargs.out, "w"), indent=1)
if not wargs.skip_lm_eval:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    t2 = time.time()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=wargs.lm_bs)
    r = lm_eval.simple_evaluate(model=lm, tasks="hellaswag,arc_easy,arc_challenge,piqa,winogrande".split(","), num_fewshot=0,
                                limit=wargs.lm_limit, bootstrap_iters=0)
    acc = {}
    for t, v in r["results"].items():
        acc[t] = {k.split(",")[0]: v[k] for k in v if k.startswith("acc")}
    res["lm_eval"] = acc
    main = {t: (a.get("acc_norm", a.get("acc")) if t != "winogrande" else a["acc"]) for t, a in acc.items()}
    res["lm_eval_main"] = main
    res["avg5"] = sum(main.values()) / len(main)
    res["lm_eval_sec"] = time.time() - t2
    print(f"[{wargs.tag}/{args.format}] " + "  ".join(f"{t} {100*v:.2f}" for t, v in main.items()) + f"  | avg {100*res['avg5']:.2f}", flush=True)
res["runtime_sec"] = time.time() - t0
json.dump(res, open(wargs.out, "w"), indent=1)
print("DONE", res["runtime_sec"], flush=True)
