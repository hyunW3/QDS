"""Pull 8B PTQ results from info107 / info103 / asri139 and print one table per model (ppl, Avg5)."""
import json, glob, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
if "--nosync" not in sys.argv:
    for h in ["info107", "info103", "asri139"]:
        (R / h).mkdir(parents=True, exist_ok=True)       # per node: the same file name can exist on two nodes (a failed and a finished copy)
        subprocess.run(f"timeout 120 rsync -a --include='*.json' --exclude='*' {h}:hyun2/fp4_ptq/out/ {R}/{h}/", shell=True, stdin=subprocess.DEVNULL)
rows = {}
for f in glob.glob(str(R / "*" / "*.json")):
    r = json.load(open(f))
    if "wikitext2_ppl" not in r: continue
    if r.get("gptq") and r["args"].get("ngptq") != 128: continue      # 8k-token GPTQ runs were overfit (rank-deficient Hessian); kept in results_gptq8k
    fmt = r.get("fmt", "nvfp4"); key = (r["model"], fmt, r["method"].replace("e2m1_h16", "e2m1_h") + ("+gptq" if r.get("gptq") else ""))
    if key in rows and "avg5" in rows[key] and "avg5" not in r: continue     # prefer the finished copy
    rows[key] = r
ORDER = ["bf16", "e2m1", "e2m1_46", "e2m1_h", "ufp4", "sq_46", "diag", "diag_46", "diag_h"]
TASKS = ["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"]
def main_acc(r):
    a = r.get("lm_eval", {})
    return {t: (a[t].get("acc_norm", a[t].get("acc")) if t != "winogrande" else a[t]["acc"]) for t in a}
for model in sorted({k[0] for k in rows}):
    print(f"\n## {model}")
    print("| fmt | method | wiki2 ppl | " + " | ".join(t[:6] for t in TASKS) + " | Avg5 |")
    print("|---|---|---|" + "---|" * (len(TASKS) + 1))
    ks = sorted([k for k in rows if k[0] == model], key=lambda k: (["nvfp4", "nvfp4_g32", "mxfp4"].index(k[1]) if k[1] in ["nvfp4", "nvfp4_g32", "mxfp4"] else 9, "gptq" in k[2], ORDER.index(k[2].split("+")[0]) if k[2].split("+")[0] in ORDER else 99))
    for k in ks:
        r = rows[k]; m = main_acc(r)
        print(f"| {k[1]} | {k[2]} | {r['wikitext2_ppl']:.3f} | " + " | ".join(f"{100*m[t]:.1f}" if t in m else "–" for t in TASKS) + f" | {100*r['avg5']:.2f} |" if "avg5" in r else f"| {k[1]} | {k[2]} | {r['wikitext2_ppl']:.3f} | (eval running) |")

# official MR-GPTQ (FP-Quant) runs, pulled into ptq8b/mrgptq/ — added as method "mrgptq+gptq" (fmt nvfp4 / mxfp4)
MR = Path(__file__).resolve().parent / "mrgptq"
for f in glob.glob(str(MR / "*_mrgptq.json")):
    r = json.load(open(f))
    if "avg5" not in r or r.get("avg5") is None: continue
    tag = r["model"].replace("llama3.1-8b", "llama31-8b"); fmt = {"nvfp": "nvfp4", "mxfp": "mxfp4"}[r["fmt"]]
    meth = "mrgptq_h32+gptq" if "had32" in f else "mrgptq+gptq"
    r2 = dict(r, lm_eval={t: {"acc_norm": v} if t != "winogrande" else {"acc": v} for t, v in r["lm_eval_main"].items()})
    rows[(tag, fmt, meth)] = r2

# machine-readable dump for make_figs.py / report: {model: {fmt: {method(+gptq): {ppl, avg5, tasks}}}}
dump = {}
for (model, fmt, meth), r in rows.items():
    dump.setdefault(model, {}).setdefault(fmt, {})[meth] = dict(ppl=r["wikitext2_ppl"], avg5=r.get("avg5"), tasks=main_acc(r))
json.dump(dump, open(Path(__file__).resolve().parent / "results_rows.json", "w"), indent=1)
