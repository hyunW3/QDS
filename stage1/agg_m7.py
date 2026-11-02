"""Aggregate stage-1 runs incl. the m7 smoothing variants -> runs/stage1_summary_m7.md (old stage1_summary.md untouched).

Per run: final val (last record), Δ vs the BF16 mean over all finished bf16 seeds, train loss (mean of last 200 steps),
steady-state s/step = (t_6000 - t_100) / 5900 (excludes compile warm-up), total hours. Per arm: mean ± sample std
(ddof=1) over finished seeds.
"""
import json, re, statistics as st
from pathlib import Path

R = Path(__file__).resolve().parent / "runs"
ARMS = ["bf16", "ufp4", "e2m1_46_smooth_sr", "e2m1_46_smoothdelay_sr", "e2m1_46_smoothlearn_sr",
        "nvfp4", "e2m1_sr", "e2m1_46_sr"]


def load(d):
    recs = [json.loads(l) for l in open(d / "log.jsonl")]
    vals = [r for r in recs if "val" in r]
    if not vals:
        return None
    last = recs[-1]; t = {r["step"]: r["t"] for r in recs}
    sps = (t[6000] - t[100]) / 5900 if 6000 in t and 100 in t else last["t"] / last["step"]
    sl = [r["slearn_L"] for r in recs[-200:] if "slearn_L" in r]
    so = [(r["slearn_L"], r["slearn_L_oracle"]) for r in recs if "slearn_L_oracle" in r]
    return dict(step=last["step"], val=vals[-1]["val"], val_step=vals[-1]["step"], done=(d / "final.pt").exists() or "val" in last,   # checkpoints are not released; a log ending in a val record = finished
                train=st.mean(r["loss"] for r in recs[-200:]), sps=sps, h=last["t"] / 3600,
                slearn_L=st.mean(sl) if sl else None,
                slearn_ratio=st.mean(a / b for a, b in so) if so else None)


runs = {}
for d in sorted(R.iterdir()):
    m = re.fullmatch(r"(.+)_s(\d+)", d.name)
    if d.is_dir() and m and m.group(1) in ARMS and (d / "log.jsonl").exists():
        r = load(d)
        if r:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = r

done = lambda a: {s: r for s, r in runs.get(a, {}).items() if r["done"]}
bf = [r["val"] for r in done("bf16").values()]
bmean = st.mean(bf)
L = [f"# Stage 1 summary (m7: deployable smoothing + 3rd seeds)\n",
     f"BF16 mean final val {bmean:.4f} over {len(bf)} seeds ({', '.join(f'{v:.4f}' for v in bf)}).",
     "Δ = final val − BF16 mean. Std = sample std over seeds. s/step = steady-state (steps 100→6000) on one RTX A5000"
     " (4 runs in parallel on the node).\n",
     "## Per arm\n", "| arm | seeds | final val mean ± std | Δ mean ± std | Δ per seed | s/step | h/run |", "|---|---|---|---|---|---|---|"]
for a in ARMS:
    rs = done(a)
    if not rs:
        continue
    v = [r["val"] for r in rs.values()]; dl = [x - bmean for x in v]
    sd = lambda x: f" ± {st.stdev(x):.4f}" if len(x) > 1 else ""
    L.append(f"| {a} | {','.join(map(str, rs))} | {st.mean(v):.4f}{sd(v)} | {st.mean(dl):+.4f}{sd(dl)} | "
             f"{' / '.join(f's{s} {r['val'] - bmean:+.4f}' for s, r in rs.items())} | "
             f"{st.mean(r['sps'] for r in rs.values()):.2f} | {st.mean(r['h'] for r in rs.values()):.1f} |")
L += ["\n## Per run\n", "| run | step | final val | Δ vs BF16 mean | train loss (last 200) | s/step | h | smooth_learn L (last 200) | L_learn / L_oracle-formula (mean of diag steps) |",
      "|---|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    for s, r in sorted(runs.get(a, {}).items()):
        run = "" if r["done"] else f" (val@{r['val_step']}, running)"
        sl = "" if r["slearn_L"] is None else f"{r['slearn_L']:.5f}"
        sr = "" if r["slearn_ratio"] is None else f"{r['slearn_ratio']:.4f}"
        L.append(f"| {a}_s{s} | {r['step']} | {r['val']:.4f}{run} | {r['val'] - bmean:+.4f} | {r['train']:.4f} | "
                 f"{r['sps']:.2f} | {r['h']:.1f} | {sl} | {sr} |")
# paired (same seed => same data order and init) differences vs the oracle-smoothing arm
ref = done("e2m1_46_smooth_sr")
L += ["\n## Paired Δ vs e2m1_46_smooth_sr (same seed)\n", "| arm | per seed | mean ± std |", "|---|---|---|"]
for a in ["ufp4", "e2m1_46_smoothdelay_sr", "e2m1_46_smoothlearn_sr"]:
    rs = done(a); ss = [x for x in rs if x in ref]
    if ss:
        d = [rs[x]["val"] - ref[x]["val"] for x in ss]
        L.append(f"| {a} | {' / '.join(f's{x} {v:+.4f}' for x, v in zip(ss, d))} | {st.mean(d):+.4f}"
                 f"{f' ± {st.stdev(d):.4f}' if len(d) > 1 else ''} |")
L += ["""
## Arm definitions (m7)

- e2m1_46_smooth_sr: ORACLE SmoothQuant α=0.5 on every GEMM's reduction axis, s = sqrt(colabsmax(A)/colabsmax(B)) from
  the tensors being quantized (unchanged from earlier runs).
- e2m1_46_smoothdelay_sr: same formula, but colabsmax from the same call site (layer, Linear, FPROP/DGRAD/WGRAD; 252
  sites) in the PREVIOUS optimizer step (elementwise max over that step's 4 grad-accum micro-batches); step 0 uses the
  current stats. Eval forwards use stored stats and do not update them.
- e2m1_46_smoothlearn_sr: s = exp(v) per site, v initialised from the SmoothQuant formula at the site's first call; each
  training call uses the current v, then one Adam step (lr 1e-2, betas 0.9/0.999, own state) on
  L = ||Q(A/s)Q(Bs)^T - AB^T||^2/||AB^T||^2 over <=1024 random rows of each operand (detached, same E2M1 4/6 + SR-on-dY
  quantizer; STE through rounding, per-block absmax kept differentiable). Objective compiled with torch.compile.
  Last column above: mean over the every-100-step diagnostics of L(learned s)/L(oracle formula s) on the same subsample
  (>1 = learned s worse than the oracle per GEMM).
- All three share grid E2M1, 4/6 block scale selection, 1x16 blocks, SR on dY, RTN elsewhere, 6000 steps x 65,536 tokens.
- WGRAD reduces over tokens, so its delayed / learned s is indexed by token position within the 16x1024 micro-batch, not
  by a persistent channel.
"""]
(R / "stage1_summary_m7.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
