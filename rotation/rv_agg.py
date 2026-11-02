"""Aggregate rht_variants JSONL -> markdown (GEMM-output SQNR by transform x grid)."""
import json, math, sys
from collections import defaultdict
from statistics import mean
db = lambda x: -10 * math.log10(max(x, 1e-30))
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
models = list(dict.fromkeys(r["model"] for r in rows))
TFS = ["I", "H2", "H4", "H8", "H16", "H32", "H64", "H128", "DCT16", "Haar16", "PERM", "PERM+H16", "ADAPT", "SIGN16", "KLT"]
out = []
for m in models:
    agg = defaultdict(list); l46 = defaultdict(list); onl = {}
    for r in rows:
        if r["model"] != m: continue
        k = (r["gemm"], r["tf"], r["grid"])
        agg[k].append(db(r["g_nmse"]))
        if r["A_lvl46"] is not None: l46[k].append(r["A_lvl46"])
        onl[(r["gemm"], r["tf"])] = r["online"]
    out.append(f"\n## {m}  (GEMM-output SQNR dB, mean over Linears; Δ vs E2M1·I; target = UNIF·H16 (UFP4))\n")
    out.append("| transform | " + " | ".join(f"{g} E2M1 (Δ) | {g} UNIF | {g} E2M1 lvl4/6 MSE share" for g in ["fprop", "dgrad", "wgrad"]) + " |")
    out.append("|---|" + "---|" * 9)
    for tn in TFS:
        cells = []
        for g in ["fprop", "dgrad", "wgrad"]:
            if tn in ("4/6", "H16+4/6"):
                k = (g, "I" if tn == "4/6" else "H16", "E2M1-4/6")
                e = mean(agg[k]); base = mean(agg[(g, "I", "E2M1")])
                cells += [f"{e:.2f} ({e-base:+.2f})", "–", "–"]
                continue
            if (g, tn, "E2M1") not in agg:
                cells += ["n/a", "n/a", "n/a"]; continue
            e = mean(agg[(g, tn, "E2M1")]); u = mean(agg[(g, tn, "UNIF")]); base = mean(agg[(g, "I", "E2M1")])
            star = "*" if onl.get((g, tn)) else ""
            cells += [f"{e:.2f} ({e-base:+.2f}){star}", f"{u:.2f}", f"{100*mean(l46[(g, tn, 'E2M1')]):.0f}%"]
        out.append(f"| {tn} | " + " | ".join(cells) + " |")
    ufp4 = {g: mean(agg[(g, "H16", "UNIF")]) for g in ["fprop", "dgrad", "wgrad"]}
    out.append("\nUFP4 target (UNIF·H16): " + ", ".join(f"{g} {v:.2f}" for g, v in ufp4.items()) + ".  * = fitted online on the evaluated batch (token axis).")
    out.append("\n### Scale-side variants (GEMM-output SQNR dB; Δ vs E2M1·I)\n")
    QS = ["E2M1", "E2M1-4/6", "E2M1-4/6/3", "E2M1-clip", "UNIF", "UNIF-clip"]
    out.append("| GEMM | transform | " + " | ".join(QS) + " |")
    out.append("|---|---|" + "---|" * len(QS))
    for g in ["fprop", "dgrad", "wgrad"]:
        base = mean(agg[(g, "I", "E2M1")])
        for tn in ["I", "H4", "H16", "PERM"]:
            cells = []
            for q in QS:
                v = agg.get((g, tn, q))
                cells.append(f"{mean(v):.2f} ({mean(v)-base:+.2f})" if v else "–")
            out.append(f"| {g} | {tn} | " + " | ".join(cells) + " |")
print("\n".join(out))
