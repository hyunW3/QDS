"""Aggregate flip_test JSONL -> markdown."""
import json, math, sys
from collections import defaultdict
from statistics import mean
db = lambda x: -10 * math.log10(max(x, 1e-30))
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
QS = ["E2M1", "FLIP", "E2M1|FLIP", "4/6", "4/6|FLIP", "4/6|FLIP4/6", "UNIF"]
out = []
for m in dict.fromkeys(r["model"] for r in rows):
    S = defaultdict(list); SH = defaultdict(list)
    for r in rows:
        if r["model"] != m: continue
        S[(r["gemm"], r["tf"], r["q"])].append(db(r["g_nmse"]))
        SH[(r["gemm"], r["tf"], r["q"])].append(r["share_A"])
    out.append(f"\n## {m}: GEMM-output SQNR (dB), mean over Linears\n")
    out.append("| GEMM | T | " + " | ".join(QS) + " |")
    out.append("|---|---|" + "---|" * len(QS))
    for g in ["fprop", "dgrad", "wgrad"]:
        for tr in ["I", "H16"]:
            base = mean(S[(g, "I", "E2M1")])
            out.append(f"| {g} | {tr} | " + " | ".join(f"{mean(S[(g, tr, q)]):.2f} ({mean(S[(g, tr, q)])-base:+.2f})" for q in QS) + " |")
    out.append("\nBlock choice share on operand A (options in order):\n")
    for g in ["fprop", "dgrad", "wgrad"]:
        for tr in ["I", "H16"]:
            for q in ["E2M1|FLIP", "4/6|FLIP", "4/6|FLIP4/6"]:
                sh = [mean(s[i] for s in SH[(g, tr, q)]) for i in range(len(SH[(g, tr, q)][0]))]
                out.append(f"- {g} {tr} {q}: " + " / ".join(f"{100*v:.1f}%" for v in sh))
print("\n".join(out))
