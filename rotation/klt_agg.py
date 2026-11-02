"""Aggregate klt_test.py -> hf/klt_tables.md: cross-coordinate vs within-block concentration and GEMM SQNR per transform."""
import json
from collections import defaultdict
from pathlib import Path
D = Path(__file__).resolve().parents[1]
K = ["top1", "top10", "cg_db", "amax_share", "top46", "blk_spread", "E2M1", "E2M1-4/6", "UNIF"]
out = ["# KLT: which energy does it concentrate? (`klt_test.py`)", "",
       "Cross-coordinate (KLT's objective): top1/top10 = share of energy in the top 1%/10% coordinates, cg = transform",
       "coding gain (dB). Within-block (what absmax block FP4 uses): amax = block-max energy share, top46 = energy in",
       "[4,6), spread = share of a token's energy in its top 10% blocks. Last three columns: GEMM SQNR (dB), delta vs I.", ""]
for tag in ("qwen2.5-0.5b", "olmo2-1b", "qwen3-1.7b"):
    p = D / f"hf/klt_{tag}.jsonl"
    if not p.exists(): continue
    R = [json.loads(l) for l in open(p)]
    out += [f"## {tag} ({len({r['layer'] for r in R})} Linears)", "",
            "| GEMM | T | top1 | top10 | cg dB | amax | top46 | spread | E2M1 | E2M1-4/6 | uniform |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for gemm in ("fprop", "dgrad"):
        acc = defaultdict(lambda: defaultdict(list))
        for r in R:
            if r["gemm"] == gemm:
                for k in K: acc[r["T"]][k].append(r[k])
        m = {t: {k: sum(v) / len(v) for k, v in d.items()} for t, d in acc.items()}
        for t in ("I", "H16", "KLT", "BKLT"):
            if t not in m: continue
            a = m[t]; b = m["I"]
            out.append(f"| {gemm} | {t} | {100*a['top1']:.1f}% | {100*a['top10']:.1f}% | {a['cg_db']:.2f} | {100*a['amax_share']:.1f}% | "
                       f"{100*a['top46']:.1f}% | {100*a['blk_spread']:.1f}% | {a['E2M1']:.2f} ({a['E2M1']-b['E2M1']:+.2f}) | "
                       f"{a['E2M1-4/6']:.2f} ({a['E2M1-4/6']-b['E2M1-4/6']:+.2f}) | {a['UNIF']:.2f} ({a['UNIF']-b['UNIF']:+.2f}) |")
    out.append("")
(D / "hf/klt_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
