"""Aggregate scale46_test.py -> hf/s46_tables.md (mean over decoder Linears; operand A = X / dY / dY^T, B = W / W^T / X^T)."""
import json
from collections import defaultdict
from pathlib import Path
D = Path(__file__).resolve().parents[1]
out = ["# 4/6 and the top interval (`scale46_test.py`)", "",
       "frac4 = blocks choosing 4; top-interval share = share of the operand's squared error from elements with 6|x|/max in [4,6),",
       "under @6 -> under 4/6; drop = relative drop of that error; near-max = mean #elements with t >= 4 in blocks choosing 4 vs 6.",
       "GEMM: SQNR gain of 4/6 over @6 (dB) and gap to UFP4 (uniform + H16) for E2M1@6 and E2M1-4/6 (dB, negative = below UFP4).", ""]
rows_all = []
for tag in ("qwen2.5-0.5b", "olmo2-1b", "qwen3-1.7b"):
    p = D / f"hf/s46_{tag}.jsonl"
    if not p.exists(): continue
    R = [json.loads(l) for l in open(p)]
    uf = {(r["layer"], r["gemm"]): r["gemm_unif"] for r in R if r["T"] == "H16"}
    out += [f"## {tag}", "", "| GEMM | T | operand | frac4 | top-interval share @6 -> 4/6 | drop | near-max (pick 4 / keep 6) | GEMM 4/6 gain | gap to UFP4 @6 / 4/6 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for gemm in ("fprop", "dgrad", "wgrad"):
        for T in ("I", "H16"):
            rs = [r for r in R if r["gemm"] == gemm and r["T"] == T]
            if not rs: continue
            g46 = sum(r["gemm_46"] - r["gemm_6"] for r in rs) / len(rs)
            gap6 = sum(r["gemm_6"] - uf[(r["layer"], gemm)] for r in rs) / len(rs)
            gap46 = sum(r["gemm_46"] - uf[(r["layer"], gemm)] for r in rs) / len(rs)
            for op in ("A", "B"):
                m = lambda k: sum(r[op][k] for r in rs if r[op][k] == r[op][k]) / max(1, sum(1 for r in rs if r[op][k] == r[op][k]))
                tail = f"{g46:+.2f} | {gap6:+.2f} / {gap46:+.2f}" if op == "A" else " | "
                out.append(f"| {gemm} | {T} | {op} | {100*m('frac4'):.0f}% | {100*m('top_err6'):.0f}% -> {100*m('top_err46'):.0f}% | {100*m('top_drop'):.0f}% | "
                           f"{m('nmax4'):.2f} / {m('nmax6'):.2f} | {tail} |")
                rows_all.append((tag, gemm, T, op, m("frac4"), m("top_err6"), m("top_err46"), m("top_drop"), m("nmax4"), m("nmax6"), g46, gap6, gap46))
    out.append("")
(D / "hf/s46_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
