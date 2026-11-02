"""Aggregate MoE dfp_test.py records (moe/dfp_*.jsonl) -> moe/dfp_tables.md, split into routed experts vs attention.

Same quantities as dfp_agg.py: rotation gain in dB (+ = H_G helps) = S (Delta_FP) + nu (placement) + c (covariance),
mean over tapped Linears; plus the share of GEMM error from outlier blocks (top-1% channel) under I / H / oH.
"""
import json, math
from collections import defaultdict
from pathlib import Path
D = Path(__file__).resolve().parents[1]
FMTS = ["E2M1", "MXE2M1", "INT4", "FP8"]
db = lambda x: 10 * math.log10(x)
GROUPS = [("experts", lambda l: ".experts." in l), ("attention", lambda l: ".self_attn." in l),
          ("shared/dense MLP", lambda l: ".mlp." in l and ".experts." not in l)]
out = ["# Delta_FP vs level placement on MoE models (`dfp_test.py --moe_expert_stride`)", "",
       "Rotation gain (dB, + = H_G helps) = S (Delta_FP) + nu (placement) + c (covariance); mean over tapped Linears.",
       "Routed experts: every 8th expert of every layer, gate_up_proj and down_proj; WGRAD uses the first multiple of 128 tokens.", ""]
for p in sorted((D / "moe").glob("dfp_*.jsonl")):
    R = [json.loads(l) for l in open(p)]
    idx = {(r["layer"], r["gemm"], r["G"], r["T"]): r for r in R}
    out += [f"## {p.stem.replace('dfp_', '')}", "",
            "| group | GEMM | G | n | fmt | actual | S (Delta_FP) | nu (placement) | c | Delta_FP I->H | oH actual | share_out I / H / oH |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for gname, pred in GROUPS:
        for gemm in ("fprop", "dgrad", "wgrad"):
            for G in (16, 32, 128):
                ks = [k for k in idx if k[1] == gemm and k[2] == G and k[3] == "I" and pred(k[0])
                      and (k[0], gemm, G, "H") in idx and (k[0], gemm, G, "oH") in idx]
                if not ks: continue
                for f in FMTS:
                    a = defaultdict(list)
                    for k in ks:
                        r, q, o = idx[k], idx[(k[0], gemm, G, "H")], idx[(k[0], gemm, G, "oH")]
                        a["act"].append(db(r[f]["act"] / q[f]["act"])); a["S"].append(db(r["S"] / q["S"]))
                        a["nu"].append(db(r[f]["nu"] / q[f]["nu"])); a["c"].append(db(r[f]["c"] / q[f]["c"]))
                        a["d0"].append(r["dfp"]); a["d1"].append(q["dfp"]); a["oact"].append(db(r[f]["act"] / o[f]["act"]))
                        a["sI"].append(r[f]["share_out"]); a["sH"].append(q[f]["share_out"]); a["sO"].append(o[f]["share_out"])
                    m = lambda x: sum(a[x]) / len(a[x])
                    out.append(f"| {gname} | {gemm} | {G} | {len(ks)} | {f} | {m('act'):+.2f} | {m('S'):+.2f} | {m('nu'):+.2f} | {m('c'):+.2f} | "
                               f"{m('d0'):.3f} -> {m('d1'):.3f} | {m('oact'):+.2f} | {100*m('sI'):.0f} / {100*m('sH'):.0f} / {100*m('sO'):.0f}% |")
    out.append("")
(D / "moe/dfp_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
