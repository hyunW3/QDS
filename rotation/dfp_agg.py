"""Aggregate dfp_test.py records -> hf/dfp_tables.md.

Per model x GEMM x format x G: mean over Linears of the rotation gain in dB (10 log10(X_I / X_T), positive = rotation helps)
for the actual GEMM error and its exact split pred = S * nu * c:
  S  (Delta_FP, joint weight-activation concentration; the only term under the O&P IEIN model)
  nu (single-operand tensor NMSE: level placement vs block-normalized values)
  c  (covariance of per-channel NMSE with per-channel joint energy)
plus the share of GEMM error coming from blocks that hold a top-1% outlier channel (I / H / oH).
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
D = Path(__file__).resolve().parents[1]
TAGS = ["qwen2.5-0.5b", "olmo2-1b", "qwen3-1.7b"]
FMTS = ["E2M1", "MXE2M1", "INT4", "FP8"]
db = lambda x: 10 * math.log10(x)

out = ["# Delta_FP vs level placement: why a block Hadamard hurts E2M1 (`dfp_test.py`)", "",
       "Rotation gain in dB = 10 log10(err_I / err_T), positive = the transform helps. pred = S * nu * c is the independent-error",
       "GEMM error (matches the measured GEMM error within ~0.1 dB); gain_pred = gain_S + gain_nu + gain_c exactly.",
       "S = Delta_FP term (joint concentration, Ordentlich & Polyanskiy 2026a); nu = single-operand tensor NMSE; c = covariance.",
       "Mean over all decoder Linears; one 4 x 1024-token DCLM-val batch.", ""]
for tag in TAGS:
    p = D / f"hf/dfp_{tag}.jsonl"
    if not p.exists(): continue
    R = [json.loads(l) for l in open(p)]
    idx = {(r["layer"], r["gemm"], r["G"], r["T"]): r for r in R}
    out += [f"## {tag} ({len({r['layer'] for r in R})} Linears)", ""]
    for T in ("H", "oH"):
        out += [f"### {T} vs I" + ("  (H_G on every block)" if T == "H" else "  (H_G only on blocks holding a top-1% outlier channel)"), "",
                "| GEMM | G | fmt | actual | pred | S (Delta_FP) | nu (placement) | c | mean Delta_FP I -> T |", "|---|---|---|---|---|---|---|---|---|"]
        for gemm in ("fprop", "dgrad", "wgrad"):
            for G in (16, 32, 128):
                acc = defaultdict(list)
                for (l, gm, g, t), r in idx.items():
                    if gm != gemm or g != G or t != "I" or (l, gm, g, T) not in idx: continue
                    q = idx[(l, gm, g, T)]
                    acc["dfp0"].append(r["dfp"]); acc["dfp1"].append(q["dfp"])
                    for f in FMTS:
                        a, b = r[f], q[f]
                        acc[f + "act"].append(db(a["act"] / b["act"])); acc[f + "pred"].append(db(a["pred"] / b["pred"]))
                        acc[f + "S"].append(db(r["S"] / q["S"])); acc[f + "nu"].append(db(a["nu"] / b["nu"])); acc[f + "c"].append(db(a["c"] / b["c"]))
                if not acc: continue
                m = lambda k: sum(acc[k]) / len(acc[k])
                for f in FMTS:
                    out.append(f"| {gemm} | {G} | {f} | {m(f+'act'):+.2f} | {m(f+'pred'):+.2f} | {m(f+'S'):+.2f} | {m(f+'nu'):+.2f} | {m(f+'c'):+.2f} | {m('dfp0'):.3f} -> {m('dfp1'):.3f} |")
        out.append("")
    out += ["### Share of GEMM error from outlier blocks (blocks holding a top-1% channel)", "",
            "| GEMM | G | blocks flagged | energy share (I) | fmt | share I | share H | share oH | err_out H/I (dB) | err_rest H/I (dB) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for gemm in ("fprop", "dgrad", "wgrad"):
        for G in (16, 32, 128):
            rows = [(idx[k], idx.get((k[0], k[1], k[2], "H")), idx.get((k[0], k[1], k[2], "oH"))) for k in idx if k[1] == gemm and k[2] == G and k[3] == "I"]
            rows = [x for x in rows if x[1] and x[2]]
            if not rows: continue
            n = len(rows)
            fb = sum(x[0]["frac_oblk"] for x in rows) / n; es = sum(x[0]["Eshare_out"] for x in rows) / n
            for f in FMTS:
                sI = sum(x[0][f]["share_out"] for x in rows) / n; sH = sum(x[1][f]["share_out"] for x in rows) / n; so = sum(x[2][f]["share_out"] for x in rows) / n
                go = [db(x[1][f]["pred_out"] / x[0][f]["pred_out"]) for x in rows if x[0][f]["pred_out"] > 0]
                gr = [db(x[1][f]["pred_rest"] / x[0][f]["pred_rest"]) for x in rows if x[0][f]["pred_rest"] > 0]
                go = sum(go) / max(1, len(go)); gr = sum(gr) / max(1, len(gr))
                out.append(f"| {gemm} | {G} | {100*fb:.1f}% | {100*es:.0f}% | {f} | {100*sI:.0f}% | {100*sH:.0f}% | {100*so:.0f}% | {go:+.2f} | {gr:+.2f} |")
    out.append("")
(D / "hf/dfp_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
