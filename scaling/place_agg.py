"""Aggregate placement_test.py -> hf/place_tables.md (dense) and moe/place_tables.md (MoE).

Mean over Linears of the GEMM SQNR gain (dB) vs E2M1 with no transform and absmax -> 6 (I@6); + = better.
'top46' = share of block energy with 6|x|/blockmax in [4, 6) before rounding (mean of the two operands).
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
D = Path(__file__).resolve().parents[1]
db = lambda x: -10 * math.log10(max(x, 1e-30))
COLS = ["I@4", "I@3", "I@4/6", "I@opt", "H16@6", "H16@4", "H16@3", "H16@4/6", "H16@opt", "OCS@6", "OCS@4/6", "OCS@opt", "SM@6", "SM@4/6", "SM@opt", "SMOCS@6", "SMOCS@4/6", "SMOCS@opt", "QDS@6", "QDS@4/6", "QDS@opt", "QDSOCS@6", "QDSOCS@4/6", "QDSOCS@opt", "IUNIF", "H16UNIF"]
HEAD = ["I@4", "I@3", "I@4/6", "I@opt", "H@6", "H@4", "H@3", "H@4/6", "H@opt", "OCS@6", "OCS@4/6", "OCS@opt", "SQ@6", "SQ@4/6", "SQ@opt", "OCS+SQ@6", "OCS+SQ@4/6", "OCS+SQ@opt", "QDS@6", "QDS@4/6", "QDS@opt", "OCS+QDS@6", "OCS+QDS@4/6", "OCS+QDS@opt", "U", "U+H"]


def table(files, title, groups):
    out = [f"# {title}", "", "GEMM SQNR gain (dB) vs E2M1 I@6 (no transform, absmax -> 6); mean over Linears. H = 16-point Hadamard;",
           "OCS = top-1% channel splitting; SQ = closed-form SmoothQuant diag (alpha 0.5); QDS = quantization-aware diagonal scaling (alpha searched by FP4 output error); OCS+X = X then OCS; @m = absmax mapped to level m; @opt = per-block min-MSE m in {3..8} (m > 6 clips);",
           "U = uniform grid (INT4/E1M2), U+H = UFP4. top46 = block energy share in [4, 6) for I / H16 / OCS.", ""]
    for tag, p in files:
        R = [json.loads(l) for l in open(p)]
        out += [f"## {tag}", "", "| group | GEMM | n | I@6 dB | " + " | ".join(HEAD) + " | top46 I / H / OCS |",
                "|---|---|---|---|" + "---|" * len(HEAD) + "---|"]
        for gname, pred in groups:
            for gemm in ("fprop", "dgrad", "wgrad"):
                rs = [r for r in R if r["gemm"] == gemm and pred(r["layer"]) and "I@6" in r]
                if not rs: continue
                base = [db(r["I@6"]) for r in rs]
                cells = []
                for c in COLS:
                    v = [db(r[c]) - b for r, b in zip(rs, base) if c in r]
                    cells.append(f"{sum(v)/len(v):+.2f}" if v else "-")
                tops = " / ".join(f"{100*sum(r[k] for r in rs)/len(rs):.0f}%" for k in ("I|top46", "H16|top46", "OCS|top46"))
                out.append(f"| {gname} | {gemm} | {len(rs)} | {sum(base)/len(base):.2f} | " + " | ".join(cells) + f" | {tops} |")
        out.append("")
    return "\n".join(out) + "\n"


dense = [(t, D / f"hf/place_{t}.jsonl") for t in ["qwen2.5-0.5b", "olmo2-1b", "qwen3-1.7b"] if (D / f"hf/place_{t}.jsonl").exists()]
if dense:
    s = table(dense, "Placing spread energy below 4 (`placement_test.py`, dense models)", [("all", lambda l: True)])
    (D / "hf/place_tables.md").write_text(s); print(s)
moe = [(p.stem.replace("place_", ""), p) for p in sorted((D / "moe").glob("place_*.jsonl"))]
if moe:
    s = table(moe, "Placing spread energy below 4 (`placement_test.py`, MoE models)",
              [("experts", lambda l: ".experts." in l), ("attention", lambda l: ".self_attn." in l),
               ("shared/dense MLP", lambda l: ".mlp." in l and ".experts." not in l)])
    (D / "moe/place_tables.md").write_text(s); print(s)
