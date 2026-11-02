"""English figures for the method comparison (written to figs/comparison/).

fig_level_placement.pdf: energy-weighted histogram of block-normalized magnitudes t = 6|x|/absmax (300M, WGRAD dY^T),
  without rotation vs with a 16-point block Hadamard, with the E2M1, uniform (E1M2/INT4) and Lloyd-Max level sets.
fig_dfp_decomp.pdf: rotation gain split into Delta_FP / placement / covariance vs block size, per format (FPROP,
  mean over Qwen2.5-0.5B, OLMo-2-1B, Qwen3-1.7B Linears).
"""
import json, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[1]
OUT = D / "figs/comparison"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9ca3af", "#1f2328", "#6b7280"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8.5, "legend.frameon": False})

# ---------- Figure 1: level placement ----------
cp = json.load(open(D / "stage0/cause_probe.json"))
E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6]
UNI = [6 * i / 7 for i in range(8)]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), sharey=True)
for ax, key, title in zip(axes, ["fwd_x|I", "fwd_x|H"], ["No rotation", "16-point block Hadamard"]):
    e = cp[key]["energy_hist"]; w = 6 / 60
    ax.bar([w * (i + .5) for i in range(59)], [100 * x for x in e[:59]], width=w * .9, color=GRAY, linewidth=0)
    top = 8.1
    for lv, y, c in [(E2M1, top, BLUE), (UNI, top - .9, ORANGE), (cp[key]["lloyd"], top - 1.8, INK)]:
        ax.scatter(lv, [y] * len(lv), s=14, color=c, zorder=3, linewidths=0, marker="^" if c == INK else "o")
    ax.text(6.3, top, "E2M1", color=INK, va="center", fontsize=7)
    ax.text(6.3, top - .9, "uniform", color=INK, va="center", fontsize=7)
    ax.text(6.3, top - 1.8, "MSE-optimal", color=INK, va="center", fontsize=7)
    c = cp[key]["contrib"]["E2M1"]
    ax.text(0.15, 5.0, f"absmax element (exact): {100*e[59]:.0f}% of energy\n"
                       f"energy in t$\\in$[4,6): {100*sum(e[40:59]):.0f}%\n"
                       f"E2M1 MSE from levels 4 and 6: {100*(c[6]+c[7]):.0f}%", fontsize=6.8, color=INK, va="top")
    ax.set_title(title, color=INK); ax.set_xlim(-.1, 7.9); ax.set_ylim(0, 8.6)
    ax.set_xticks(range(7)); ax.set_xlabel("block-normalized magnitude  $t = 6|x|/\\max|x|$")
    ax.grid(axis="y", color="#e5e7eb", linewidth=.5); ax.set_axisbelow(True)
axes[0].set_ylabel("share of block energy (%)")
fig.tight_layout(); fig.savefig(OUT / "fig_level_placement.pdf", bbox_inches="tight")
fig.savefig(D / "figs/fig23_level_placement.png", bbox_inches="tight", dpi=160); plt.close(fig)

# ---------- Figure 2: Delta_FP vs placement ----------
db = lambda x: 10 * math.log10(x)
acc = defaultdict(list)
for tag in ["qwen2.5-0.5b", "olmo2-1b", "qwen3-1.7b"]:
    R = [json.loads(l) for l in open(D / f"hf/dfp_{tag}.jsonl")]
    idx = {(r["layer"], r["gemm"], r["G"], r["T"]): r for r in R}
    for (l, gm, g, t), r in idx.items():
        if gm != "fprop" or t != "I": continue
        q = idx[(l, gm, g, "H")]
        for f in ("E2M1", "MXE2M1", "INT4", "FP8"):
            acc[(f, g, "S")].append(db(r["S"] / q["S"])); acc[(f, g, "nu")].append(db(r[f]["nu"] / q[f]["nu"]))
            acc[(f, g, "c")].append(db(r[f]["c"] / q[f]["c"])); acc[(f, g, "act")].append(db(r[f]["act"] / q[f]["act"]))
m = lambda k: sum(acc[k]) / len(acc[k])
GS = [16, 32, 128]
names = {"E2M1": "NVFP4-style (FP4 E2M1)", "MXE2M1": "MXFP4-style (FP4 E2M1)", "INT4": "INT4 (uniform)", "FP8": "FP8 (E4M3)"}
fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.0), sharey=True)
bw = .26
for ax, f in zip(axes, names):
    for j, (term, c, lab) in enumerate([("S", AQUA, "$\\Delta_{FP}$ (joint alignment)"), ("nu", BLUE, "level placement"), ("c", GRAY, "covariance")]):
        ax.bar([i + (j - 1) * bw for i in range(3)], [m((f, g, term)) for g in GS], width=bw * .92, color=c, label=lab, linewidth=0)
    ax.scatter(range(3), [m((f, g, "act")) for g in GS], marker="D", s=14, color=INK, zorder=3, label="measured total")
    ax.axhline(0, color=MUTED, linewidth=.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"G={g}" for g in GS]); ax.set_title(names[f], color=INK, fontsize=7.5)
    ax.grid(axis="y", color="#e5e7eb", linewidth=.5); ax.set_axisbelow(True)
axes[0].set_ylabel("rotation gain (dB)\n(+ = Hadamard helps)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.08), fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_dfp_decomp.pdf", bbox_inches="tight")
fig.savefig(D / "figs/fig24_dfp_decomp.png", bbox_inches="tight", dpi=160); plt.close(fig)
for f in names:
    print(f, " ".join(f"G{g}: tot {m((f,g,'act')):+.2f} S {m((f,g,'S')):+.2f} nu {m((f,g,'nu')):+.2f} c {m((f,g,'c')):+.2f}" for g in GS))

# ---------- Figure 2 (main): measured rotation effect, tensor level and end to end ----------
GS3 = [16, 32, 128]
fmt_names = [("E2M1", "NVFP4-style\n(FP4 E2M1)"), ("MXE2M1", "MXFP4-style\n(FP4 E2M1)"), ("INT4", "INT4\n(uniform)"), ("FP8", "FP8\n(E4M3)")]
# 8B W4A4 RTN WikiText-2 ppl: (no rotation, rotated); NVFP4 G16 rotated = RHT16, others fixed block Hadamard
ppl = {"NVFP4 (G16)": {"Llama-3-8B": (6.887, 7.185), "Llama-3.1-8B": (6.946, 7.195), "Qwen3-8B": (10.100, 10.880)},
       "NVFP4-G32": {"Llama-3-8B": (7.083, 7.390), "Llama-3.1-8B": (7.108, 7.394), "Qwen3-8B": (10.187, 10.950)},
       "NVFP4-G128": {"Llama-3-8B": (7.561, 7.672), "Llama-3.1-8B": (7.625, 7.581), "Qwen3-8B": (10.675, 11.492)},   # RHT128 (snu233, 2026-10-30)
       "MXFP4 (G32)": {"Llama-3-8B": (8.310, 7.784), "Llama-3.1-8B": (8.360, 7.765), "Qwen3-8B": (11.886, 11.490)}}
xlab = {"NVFP4 (G16)": "NVFP4\n(G16)", "NVFP4-G32": "NVFP4-G32", "NVFP4-G128": "NVFP4-G128", "MXFP4 (G32)": "MXFP4\n(G32)"}
SEQ = ["#9dc3f0", "#4f94e3", "#1d5fb8"]          # one hue, light -> dark = small -> large block
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1.2]))
bw = .26
for j, g in enumerate(GS3):
    vals = [m((f, g, "act")) for f, _ in fmt_names]
    a1.bar([i + (j - 1) * bw for i in range(4)], vals, width=bw * .92, color=SEQ[j], label=f"G={g}", linewidth=0)
a1.axhline(0, color=MUTED, linewidth=.6); a1.set_xticks(range(4)); a1.set_xticklabels([n for _, n in fmt_names], fontsize=7)
a1.set_ylabel("GEMM error reduction\nfrom rotation (dB)"); a1.set_title("(a) tensor level, FPROP, 3 open models", color=INK, fontsize=8)
a1.legend(fontsize=7, loc="upper left"); a1.grid(axis="y", color="#e5e7eb", linewidth=.5); a1.set_axisbelow(True)
MC = [BLUE, ORANGE, AQUA]
settings = list(ppl)
for j, mdl in enumerate(["Llama-3-8B", "Llama-3.1-8B", "Qwen3-8B"]):
    vals = [100 * (ppl[st][mdl][0] - ppl[st][mdl][1]) / ppl[st][mdl][0] for st in settings]
    a2.bar([i + (j - 1) * bw for i in range(len(settings))], vals, width=bw * .92, color=MC[j], label=mdl, linewidth=0)
a2.axhline(0, color=MUTED, linewidth=.6); a2.set_xticks(range(len(settings))); a2.set_xticklabels([xlab[x] for x in settings], fontsize=6.5)
a2.set_ylabel("ppl reduction from\nrotation (%)"); a2.set_title("(b) 8B W4A4 RTN, WikiText-2", color=INK, fontsize=8)
a2.legend(fontsize=6.5, loc="upper left"); a2.grid(axis="y", color="#e5e7eb", linewidth=.5); a2.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "fig_rotation_gain.pdf", bbox_inches="tight")
fig.savefig(D / "figs/fig25_rotation_gain.png", bbox_inches="tight", dpi=160); plt.close(fig)
