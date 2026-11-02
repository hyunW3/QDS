"""Figures for the FP4 shrinkage-bias Stage-0 study. Run from the study folder."""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

D = Path(__file__).resolve().parents[1]
FIG = D / "figs"; FIG.mkdir(exist_ok=True)
for _f in (D / "fonts").glob("*.[ot]tf"):
    fm.fontManager.addfont(str(_f))
C_E2M1, C_UNIF, C_SR, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
db = lambda x: -10 * math.log10(max(x, 1e-30))
MODELS = ["llama100M", "llama300M", "llama760M_s7980", "llama760M"]
MLAB = {"llama100M": "100M", "llama300M": "300M", "llama760M_s7980": "760M (초기)", "llama760M": "760M"}
rows = [json.loads(l) for m in MODELS for l in open(D / f"stage0/s0_{m}.jsonl")]
ROLES = [("fprop", "A", "fwd X"), ("fprop", "B", "fwd W"), ("dgrad", "A", "dgrad dY"),
         ("dgrad", "B", "dgrad Wᵀ"), ("wgrad", "A", "wgrad dYᵀ"), ("wgrad", "B", "wgrad Xᵀ")]


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def mean_db(rs, key):
    return np.mean([db(r[key]["nmse"] if isinstance(r[key], dict) else r[key]) for r in rs])


# ---------------- Fig 1: SQNR before -> after RHT, per operand (300M)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, (gm, ro, lab) in enumerate(ROLES):
    for g, c, dy in [("E2M1", C_E2M1, 0.14), ("UNIF", C_UNIF, -0.14)]:
        sI = np.mean([db(r[ro]["nmse"]) for r in sel(model="llama300M", gemm=gm, grid=g, tr="I", scale="fp32", dy_round="rtn")])
        sH = np.mean([db(r[ro]["nmse"]) for r in sel(model="llama300M", gemm=gm, grid=g, tr="H", scale="fp32", dy_round="rtn")])
        y = i + dy
        ax.annotate("", xy=(sH, y), xytext=(sI, y), arrowprops=dict(arrowstyle="-|>", color=c, lw=2, shrinkA=4, shrinkB=4))
        ax.scatter([sI], [y], s=46, facecolor="white", edgecolor=c, lw=2, zorder=3)
        ax.scatter([sH], [y], s=46, color=c, edgecolor="white", lw=1.5, zorder=3)
ax.set_yticks(range(len(ROLES))); ax.set_yticklabels([r[2] for r in ROLES]); ax.invert_yaxis()
ax.set_xlabel("텐서 SQNR (dB, 높을수록 좋음)")
ax.set_title("활성값·gradient에서 RHT는 E2M1을 손해 보게 하고 균일 격자를 이득 보게 한다\nllama 300M, 1×16 블록, RTN, FP32 스케일 · 빈 원 = RHT 없음, 채운 원 = RHT 후", loc="left", fontsize=10, color=INK)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=C_E2M1, lw=2, marker="o", label="E2M1"),
                   Line2D([], [], color=C_UNIF, lw=2, marker="o", label="균일 격자 (E1M2 / INT4)")], loc="lower right")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "fig1_sqnr_rht.png"); plt.close(fig)

# ---------------- Fig 2: per-linear alpha of E2M1 (I vs H), per operand (300M)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
rng = np.random.default_rng(0)
for i, (gm, ro, lab) in enumerate(ROLES):
    for tr, c, dy, fc in [("I", C_GRAY, -0.17, "white"), ("H", C_E2M1, 0.17, C_E2M1)]:
        a = np.array([r[ro]["alpha"] for r in sel(model="llama300M", gemm=gm, grid="E2M1", tr=tr, scale="fp32", dy_round="rtn")])
        ax.scatter(a, i + dy + rng.uniform(-0.07, 0.07, len(a)), s=10, facecolor=fc, edgecolor=c, lw=0.8, alpha=0.8, zorder=3)
        ax.plot([a.mean()] * 2, [i + dy - 0.13, i + dy + 0.13], color=INK, lw=2, zorder=4)
ax.axvline(1.0, color=INK, lw=1, ls="--", zorder=2)
ax.text(1.0004, 5.55, "편향 없음 (α = 1)", fontsize=8, color=INK2, va="center")
ax.set_yticks(range(len(ROLES))); ax.set_yticklabels([r[2] for r in ROLES]); ax.invert_yaxis()
ax.set_xlabel("α = ⟨Q(A), A⟩ / ‖A‖²   (1 미만 = 축소, 1 초과 = 확대)")
ax.set_title("E2M1 축소는 텐서당 0.5% 이하이고, RHT는 α를 1 쪽으로 또는 그 너머로 옮긴다\nllama 300M, 점 하나 = Linear 하나 · 회색 = RHT 없음, 파랑 = RHT 후 · 막대 = 평균", loc="left", fontsize=10, color=INK)
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "fig2_alpha_per_layer.png"); plt.close(fig)

# ---------------- Fig 3: dB budget at GEMM output
def gm_db(m, gemm, g, tr, dr, key="g_nmse"):
    return np.mean([db(r[key]) for r in sel(model=m, gemm=gemm, grid=g, tr=tr, scale="fp32", dy_round=dr)])

EFF = [("per-tensor α 보정\n(E2M1 + RHT)", C_GRAY,
        lambda m, G: gm_db(m, G, "E2M1", "H", "rtn", "g_nmse_acorr") - gm_db(m, G, "E2M1", "H", "rtn")),
       ("격자 교체 E2M1 → 균일\n(둘 다 RHT)", C_UNIF,
        lambda m, G: gm_db(m, G, "UNIF", "H", "rtn") - gm_db(m, G, "E2M1", "H", "rtn")),
       ("dY에 stochastic rounding\n(E2M1 + RHT)", C_SR,
        lambda m, G: gm_db(m, G, "E2M1", "H", "sr") - gm_db(m, G, "E2M1", "H", "rtn"))]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
GEMMS = ["fprop", "dgrad", "wgrad"]; w = 0.26
for j, (lab, c, f) in enumerate(EFF):
    for i, G in enumerate(GEMMS):
        if j == 2 and G == "fprop":
            continue
        vals = [f(m, G) for m in MODELS]
        x = i + (j - 1) * (w + 0.02)
        v300 = f("llama300M", G)
        ax.bar(x, v300, width=w, color=c, label=lab if i == 1 else None, zorder=2)
        ax.scatter([x] * len(vals), vals, s=12, color=INK, zorder=3)
        ytxt = (max(vals + [v300]) + 0.08) if v300 >= 0 else (min(vals + [v300]) - 0.08)
        ax.text(x, ytxt, f"{v300:+.2f}", ha="center",
                va="bottom" if v300 >= 0 else "top", fontsize=8, color=INK)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["FPROP  X·Wᵀ", "DGRAD  dY·W", "WGRAD  dYᵀ·X"])
ax.set_ylabel("GEMM 출력 SQNR 변화 (dB)")
ax.set_ylim(-2.4, 2.4)
ax.set_title("bias 제거의 이득은 약 0 dB, 격자 해상도의 이득은 1.3–1.9 dB\n막대 = 300M, 점 = 체크포인트 4개 (100M / 300M / 760M 초기 / 760M)", loc="left", fontsize=10, color=INK)
ax.legend(loc="lower left", fontsize=8, ncol=3, bbox_to_anchor=(0, -0.32))
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig3_db_budget.png"); plt.close(fig)

# ---------------- Fig 4: E2M1 per-level bias and its offset (fwd X, 300M)
LV = [0, .5, 1, 1.5, 2, 3, 4, 6]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
xs = np.arange(len(LV))
for k, (tr, c, fc, lab) in enumerate([("I", C_GRAY, "white", "RHT 없음"), ("H", C_E2M1, C_E2M1, "RHT 후")]):
    rs = sel(model="llama300M", gemm="fprop", grid="E2M1", tr=tr, scale="fp32", dy_round="rtn")
    mass = np.mean([r["A"]["mass"] for r in rs], 0)
    contrib = np.mean([np.array(r["A"]["mass"]) * np.array(r["A"]["lvl_bias"]) for r in rs], 0)
    cond = contrib / np.maximum(mass, 1e-12)
    off = (k - 0.5) * 0.36
    axs[0].bar(xs + off, cond, width=0.34, color=fc, edgecolor=c, lw=1.5, label=lab, zorder=2)
    axs[1].bar(xs + off, contrib, width=0.34, color=fc, edgecolor=c, lw=1.5, zorder=2)
    alpha_m = np.mean([r["A"]["alpha"] for r in rs])
    axs[1].text(0.01, 0.97 - 0.11 * k, f"{lab}: 개수 기준 net E[q − t] = {contrib.sum():+.4f}  ·  에너지 기준 α = {alpha_m:.4f}",
                transform=axs[1].transAxes, ha="left", va="top", fontsize=8.5, color=INK)
for a in axs:
    a.axhline(0, color=INK, lw=1); a.grid(axis="x", visible=False)
axs[0].set_ylabel("E[q − t | 레벨]")
axs[1].set_ylabel("비중 × bias\n(기여도)"); axs[1].set_ylim(top=0.055)
axs[1].set_xticks(xs); axs[1].set_xticklabels([str(v) for v in LV]); axs[1].set_xlabel("E2M1 크기 레벨 (정규화 단위, 최대 = 6)")
axs[0].legend(loc="upper left")
axs[0].set_title("비대칭인 2·4 bin은 값을 줄이지만, 최상위 bin(6)의 올림이 이를 상쇄한다\nllama 300M, fwd X operand, RTN, FP32 스케일 · Linear 105개 평균", loc="left", fontsize=10, color=INK)
fig.savefig(FIG / "fig4_level_bias.png"); plt.close(fig)

# ---------------- Fig 5: consistency across checkpoints
OPS = [("fprop", "A", "fwd X"), ("dgrad", "A", "dgrad dY"), ("wgrad", "A", "wgrad dYᵀ"), ("fprop", "B", "fwd W")]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
mk = {"llama100M": "o", "llama300M": "s", "llama760M_s7980": "^", "llama760M": "D"}
for i, (gm, ro, lab) in enumerate(OPS):
    for jm, m in enumerate(MODELS):
        y = i + (jm - 1.5) * 0.14
        e = np.mean([db(r[ro]["nmse"]) for r in sel(model=m, gemm=gm, grid="E2M1", tr="H", scale="fp32", dy_round="rtn")])
        u = np.mean([db(r[ro]["nmse"]) for r in sel(model=m, gemm=gm, grid="UNIF", tr="H", scale="fp32", dy_round="rtn")])
        aI = np.mean([r[ro]["alpha"] for r in sel(model=m, gemm=gm, grid="E2M1", tr="I", scale="fp32", dy_round="rtn")])
        aH = np.mean([r[ro]["alpha"] for r in sel(model=m, gemm=gm, grid="E2M1", tr="H", scale="fp32", dy_round="rtn")])
        axs[0].scatter(u - e, y, marker=mk[m], s=28, color=C_UNIF, label=MLAB[m] if i == 0 else None)
        axs[1].annotate("", xy=(aH, y), xytext=(aI, y), arrowprops=dict(arrowstyle="-|>", color=C_E2M1, lw=1.4, shrinkA=2, shrinkB=2))
        axs[1].scatter([aI], [y], marker=mk[m], s=24, facecolor="white", edgecolor=C_E2M1, zorder=3)
        axs[1].scatter([aH], [y], marker=mk[m], s=24, color=C_E2M1, zorder=3)
axs[0].axvline(0, color=INK, lw=1); axs[1].axvline(1, color=INK, lw=1, ls="--")
axs[0].set_yticks(range(len(OPS))); axs[0].set_yticklabels([o[2] for o in OPS]); axs[0].invert_yaxis()
axs[0].set_xlabel("SQNR(균일) − SQNR(E2M1), RHT 후 (dB)")
axs[1].set_xlabel("E2M1 α: RHT 없음(빈 표시) → RHT 후(채운 표시)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.13))
for a in axs: a.grid(axis="y", visible=False)
fig.suptitle("모델 크기·학습 단계와 무관하게 같은 경향", x=0.02, ha="left", fontsize=10, color=INK)
fig.savefig(FIG / "fig5_across_models.png"); plt.close(fig)

# ---------------- Fig 6: synthetic outlier distributions
S = json.load(open(D / "synthetic/sim3_results.json"))
dists = list(dict.fromkeys(s["dist"] for s in S))
parse = lambda s: tuple(float(v) for v in s.split("/"))
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.8), sharey=True)
for i, dname in enumerate(dists):
    for s in S:
        if s["dist"] != dname: continue
        aE, sE = parse(s["rows"][0]); aS, sS = parse(s["rows"][1]); aU, sU = parse(s["rows"][3])
        y = i + (0.15 if s["rht"] else -0.15)
        hollow = not s["rht"]
        for x, c in [(aE, C_E2M1), (aU, C_UNIF)]:
            axs[0].scatter(x, y, s=34, facecolor="white" if hollow else c, edgecolor=c, lw=1.6, zorder=3)
        for x, c in [(sE, C_E2M1), (sU, C_UNIF), (sS, C_SR)]:
            axs[1].scatter(x, y, s=34, facecolor="white" if hollow else c, edgecolor=c, lw=1.6, zorder=3)
axs[0].axvline(1, color=INK, lw=1, ls="--")
KO = {"Gaussian": "Gaussian", "Laplace": "Laplace", "Student-t df5": "Student-t (df5)", "Student-t df3": "Student-t (df3)", "Gauss+1%ch x20": "Gauss + 채널 1% ×20", "Gauss+0.1%pt x50": "Gauss + 원소 0.1% ×50"}
axs[0].set_yticks(range(len(dists))); axs[0].set_yticklabels([KO.get(d, d) for d in dists]); axs[0].invert_yaxis()
axs[0].set_xlabel("α (텐서)"); axs[1].set_xlabel("SQNR (dB)")
for a in axs: a.grid(axis="y", visible=False)
axs[1].legend(handles=[Line2D([], [], ls="", marker="o", color=C_E2M1, label="E2M1 RTN"),
                       Line2D([], [], ls="", marker="o", color=C_UNIF, label="균일 RTN"),
                       Line2D([], [], ls="", marker="o", color=C_SR, label="E2M1 SR"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, label="RHT 없음 (빈 원)"),
                       Line2D([], [], ls="", marker="o", color=INK2, label="RHT 후 (채운 원)")],
              fontsize=8, loc="upper center", ncol=5, bbox_to_anchor=(-0.15, -0.14))
fig.suptitle("합성 분포: outlier + RHT에서는 E2M1 bias가 확대로 뒤집힌다 (α 최대 1.036)\n16×16 RHT, 1×16 블록, FP32 스케일", x=0.02, ha="left", fontsize=10, color=INK)
fig.savefig(FIG / "fig6_synthetic_outliers.png"); plt.close(fig)
print("wrote", sorted(p.name for p in FIG.iterdir()))

# ---------------- Fig 7: E1M2 vs INT4 (symmetric / full-range), 300M
J = json.load(open(D / "stage0/int4_vs_e1m2.json"))
QN = [("E2M1", "E2M1", C_E2M1), ("E1M2", "E1M2 = INT4 대칭(±7)\n(비트 단위 동일)", C_UNIF), ("INT4asym(-8..7)", "INT4 비대칭(−8..7)\n스케일 absmax/8", "#eda100")]
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.6), sharey=True)
for i, (key, lab, c) in enumerate(QN):
    for tr, dy, fc in [("I", -0.13, "white"), ("H", 0.13, c)]:
        rs = [r for r in J if r["grid"] == key and r["tr"] == tr and r["scale"] == "fp32"]
        s = np.mean([db(r["fprop_Y"]["nmse"]) for r in rs]); a = np.mean([r["fprop_Y"]["alpha"] for r in rs])
        axs[0].scatter(s, i + dy, s=60, facecolor=fc, edgecolor=c, lw=2, zorder=3)
        axs[1].scatter(a, i + dy, s=60, facecolor=fc, edgecolor=c, lw=2, zorder=3)
        axs[0].text(s + 0.08, i + dy, f"{s:.2f}", va="center", fontsize=8, color=INK)
        axs[1].text(a + 0.002, i + dy, f"{a:.3f}", va="center", fontsize=8, color=INK)
axs[1].axvline(1, color=INK, lw=1, ls="--")
axs[0].set_yticks(range(len(QN))); axs[0].set_yticklabels([q[1] for q in QN]); axs[0].invert_yaxis()
axs[0].set_xlabel("FPROP 출력 SQNR (dB)"); axs[1].set_xlabel("FPROP 출력 α")
axs[0].set_xlim(18.5, 22.6); axs[1].set_xlim(0.935, 1.02)
for a in axs: a.grid(axis="y", visible=False)
axs[1].legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, label="RHT 없음"),
                       Line2D([], [], ls="", marker="o", color=INK2, label="RHT 후")], loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(-0.1, -0.16))
fig.suptitle("INT4 대칭은 E1M2와 동일하고, −8을 쓰는 INT4는 2–4% 축소되지만 RHT 후 SQNR은 여전히 E2M1보다 높다\nllama 300M, FPROP X·Wᵀ, RTN, FP32 스케일, Linear 105개 평균", x=0.02, ha="left", fontsize=10, color=INK)
fig.savefig(FIG / "fig7_int4_vs_e1m2.png"); plt.close(fig)
print("wrote fig7")

# ---------------- Fig 8/9: cause = grid placement vs post-normalization distribution (cause_probe.json, 300M)
CP = json.load(open(D / "stage0/cause_probe.json"))
C_LM = "#6250d6"
E2 = [0, .5, 1, 1.5, 2, 3, 4, 6]; UN = list(np.arange(8) * 6 / 7)
fig, axs = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True)
centers = np.linspace(0.05, 5.95, 60)
for ax, tr, ttl in [(axs[0], "I", "RHT 없음"), (axs[1], "H", "RHT 후")]:
    v = CP[f"wgrad_dyT|{tr}"]
    eh = np.array(v["energy_hist"][:-1]); eh = eh / eh.sum() * 100   # drop the absmax bin (t=6, zero error)
    ax.fill_between(centers[:-1], 0, eh, step="mid", color=C_GRAY, alpha=0.35, lw=0)
    ax.step(centers[:-1], eh, where="mid", color=C_GRAY, lw=1.2, label="에너지 분포 (|x|² 가중, absmax 원소 제외)")
    top = ax.get_ylim()[1]
    for k, (g, c, lab) in enumerate([(E2, C_E2M1, "E2M1"), (UN, C_UNIF, "균일 (E1M2 = INT4)"), (v["lloyd"], C_LM, "최적 (Lloyd-Max)")]):
        y = top * (1.08 + 0.1 * k)
        ax.scatter(g, [y] * len(g), marker="|", s=160, lw=2.2, color=c, label=lab, clip_on=False)
    s = v["sqnr"]
    ax.text(0.01, 0.62, f"{ttl}\nSQNR  E2M1 {s['E2M1']:.2f} · 균일 {s['UNIF']:.2f} · 최적 {s['LloydMax']:.2f} dB\ncrest factor 중앙값 {v['crest']['p50']:.2f}, 90% {v['crest']['p90']:.2f}\nabsmax 원소(오차 0)의 블록 에너지 비중 {100*v['energy_hist'][-1]:.0f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.set_ylim(0, top * 1.4); ax.set_ylabel("에너지 비중 (%)"); ax.grid(axis="x", visible=False)
axs[1].set_xlabel("블록 정규화 크기 t = |x| / (absmax/6)")
axs[1].legend(loc="upper center", fontsize=8, ncol=4, bbox_to_anchor=(0.5, -0.2))
fig.suptitle("원인: 격자 레벨 위치와 값 분포의 불일치. RHT 후에는 오차 0인 absmax 원소의 에너지 비중이 줄고,\n나머지 에너지가 E2M1이 가장 성긴 4–6 구간으로 쏠린다. 최적 격자는 균일 격자에 가까워진다\nllama 300M, wgrad dYᵀ operand · 위쪽 눈금 = 각 격자의 레벨 위치 (8개씩)", x=0.02, ha="left", fontsize=10, color=INK)
fig.savefig(FIG / "fig8_cause_distribution.png"); plt.close(fig)

OPN = [("fwd_x", "fwd X"), ("fwd_w", "fwd W"), ("dgrad_dy", "dgrad dY"), ("wgrad_dyT", "wgrad dYᵀ")]
fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
for i, (k, lab) in enumerate(OPN):
    for tr, dy, hollow in [("I", -0.15, True), ("H", 0.15, False)]:
        s = CP[f"{k}|{tr}"]["sqnr"]
        for g, c in [("E2M1", C_E2M1), ("UNIF", C_UNIF)]:
            gap = s["LloydMax"] - s[g]
            axs[0].scatter(gap, i + dy, s=44, facecolor="white" if hollow else c, edgecolor=c, lw=1.8, zorder=3)
axs[0].set_yticks(range(len(OPN))); axs[0].set_yticklabels([o[1] for o in OPN]); axs[0].invert_yaxis()
axs[0].axvline(0, color=INK, lw=1)
axs[0].set_xlabel("최적 격자 대비 SQNR 손실 (dB, 0 = 최적)")
axs[0].legend(handles=[Line2D([], [], ls="", marker="o", color=C_E2M1, label="E2M1"), Line2D([], [], ls="", marker="o", color=C_UNIF, label="균일"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, label="RHT 없음"), Line2D([], [], ls="", marker="o", color=INK2, label="RHT 후")],
              fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.17))
axs[0].grid(axis="y", visible=False)
lv = ["0", "0.5", "1", "1.5", "2", "3", "4", "6"]
for k, (key, c, fc, lab) in enumerate([("wgrad_dyT|I", C_GRAY, "white", "RHT 없음"), ("wgrad_dyT|H", C_E2M1, C_E2M1, "RHT 후")]):
    sh = np.array(CP[key]["contrib"]["E2M1"]) * 100
    axs[1].bar(np.arange(8) + (k - 0.5) * 0.38, sh, width=0.36, color=fc, edgecolor=c, lw=1.4, label=lab, zorder=2)
axs[1].set_xticks(range(8)); axs[1].set_xticklabels(lv)
axs[1].set_xlabel("E2M1 레벨 (반올림 결과)")
axs[1].legend(fontsize=8, loc="upper left"); axs[1].grid(axis="x", visible=False)
axs[1].set_ylabel("E2M1 전체 MSE 중 비중 (%), wgrad dYᵀ")
fig.suptitle("RHT 후 균일 격자는 최적에서 0.05–0.3 dB, E2M1은 1.6–3 dB 떨어진다\nE2M1 오차의 75%는 간격 2인 상위 bin(4, 6)에서 나온다\nllama 300M, 8-레벨 Lloyd-Max (0과 6 고정, |x|² 가중)", y=1.06, x=0.02, ha="left", fontsize=10, color=INK)
fig.savefig(FIG / "fig9_gap_to_optimal.png"); plt.close(fig)
print("wrote fig8, fig9")

# ---------------- Fig 10/11: rotation variants vs scale-side fix (rht_variants)
RV = defaultdict(list); RL = defaultdict(list)
for m in ["llama100M", "llama300M", "llama760M"]:
    for l in open(D / f"rotation/rv_{m}.jsonl"):
        r = json.loads(l)
        RV[(m, r["gemm"], r["tf"], r["grid"])].append(db(r["g_nmse"]))
        if r["A_lvl46"] is not None:
            RL[(m, r["gemm"], r["tf"], r["grid"])].append(r["A_lvl46"])
TFL = ["I", "H2", "H4", "H8", "H16", "H32", "H64", "H128", "DCT16", "Haar16", "PERM", "PERM+H16", "ADAPT", "SIGN16", "KLT"]
fam = {"I": C_GRAY, "PERM": C_GRAY, "ADAPT": C_GRAY, "H2": C_E2M1, "H4": C_E2M1, "H8": C_E2M1, "H16": C_E2M1, "H32": C_E2M1,
       "H64": C_E2M1, "H128": C_E2M1, "SIGN16": C_E2M1, "PERM+H16": C_E2M1, "DCT16": C_SR, "Haar16": C_SR, "KLT": C_SR}
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.9), sharey=False)
for ax, g in zip(axs, ["fprop", "dgrad", "wgrad"]):
    for tn in TFL:
        k = ("llama300M", g, tn, "E2M1")
        if k not in RV or tn == "KLT": continue
        x = 100 * np.mean(RL[k]); y = np.mean(RV[k]) - np.mean(RV[("llama300M", g, "I", "E2M1")])
        ax.scatter(x, y, s=38, color=fam[tn], edgecolor="white", lw=1, zorder=3)
        if tn in ("I", "H2", "H4", "H16", "PERM", "Haar16", "DCT16", "ADAPT"):
            ax.annotate(tn, (x, y), xytext=(4, 3 if tn != "ADAPT" else -10), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title({"fprop": "FPROP", "dgrad": "DGRAD", "wgrad": "WGRAD"}[g], fontsize=9, color=INK, loc="left")
    ax.set_xlabel("E2M1 MSE 중 레벨 4·6 비중 (%)")
axs[0].set_ylabel("E2M1 GEMM 출력 SQNR,\n회전 없음(I) 대비 (dB)")
axs[2].legend(handles=[Line2D([], [], ls="", marker="o", color=C_GRAY, label="회전 없음 / 재배치 / 선택"),
                       Line2D([], [], ls="", marker="o", color=C_E2M1, label="Hadamard 계열"),
                       Line2D([], [], ls="", marker="o", color=C_SR, label="에너지 집중 (DCT/Haar)")],
              fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(-0.75, -0.2))
fig.suptitle("E2M1에서는 섞을수록 상위 bin(4·6) 오차가 늘고 SQNR이 떨어진다. 어떤 회전도 '회전 없음'을 이기지 못한다\nllama 300M, 회전 13종 (KLT는 −0.7~−2.7 dB로 범위 밖), 점 = Linear 105개 평균", x=0.02, ha="left", fontsize=10, color=INK, y=1.04)
fig.savefig(FIG / "fig10_rotation_sweep.png"); plt.close(fig)

CONF = [("E2M1 · H16 (NVFP4식)", "H16", "E2M1", C_E2M1, False), ("E2M1 · 회전 없음", "I", "E2M1", C_E2M1, True),
        ("E2M1-4/6 · 회전 없음", "I", "E2M1-4/6", "#6250d6", False), ("E2M1-4/6 · PERM", "PERM", "E2M1-4/6", "#6250d6", True),
        ("균일 · H16 (UFP4)", "H16", "UNIF", C_UNIF, False), ("균일-clip · H16", "H16", "UNIF-clip", C_UNIF, True)]
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.9), sharey=True)
for ax, g in zip(axs, ["fprop", "dgrad", "wgrad"]):
    for i, (lab, tf, gr, c, hollow) in enumerate(CONF):
        for m in ["llama100M", "llama300M", "llama760M"]:
            v = np.mean(RV[(m, g, tf, gr)])
            ax.scatter(v, i, marker=mk[m], s=36, facecolor="white" if hollow else c, edgecolor=c, lw=1.6, zorder=3)
    ax.axvline(np.mean([np.mean(RV[(m, g, "H16", "UNIF")]) for m in ["llama100M", "llama300M", "llama760M"]]), color=C_UNIF, lw=1, ls=":")
    ax.set_title({"fprop": "FPROP", "dgrad": "DGRAD", "wgrad": "WGRAD"}[g], fontsize=9, color=INK, loc="left")
    ax.set_xlabel("GEMM 출력 SQNR (dB)"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(CONF))); axs[0].set_yticklabels([c[0] for c in CONF]); axs[0].invert_yaxis()
axs[1].legend(handles=[Line2D([], [], ls="", marker=mk[m], color=INK2, label=MLAB[m]) for m in ["llama100M", "llama300M", "llama760M"]],
              fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
fig.suptitle("상위 bin 문제는 회전이 아니라 스케일 선택(absmax→4 또는 6)으로 풀린다: 회전 없는 E2M1-4/6이 UFP4에 0.2–0.6 dB까지 접근\n점선 = UFP4 평균 · 균일 격자도 블록별 clip 탐색을 하면 0.15–0.2 dB 더 오른다", x=0.02, ha="left", fontsize=10, color=INK, y=1.04)
fig.savefig(FIG / "fig11_best_configs.png"); plt.close(fig)
print("wrote fig10, fig11")

# ---------------- Fig 12/13: flipped E2M1 (flip_test)
e2 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6.])
GRIDS_A = [("E2M1@6 (기본)", e2 / 6, C_E2M1), ("E2M1@4", e2[e2 <= 4] / 4, "#6250d6"),
           ("FLIP@6 (뒤집기)", np.sort(1 - e2 / 6), C_SR), ("FLIP@4", np.sort(1 - e2[e2 <= 4] / 4), C_SR), ("균일 (E1M2)", np.arange(8) / 7, C_UNIF)]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for i, (lab, lv, c) in enumerate(GRIDS_A):
    ax.plot([0, 1], [i, i], color=GRID, lw=1, zorder=1)
    ax.scatter(lv, [i] * len(lv), marker="|", s=300, lw=2.4, color=c, zorder=2)
    gaps = np.diff(lv); j = int(np.argmax(gaps))
    ax.annotate("", xy=(lv[j + 1], i + 0.22), xytext=(lv[j], i + 0.22), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    ax.text((lv[j] + lv[j + 1]) / 2, i + 0.27, f"최대 간격 {gaps[j]:.2f}", ha="center", va="top", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(GRIDS_A))); ax.set_yticklabels([g[0] for g in GRIDS_A]); ax.set_ylim(len(GRIDS_A) - 0.4, -0.4)
ax.set_xlim(-0.03, 1.03); ax.set_xlabel("블록 내 크기 |x| / absmax"); ax.grid(False)
ax.set_title("블록 하나가 쓸 수 있는 크기 레벨 (8개씩). FLIP은 E2M1을 absmax 기준으로 뒤집어 absmax 근처를 촘촘하게 만든다", loc="left", fontsize=9.5, color=INK)
fig.savefig(FIG / "fig12_flip_grids.png"); plt.close(fig)

FL = defaultdict(list)
for m in ["llama100M", "llama300M", "llama760M"]:
    for l in open(D / f"flip/flip_{m}.jsonl"):
        r = json.loads(l); FL[(m, r["gemm"], r["tf"], r["q"])].append(db(r["g_nmse"]))
for l in open(D / "flip/flipA_llama300M.jsonl"):
    r = json.loads(l); FL[("llama300M", r["gemm"], r["tf"], r["q"])].append(db(r["g_nmse"]))
CF = [("E2M1 · H16 (NVFP4식)", "H16", "E2M1", C_E2M1, False), ("E2M1|FLIP · H16", "H16", "E2M1|FLIP", C_SR, True),
      ("4/6 · 회전 없음", "I", "4/6", "#6250d6", False), ("4/6|FLIP4/6 · H16", "H16", "4/6|FLIP4/6", C_SR, False),
      ("4/6|FLIP4/6 (A만) · H16", "H16", "4/6|FLIP4/6 (A only)", C_SR, True), ("균일 · H16 (UFP4)", "H16", "UNIF", C_UNIF, False)]
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.9), sharey=True)
for ax, g in zip(axs, ["fprop", "dgrad", "wgrad"]):
    for i, (lab, tf, q, c, hollow) in enumerate(CF):
        for m in ["llama100M", "llama300M", "llama760M"]:
            if (m, g, tf, q) not in FL: continue
            ax.scatter(np.mean(FL[(m, g, tf, q)]), i, marker=mk[m], s=36, facecolor="white" if hollow else c, edgecolor=c, lw=1.6, zorder=3)
    ax.axvline(np.mean([np.mean(FL[(m, g, "H16", "UNIF")]) for m in ["llama100M", "llama300M", "llama760M"]]), color=C_UNIF, lw=1, ls=":")
    ax.set_title({"fprop": "FPROP", "dgrad": "DGRAD", "wgrad": "WGRAD"}[g], fontsize=9, color=INK, loc="left")
    ax.set_xlabel("GEMM 출력 SQNR (dB)"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(CF))); axs[0].set_yticklabels([c[0] for c in CF]); axs[0].invert_yaxis()
axs[1].legend(handles=[Line2D([], [], ls="", marker=mk[m], color=INK2, label=MLAB[m]) for m in ["llama100M", "llama300M", "llama760M"]],
              fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
fig.suptitle("블록별로 일반/뒤집기 · absmax→4/6을 고르면(4가지) RHT와 함께 E2M1이 UFP4와 0.15 dB 이내로 같아진다\n순수 FLIP은 −4~−6 dB로 범위 밖 · 점선 = UFP4 평균 · 'A만' = 300M만 측정", x=0.02, ha="left", fontsize=10, color=INK, y=1.04)
fig.savefig(FIG / "fig13_flip_results.png"); plt.close(fig)
print("wrote fig12, fig13")

# ---------------- Fig 14: E2M1@6 vs FLIP@6 only, P(FLIP) by number of near-absmax elements (block_profile2)
BP = json.load(open(D / "flip/block_profile2.json"))
fig, axs = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
ks = np.arange(17)
for tr, c, fc, lab in [("I", C_GRAY, "white", "회전 없음"), ("H16", C_SR, C_SR, "H16 후")]:
    rs = [r for r in BP if r["tr"] == tr]
    tot = np.array([sum(r["n_by_large"][k] for r in rs) for k in ks], float)
    pf = np.array([sum((r["p_flip_by_large"][k] or 0) * r["n_by_large"][k] for r in rs) / tot[k] if tot[k] else np.nan for k in ks])
    ok = tot / tot.sum() > 0.0005
    axs[0].plot(ks[ok], 100 * pf[ok], color=c, lw=2, marker="o", ms=6, mfc=fc, mec=c, label=lab)
    off = -0.2 if tr == "I" else 0.2
    axs[1].bar(ks + off, 100 * tot / tot.sum(), width=0.38, color=fc, edgecolor=c, lw=1.2, label=lab)
axs[0].set_ylabel("FLIP@6을 고를 확률 (%)"); axs[0].set_ylim(-3, 103)
axs[0].legend(loc="lower right", fontsize=8)
axs[1].set_ylabel("블록 비율 (%)")
axs[1].set_xlabel("블록 16개 원소 중 absmax 근처(t > 2/3)인 원소 수")
axs[1].set_xticks(ks)
for a in axs: a.grid(axis="x", visible=False)
axs[0].annotate("outlier 하나에 Hadamard → 16개 값이 같은 크기\n→ FLIP이 거의 오차 없이 표현", xy=(16, 99.9), xytext=(10.6, 55),
                fontsize=8, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
fig.suptitle("E2M1@6 대 FLIP@6: 블록에 absmax 근처 원소가 많을수록 FLIP을 고른다 (5개 ≈ 25%, 7개 ≈ 60%, 9개 이상 ≈ 90–100%)\nllama 300M, 레이어 0·7·14의 wq·w2, fwd X·dgrad dY·wgrad dYᵀ (텐서 18개) 합산", x=0.02, ha="left", fontsize=10, color=INK)
fig.savefig(FIG / "fig14_flip_vs_large.png"); plt.close(fig)
print("wrote fig14")

# ---------------- Fig 15: Stage 1 training (val loss curves + final deltas)
import glob as _glob
S1 = {}
for d in sorted(_glob.glob(str(D / "stage1/runs/*_s*/log.jsonl"))):
    name = Path(d).parent.name
    R = [json.loads(l) for l in open(d)]
    v = [(x["step"], x["val"]) for x in R if "val" in x]
    if v: S1[name] = v
LAB = {"bf16_s0": "BF16 (seed 0)", "bf16_s1": "BF16 (seed 1)", "e2m1_fprop8_s0": "E2M1, FPROP만 FP8", "ufp4_s0": "UFP4 (균일+H16, dY SR)",
       "nvfp4_s0": "NVFP4식 (E2M1, WGRAD H16, dY SR)", "e2m1_wgrad8_s0": "E2M1, WGRAD만 FP8", "e2m1_dgrad8_s0": "E2M1, DGRAD만 FP8",
       "e2m1_h16all_s0": "E2M1 + H16 전체, dY SR", "e2m1_ofp8_s0": "E2M1 + outlier 1% FP8", "e2m1_46_rperm_s0": "E2M1-4/6 + 무작위 재배치",
       "e2m1_46_s0": "E2M1-4/6", "e2m1_smooth_s0": "E2M1 + smoothing", "e2m1_rtn_s0": "E2M1 (변환 없음, RTN)",
       "e2m1_sr_s0": "E2M1 (변환 없음, dY SR)", "e2m1_46_sr_s0": "E2M1-4/6, dY SR", "e2m1_smooth_sr_s0": "E2M1 + smoothing, dY SR",
       "e2m1_46_smooth_s0": "E2M1-4/6 + smoothing", "ufp4_s1": "UFP4 (seed 1)",
       "e2m1_46_smooth_sr_s0": "E2M1-4/6 + smoothing, dY SR", "e2m1_46_smooth_sr_s1": "E2M1-4/6 + smoothing, dY SR (seed 1)"}
fin = {k: v[-1][1] for k, v in S1.items() if v[-1][0] >= 6000}
if "bf16_s0" in fin and "bf16_s1" in fin:
    ref = (fin["bf16_s0"] + fin["bf16_s1"]) / 2
    noise = abs(fin["bf16_s0"] - fin["bf16_s1"]) / 2
    order = sorted([k for k in fin if not k.startswith("bf16")], key=fin.get)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1]})
    hi = {"ufp4_s0": C_UNIF, "ufp4_s1": C_UNIF, "nvfp4_s0": C_E2M1, "e2m1_rtn_s0": C_GRAY, "e2m1_fprop8_s0": C_SR, "e2m1_46_smooth_sr_s0": "#6250d6", "e2m1_46_smooth_sr_s1": "#6250d6"}
    for k in ["bf16_s0"] + order:
        st = [s for s, _ in S1[k] if s >= 1000]; vv = [v for s, v in S1[k] if s >= 1000]
        c = hi.get(k, "#c7c6c0") if k != "bf16_s0" else INK
        axs[0].plot(st, vv, color=c, lw=2 if (k in hi or k == "bf16_s0") else 1, label=LAB.get(k, k) if (k in hi or k == "bf16_s0") else None, zorder=3 if k in hi else 2)
    axs[0].set_xlabel("step"); axs[0].set_ylabel("val loss"); axs[0].legend(fontsize=7.5, loc="upper right"); axs[0].grid(axis="x", visible=False)
    y = np.arange(len(order))
    vals = [fin[k] - ref for k in order]
    axs[1].barh(y, vals, color=[hi.get(k, C_GRAY) for k in order], height=0.62, zorder=2)
    axs[1].axvspan(-noise, noise, color=GRID, zorder=1); axs[1].axvline(0, color=INK, lw=1)
    for yy, vv in zip(y, vals): axs[1].text(vv + 0.001, yy, f"+{vv:.3f}", va="center", fontsize=8, color=INK)
    axs[1].set_yticks(y); axs[1].set_yticklabels([LAB.get(k, k) for k in order], fontsize=8); axs[1].invert_yaxis()
    axs[1].set_xlabel("최종 val loss − BF16 평균 (회색 띠 = BF16 seed 간 차이)"); axs[1].grid(axis="y", visible=False)
    fig.suptitle("Stage 1: 66M 모델, 3.9억 토큰 FP4 학습 (GEMM 3종 에뮬레이션, seed 1개)", x=0.02, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(wspace=0.95)
    fig.savefig(FIG / "fig15_stage1.png"); plt.close(fig)
    print("wrote fig15")

# ---------------- Fig 16: open-source PTQ quality (learned_eval)
LE = {m: json.load(open(D / f"hf/learned_eval_{m}.json")) for m in ["qwen2.5-0.5b", "olmo2-1b", "qwen2.5-1.5b"] if (D / f"hf/learned_eval_{m}.json").exists()}
if LE:
    rows = [("E2M1 + H16", C_E2M1), ("E2M1 회전 없음", C_GRAY), ("UFP4 (균일 + H16)", C_UNIF), ("E2M1-4/6", "#6250d6"),
            ("E2M1 + 그룹별 대각", C_SR), ("E2M1-4/6 + 그룹별 대각", C_SR)]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    w = 0.13; ms = list(LE)
    for j, (r, c) in enumerate(rows):
        vals = [100 * (LE[m][r]["bpb"] / LE[m]["BF16"]["bpb"] - 1) for m in ms]
        xs = np.arange(len(ms)) + (j - (len(rows) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=c, alpha=1.0 if j != 4 else 0.55, label=r, zorder=2)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(["Qwen2.5-0.5B", "OLMo-2-1B", "Qwen2.5-1.5B"][:len(ms)])
    ax.set_ylabel("BF16 대비 BPB 증가 (%)"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("FPROP W4A4 PTQ: 학습형 채널별 스케일(그룹별 대각)이 UFP4·4/6보다 손해가 작다 (낮을수록 좋음)", loc="left", fontsize=10, color=INK)
    fig.savefig(FIG / "fig16_ptq_quality.png"); plt.close(fig)
    print("wrote fig16")

# ---------------- Fig 18: end-to-end latency on Blackwell (latency/e2e_*.json)
E2E = [D / "latency/e2e_13b.json", D / "latency/e2e_rest.json"]
if all(p.exists() for p in E2E):
    rows = sum((json.load(open(p))["rows"] for p in E2E), [])
    modes = [("fp8", "FP8 (E4M3)", "#8a8984"), ("nvfp4_noq", "NVFP4 GEMM만 (양자화 완전 융합 가정)", "#9bbbe6"),
             ("nvfp4", "NVFP4 + 활성값 양자화", C_E2M1), ("nvfp4_46", "NVFP4-4/6 + 양자화", "#6250d6"), ("nvfp4_rht", "NVFP4 + H16 + 양자화", C_UNIF)]
    MNM = {"1.3B-class": "1.3B급 (d2048)", "Qwen2.5-1.5B-shape": "Qwen2.5-1.5B 형태", "Llama-3-8B-shape": "Llama-3-8B 형태"}
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, mn in zip(axs, MNM):
        rr = [r for r in rows if r["model"] == mn]
        labs = [f"prefill\n{r['T']//1024}k" if r["kind"] == "prefill" else f"decode\nB={r['B']}" for r in rr]
        x = np.arange(len(rr)); w = 0.16
        for j, (k, lab, c) in enumerate(modes):
            ax.bar(x + (j - 2) * w, [r["bf16"] / r[k] for r in rr], width=w * 0.92, color=c, label=lab, zorder=2)
        ax.axhline(1, color=INK, lw=1); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(MNM[mn], fontsize=9, color=INK, loc="left"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("BF16 대비 속도 (배, 높을수록 빠름)")
    axs[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("RTX PRO 6000 Blackwell(sm_120)에서 모델 전체 forward 지연시간: FP4는 prefill에서 1.1~2.1배, decode에서는 KV/어텐션 때문에 이득이 작다\n"
                 "랜덤 가중치 Llama형 디코더, CUDA Graph, decode는 context 4096(B=128은 2048)·BF16 KV cache, embedding·lm_head는 BF16",
                 x=0.02, ha="left", fontsize=10, color=INK, y=1.06)
    fig.savefig(FIG / "fig18_latency.png"); plt.close(fig)
    print("wrote fig18")

# ---------------- Fig 20: 8B W4A4 PTQ, format x transform x (RTN | GPTQ)   (ptq8b/results_rows.json)
PR = D / "ptq8b/results_rows.json"
if PR.exists():
    P = json.load(open(PR))
    MODELS8 = [("llama3-8b", "Llama-3-8B"), ("llama31-8b", "Llama-3.1-8B"), ("qwen3-8b", "Qwen3-8B")]
    METH = [("e2m1", "E2M1"), ("e2m1_46", "E2M1-4/6"), ("e2m1_h", "E2M1 + H"), ("ufp4", "UFP4 (균일 + H)"), ("sq_46", "SmoothQuant + 4/6"),
            ("diag", "대각 (ours)"), ("diag_46", "대각 + 4/6 (ours)"), ("diag_h", "대각 + H (ours)"), ("diag_46_h", "대각 + 4/6 + H (ours)"),
            ("mrgptq", "MR-GPTQ (공식)")]
    FMTS = [("nvfp4", "NVFP4 (G16, E4M3 스케일)", True), ("nvfp4_g32", "NVFP4-G32 (G32, E4M3)", False), ("mxfp4", "MXFP4 (G32, E8M0 스케일)", True)]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [6, 3, 6]})
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("q", ["#1baf7a", "#f4f1e6", "#eb6834"])
    for ax, (fmt, title, has_g) in zip(axs, FMTS):
        cols = [(m, g) for m, _ in MODELS8 for g in ([False, True] if has_g else [False])]
        Z = np.full((len(METH), len(cols)), np.nan)
        for j, (m, g) in enumerate(cols):
            bf = P[m]["nvfp4"]["bf16"]["ppl"]
            for i, (k, _) in enumerate(METH):
                r = P[m].get(fmt, {}).get(k + ("+gptq" if g else ""))
                if r: Z[i, j] = 100 * (r["ppl"] / bf - 1)
        Zr = Z.copy()
        for j in range(Z.shape[1]):                      # colour = rank within the column (best green, worst orange)
            col = Z[:, j]; ok = ~np.isnan(col)
            Zr[ok, j] = (np.argsort(np.argsort(col[ok])) / max(ok.sum() - 1, 1))
        ax.imshow(Zr, cmap=cm, aspect="auto", vmin=0, vmax=1)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if not np.isnan(Z[i, j]):
                    best = Z[i, j] == np.nanmin(Z[:, j])
                    ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold" if best else "normal")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{dict(MODELS8)[m]}\n{'GPTQ' if g else 'RTN'}" for m, g in cols], fontsize=7)
        ax.set_yticks(range(len(METH))); ax.set_yticklabels([l for _, l in METH] if fmt == "nvfp4" else [""] * len(METH), fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left"); ax.grid(False)
    fig.suptitle("8B W4A4 PTQ: BF16 대비 WikiText-2 PPL 증가(%), 열마다 초록 = 좋음 · 주황 = 나쁨, 굵은 글씨 = 열 최선\n"
                 "E4M3 스케일(NVFP4)에서는 회전 없는 대각(+4/6), E8M0 스케일(MXFP4)에서는 대각 + 4/6 + H가 최선이며 둘 다 공식 MR-GPTQ보다 낮다",
                 x=0.02, ha="left", fontsize=10, color=INK, y=1.04)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(FIG / "fig20_ptq8b.png"); plt.close(fig)
    print("wrote fig20")

# ---------------- Fig 21: error anatomy of channel scaling vs rotation (ana/ana_*.json)
AN = {m: D / f"ana/ana_{m}.json" for m in ["llama3-8b", "qwen3-8b"]}
if all(p.exists() for p in AN.values()):
    import statistics as _st
    TFS = [("none", "없음", C_GRAY), ("sq", "SmoothQuant", "#9bbbe6"), ("learned", "학습한 대각", C_SR), ("none+rot", "회전(H16)", C_UNIF), ("learned+rot", "대각 + 회전", "#6250d6")]
    BINS = [("dead", "0 근처(dead-zone)", "#d9d6cc"), ("fine", "0.5~2 (촘촘)", "#9bd6bf"), ("mid", "2~4", "#f2b591"), ("top", "4~6 (간격 2)", C_UNIF), ("clip", "클리핑", INK)]
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for row, (m, p) in enumerate(AN.items()):
        R = json.load(open(p))["rows"]
        A = lambda q, tf, k: _st.mean(x[k] for x in R if x["quant"] == q and x["tf"] == tf)
        ax = axs[row, 0]; x = np.arange(len(TFS)); w = 0.27
        for j, (k, lab, c) in enumerate([("out_nmse", "GEMM 출력 오차", INK), ("x_nmse", "X 텐서 오차", "#9bbbe6"), ("w_nmse", "W 텐서 오차", "#f2b591")]):
            base = A("nvfp4_46", "none", k)
            ax.bar(x + (j - 1) * w, [100 * (A("nvfp4_46", t, k) / base - 1) for t, _, _ in TFS], width=w * 0.9, color=c, label=lab, zorder=2)
        ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in TFS], fontsize=7.5, rotation=15)
        ax.set_ylabel("변환 없음 대비 변화 (%)"); ax.set_title(f"{m}: NVFP4-4/6 오차 변화", fontsize=9, loc="left", color=INK); ax.grid(axis="x", visible=False)
        if row == 0: ax.legend(fontsize=7.5)
        for col, q, title in [(1, "nvfp4_46", "NVFP4-4/6: X 오차의 bin별 비중"), (2, "mxfp4", "MXFP4: X 오차의 bin별 비중")]:
            ax = axs[row, col]; bottom = np.zeros(len(TFS))
            for b, lab, c in BINS:
                v = np.array([A(q, t, "x_share_" + b) for t, _, _ in TFS]); ax.bar(x, v, bottom=bottom, color=c, width=0.6, label=lab, zorder=2); bottom += v
            ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in TFS], fontsize=7.5, rotation=15); ax.set_ylim(0, 1)
            ax.set_title(f"{m}: {title}", fontsize=9, loc="left", color=INK); ax.grid(axis="x", visible=False)
            if row == 0 and col == 2: ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("학습한 대각 스케일은 X·W 텐서 오차를 거의 바꾸지 않고 GEMM 출력 오차만 줄인다. 회전은 dead-zone 오차를 없애는 대신 성긴 2~6 구간으로 몰고,\n"
                 "MXFP4(E8M0)에서는 오차의 20~25%가 클리핑이라 회전만이 이를 줄인다 (레이어·그룹 평균, 평가 창 8×1024 토큰)", x=0.02, ha="left", fontsize=10, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "fig21_anatomy.png"); plt.close(fig)
    print("wrote fig21")

# ---------------- Fig 22: loss sensitivity (single-group quantization, 48 windows) by depth
SN = {m: D / f"ana/sens48_{m}.json" for m in ["llama3-8b", "qwen3-8b"]}
if all(p.exists() for p in SN.values()):
    TFS = [("none", "없음", C_GRAY), ("sq", "SmoothQuant", "#9bbbe6"), ("learned", "학습한 대각", C_SR), ("none+rot", "회전(H16)", C_UNIF), ("learned+rot", "대각 + 회전", "#6250d6")]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for ax, (m, p) in zip(axs, SN.items()):
        S = json.load(open(p))["sens"]; L = sorted(set(s["layer"] for s in S)); n = len(L)
        x = np.arange(4); w = 0.16
        for j, (t, lab, c) in enumerate(TFS):
            v = [sum(s["d_nll"] for s in S if s["tf"] == t and s["layer"] in L[k * n // 4:(k + 1) * n // 4]) for k in range(4)]
            ax.bar(x + (j - 2) * w, v, width=w * 0.92, color=c, label=f"{lab} (합 {sum(v):+.3f})", zorder=2)
        ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["앞 1/4", "2/4", "3/4", "마지막 1/4"])
        ax.set_title(m, fontsize=9, loc="left", color=INK); ax.grid(axis="x", visible=False); ax.legend(fontsize=7)
    axs[0].set_ylabel("ΔNLL 합 (그룹 하나씩만 양자화)")
    fig.suptitle("loss 민감도: 학습한 대각은 NVFP4-4/6 손해를 37~43% 줄이고 회전은 15~21% 늘린다. 손해는 마지막 층과 gate/up·down에 몰린다 (평가 창 48×1024 토큰)",
                 x=0.02, ha="left", fontsize=10, color=INK, y=1.04)
    fig.savefig(FIG / "fig22_sensitivity.png"); plt.close(fig)
    print("wrote fig22")

# ---------------- Fig 28: block-normalised magnitude distribution before/after the learned diagonal (and RHT), density & energy
DS = {m: D / f"ana/dist_{m}.json" for m in ["llama3-8b", "qwen3-8b"]}
if all(p.exists() for p in DS.values()):
    E2l = [0, .5, 1, 1.5, 2, 3, 4, 6]
    TFS = [("none", "변환 없음", C_GRAY), ("diag", "학습한 대각 (ours)", C_SR), ("rht", "RHT (H16)", C_UNIF)]
    fig, axs = plt.subplots(2, 4, figsize=(15.5, 6.8), sharex=True)
    for r_, (m, p) in enumerate(DS.items()):
        J = json.load(open(p)); ed = np.array(J["edges"]); ctr = (ed[:-1] + ed[1:]) / 2
        for c_, (op, kind, ylab) in enumerate([("X", "dens", "원소 비율 (%)"), ("X", "ener", "에너지 비율 (%)"), ("W", "dens", "원소 비율 (%)"), ("W", "ener", "에너지 비율 (%)")]):
            ax = axs[r_, c_]; txt = []
            for tf, lab, col in TFS:
                v = J["res"][f"{op}|{tf}"]; y = 100 * np.array(v[kind])
                ax.step(ctr, y, where="mid", color=col, lw=1.5, label=lab)
                if tf == "none": ax.fill_between(ctr, 0, y, step="mid", color=col, alpha=0.15, lw=0)
                am = 100 * (v["absmax_energy"] if kind == "ener" else v["absmax_count"])
                txt.append(f"{lab.split(' ')[0]}: absmax {am:.0f}%" + (f", SQNR {v['sqnr']:.1f}dB, crest {v['crest_p50']:.2f}" if kind == "ener" else ""))
            top = ax.get_ylim()[1]
            ax.scatter(E2l, [top * 1.06] * len(E2l), marker="|", s=120, lw=2, color=C_E2M1, clip_on=False)
            ax.set_ylim(0, top * 1.12)
            ax.text(0.98, 0.95, "\n".join(txt), transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
            ax.set_title(f"{m} · {op} · {'density' if kind == 'dens' else 'energy (|x|² 가중)'}", fontsize=9, loc="left", color=INK)
            ax.grid(axis="x", visible=False)
            if c_ in (0, 2): ax.set_ylabel(ylab)
            if r_ == 1: ax.set_xlabel("블록 정규화 크기 t = |x| / (absmax/6)")
    axs[0, 0].legend(fontsize=8, loc="upper center")
    fig.suptitle("학습한 대각 vs RHT: 블록 안 값의 분포 (absmax 원소 t=6은 제외하고 'absmax %'로 표시, 위쪽 눈금 = E2M1 레벨)\n"
                 "density: 대각은 X·W의 원소 분포를 거의 바꾸지 않지만, RHT는 0 근처를 비우고 4–6 구간을 두 배로 만든다. "
                 "energy: 대각도 X의 absmax 에너지를 일부 2–6 구간으로 옮긴다(RHT보다 약함). W는 대각에서 오히려 absmax 쪽으로 약간 몰린다",
                 x=0.01, ha="left", fontsize=10, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "fig28_diag_distribution.png"); plt.close(fig)
    print("wrote fig28")
