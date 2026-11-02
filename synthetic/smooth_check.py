"""Spread outliers without mixing? Scalar on RHT vs per-channel smoothing (SmoothQuant-style), synthetic, CPU.

GEMM Y = X W^T (reduce over C). Transforms on the reduction axis, applied so that the product is unchanged:
  I            none
  H16 / H16/32 orthonormal 16x16 RHT / the same scaled by 1/32 on both operands (product rescaled back)
  SMOOTH(a)    X' = X / s, W' = W * s with s_c = max|X_c|^a / max|W_c|^(1-a)  (diagonal, not orthogonal)
  SMOOTH+H16   smoothing, then RHT
Quantization: 1x16 blocks along C, absmax scale (FP32), RTN; E2M1, E2M1 4/6, uniform (E1M2). Metric: GEMM-output SQNR.
Also reports mean(t^2) vs 1/crest^2 (identity) for X after each transform.
"""
import numpy as np
rng = np.random.default_rng(0)
B = 16
E2 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6.])


def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def quant(A, mode):
    Ab = A.reshape(A.shape[0], -1, B)
    amax = np.abs(Ab).max(-1, keepdims=True); amax[amax == 0] = 1
    t = np.abs(Ab) / amax
    def rq(lv):
        q = lv[np.searchsorted((lv[1:] + lv[:-1]) / 2, t)]
        return np.sign(Ab) * q * amax, ((q - t) ** 2).sum(-1, keepdims=True)
    if mode == "E2M1": return rq(E2 / 6)[0].reshape(A.shape)
    if mode == "UNIF": return rq(np.arange(8) / 7)[0].reshape(A.shape)
    if mode == "E2M1-4/6":
        a6, e6 = rq(E2 / 6); a4, e4 = rq(E2[E2 <= 4] / 4)
        return np.where(e4 < e6, a4, a6).reshape(A.shape)


def blockmix(A, M, sg):
    return ((A * sg).reshape(A.shape[0], -1, B) @ M).reshape(A.shape)


N, C, O = 4096, 1024, 1024
W = rng.standard_normal((O, C)) / np.sqrt(C)
datasets = {}
X = rng.standard_normal((N, C)); X[:, rng.choice(C, C // 100, replace=False)] *= 20; datasets["Gauss + 채널 1% ×20"] = X
X = rng.standard_normal((N, C)); X *= np.exp(rng.standard_normal(C) * 0.8)[None, :]; datasets["채널별 크기 차이 (lognormal σ=0.8)"] = X
X = rng.standard_normal((N, C)); m = rng.random((N, C)) < 0.001; X[m] *= 50; datasets["Gauss + 원소 0.1% ×50 (채널 무관)"] = X
H = hadamard(B); sg = rng.choice([-1, 1], size=C)
print("| 데이터 | 변환 | X: mean(t²) / 1/crest² | E2M1 | E2M1-4/6 | 균일 |")
print("|---|---|---|---|---|---|")
for dn, X in datasets.items():
    Y = X @ W.T; nY = (Y ** 2).sum()
    tf = {"I": (X, W, 1.0), "H16": (blockmix(X, H, sg), blockmix(W, H, sg), 1.0),
          "H16 × 1/32": (blockmix(X, H / 32, sg), blockmix(W, H / 32, sg), 32.0 ** 2)}
    for a in (0.5, 0.75):
        s = np.abs(X).max(0) ** a / np.abs(W).max(0) ** (1 - a)
        tf[f"SMOOTH α={a}"] = (X / s, W * s, 1.0)
    s = np.abs(X).max(0) ** 0.5 / np.abs(W).max(0) ** 0.5
    tf["SMOOTH α=0.5 + H16"] = (blockmix(X / s, H, sg), blockmix(W * s, H, sg), 1.0)
    for tn, (Xt, Wt, k) in tf.items():
        Xb = Xt.reshape(N, -1, B); amax = np.abs(Xb).max(-1); t = np.abs(Xb) / amax[..., None]
        crest = amax / np.sqrt((Xb ** 2).mean(-1))
        cells = []
        for q in ["E2M1", "E2M1-4/6", "UNIF"]:
            Yh = (quant(Xt, q) @ quant(Wt, q).T) * k
            cells.append(f"{-10*np.log10(((Yh - Y) ** 2).sum() / nY):.2f}")
        print(f"| {dn} | {tn} | {(t**2).mean():.4f} / {(1/crest**2).mean():.4f} | " + " | ".join(cells) + " |")
