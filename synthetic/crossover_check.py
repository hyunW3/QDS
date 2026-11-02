"""MR-GPTQ (ICLR 2026) Lemma 2 crossover vs grid: Laplace (native) vs Hadamard-rotated Laplace, E2M1 vs uniform grid.

Per-element relative MSE (sum err^2 / sum x^2) under 1xG absmax block quantization (no scale quantization), RTN.
Also the error decomposition at G=16 by normalized-magnitude region: dead-zone (t < q_min/2), middle, top bins.
"""
import numpy as np
rng = np.random.default_rng(0)
E2 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6.]) / 6
UN = np.arange(8) / 7
GRIDS = {"E2M1": E2, "uniform(E1M2/INT4)": UN}


def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def quant(X, lv):
    amax = np.abs(X).max(1, keepdims=True); amax[amax == 0] = 1
    t = np.abs(X) / amax
    q = lv[np.searchsorted((lv[1:] + lv[:-1]) / 2, t)]
    return np.sign(X) * q * amax, t, q


N = 1 << 21
print("relative MSE (dB = -10log10); 'rot' = real Hadamard applied to Laplace blocks")
print(f"{'G':>4} | " + " | ".join(f"{g:>26s}" for g in GRIDS))
cross = {}
for G in (4, 8, 16, 32, 64, 128, 256):
    X = rng.laplace(size=(N // G, G)) / np.sqrt(2)
    Xr = X @ hadamard(G)
    cells = []
    for g, lv in GRIDS.items():
        e = lambda A: ((quant(A, lv)[0] - A) ** 2).sum() / (A ** 2).sum()
        a, b = -10 * np.log10(e(X)), -10 * np.log10(e(Xr))
        cells.append(f"native {a:5.2f} rot {b:5.2f} ({'rot wins' if b > a else 'native '})")
        cross.setdefault(g, []).append((G, b > a))
    print(f"{G:>4} | " + " | ".join(f"{c:>26s}" for c in cells))
for g, v in cross.items():
    first = next((G for G, w in v if w), None)
    print(f"{g}: rotation first wins at G = {first}")

print("\nG=16 error share by normalized region (t = |x|/block absmax)")
for g, lv in GRIDS.items():
    dz = lv[1] / 2
    for name, A in [("native Laplace", rng.laplace(size=(N // 16, 16)) / np.sqrt(2))]:
        for lab, B in [("native", A), ("rotated", A @ hadamard(16))]:
            Bh, t, q = quant(B, lv)
            e = (Bh - B) ** 2
            top = t > lv[-3]          # region above the 3rd-highest level: the two top bins
            parts = {"dead-zone t<qmin/2": e[t < dz].sum(), "top two bins": e[top].sum()}
            parts["middle"] = e.sum() - parts["dead-zone t<qmin/2"] - parts["top two bins"]
            print(f"  {g:20s} {lab:8s} dB {-10*np.log10(e.sum()/(B**2).sum()):5.2f} | " + ", ".join(f"{k} {100*v/e.sum():4.1f}%" for k, v in parts.items()))
