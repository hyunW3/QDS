"""All 4-bit sign+3-bit formats under 1x16 absmax block scaling: E3M0, E2M1, E1M2 (=E0M3=INT4-sym). Synthetic, CPU.
Level sets in units of the block absmax (subnormals included; the exponent bias does not matter under absmax scaling).
"""
import numpy as np
rng = np.random.default_rng(0)
B = 16
F = {
    "E3M0 (log)": np.array([0] + [2.0 ** -k for k in range(6, -1, -1)]),            # 0, 1/64 ... 1/2, 1
    "E2M1": np.array([0, .5, 1, 1.5, 2, 3, 4, 6.]) / 6,
    "E1M2 = E0M3 = INT4": np.arange(8) / 7,
}
H = np.ones((1, 1))
while H.shape[0] < B:
    H = np.block([[H, H], [H, -H]])
H /= np.sqrt(B)


def sqnr(X, lv):
    Xb = X.reshape(-1, B); amax = np.abs(Xb).max(1, keepdims=True); amax[amax == 0] = 1
    t = np.abs(Xb) / amax
    q = lv[np.searchsorted((lv[1:] + lv[:-1]) / 2, t)]
    e = ((np.sign(Xb) * q * amax - Xb) ** 2).sum(); return -10 * np.log10(e / (Xb ** 2).sum())


N, C = 4096, 1024
data = {"Gaussian": rng.standard_normal((N, C)), "Laplace": rng.laplace(size=(N, C)), "Student-t df3": rng.standard_t(3, (N, C)),
        "log-normal |x| (σ=1.5)": rng.choice([-1, 1], (N, C)) * np.exp(1.5 * rng.standard_normal((N, C)))}
X = rng.standard_normal((N, C)); X[:, rng.choice(C, C // 100, replace=False)] *= 20; data["Gauss + 채널 1% ×20"] = X
sg = rng.choice([-1, 1], C)
print("| 분포 | 변환 | " + " | ".join(F) + " |"); print("|---|---|" + "---|" * len(F))
for dn, X in data.items():
    for tn in ["없음", "H16"]:
        Xt = X if tn == "없음" else ((X * sg).reshape(N, -1, B) @ H).reshape(N, C)
        print(f"| {dn} | {tn} | " + " | ".join(f"{sqnr(Xt, lv):.2f}" for lv in F.values()) + " |")
