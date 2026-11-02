"""Is the 'values pile up near absmax after RHT' effect caused by the Hadamard normalization (1/sqrt(16))?

Checks on synthetic data (CPU):
 (1) scale invariance: H/sqrt(16), un-normalized H, 0.1*H give identical t = |x|/absmax and identical E2M1 SQNR;
 (2) mixing shape: I vs Hadamard vs a Haar-random 16x16 orthogonal matrix -> crest factor, number of near-absmax
     elements per block, E2M1@6-vs-FLIP@6 and E2M1@6-vs-@4 choice shares, E2M1 / uniform SQNR.
"""
import numpy as np
rng = np.random.default_rng(0)
B = 16
E2 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6.])
LV = {"E2M1@6": E2 / 6, "E2M1@4": E2[E2 <= 4] / 4, "FLIP@6": np.sort(1 - E2 / 6), "UNIF": np.arange(8) / 7}


def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def haar_orth(n):
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))


def qerr(t, lv):
    q = lv[np.searchsorted((lv[1:] + lv[:-1]) / 2, t)]
    return ((q - t) ** 2)


def analyze(X, M):
    Xb = X.reshape(-1, B) * rng.choice([-1, 1], size=B)    # random signs shared by all blocks
    Y = Xb @ M
    amax = np.abs(Y).max(1, keepdims=True); amax[amax == 0] = 1
    t = np.abs(Y) / amax
    w = (amax ** 2)[:, 0]                                    # block scale^2 -> absolute-domain MSE
    e = {k: qerr(t, lv).sum(1) for k, lv in LV.items()}
    sig = (w * (t ** 2).sum(1)).sum()
    sq = lambda k: -10 * np.log10((w * e[k]).sum() / sig)
    crest = amax[:, 0] / np.sqrt((Y ** 2).mean(1)).clip(1e-30)
    return dict(crest50=np.median(crest), crest90=np.percentile(crest, 90), nlarge=(t > 2 / 3).sum(1).mean(),
                all16=((t > 2 / 3).sum(1) == 16).mean(), flip=(e["FLIP@6"] < e["E2M1@6"]).mean(),
                four=(e["E2M1@4"] < e["E2M1@6"]).mean(), sq_e2m1=sq("E2M1@6"), sq_unif=sq("UNIF"), t=t)


N, C = 8192, 1024
data = {"Gaussian": rng.standard_normal((N, C)), "Laplace": rng.laplace(size=(N, C)), "Student-t df3": rng.standard_t(3, (N, C))}
X = rng.standard_normal((N, C)); X[:, rng.choice(C, C // 100, replace=False)] *= 20; data["Gauss + 채널 1% ×20"] = X
X = rng.standard_normal((N, C)); m = rng.random((N, C)) < 0.001; X[m] *= 50; data["Gauss + 원소 0.1% ×50"] = X

H = hadamard(B)
print("(1) scale invariance of the Hadamard normalization (Gauss + 원소 0.1% ×50):")
X = data["Gauss + 원소 0.1% ×50"]
ref = None
for nm, M in [("H/sqrt(16)", H / 4), ("H (un-normalized)", H), ("0.1*H", 0.1 * H)]:
    rng = np.random.default_rng(1); r = analyze(X, M)
    if ref is None: ref = r["t"]
    print(f"  {nm:18s} max|t - t_ref| = {np.abs(r['t'] - ref).max():.1e}   E2M1 SQNR {r['sq_e2m1']:.4f} dB   #near-absmax {r['nlarge']:.3f}")

print("\n(2) mixing shape: I vs Hadamard vs Haar-random orthogonal (all normalized)")
print("| 분포 | 변환 | crest 중앙값 / 90% | absmax 근처 원소 수 (t>2/3) | 16개 모두 근처 | FLIP@6 선택 | E2M1@4 선택 | SQNR E2M1 / 균일 |")
print("|---|---|---|---|---|---|---|---|")
for dn, X in data.items():
    for tn, M in [("I", np.eye(B)), ("Hadamard", H / 4), ("무작위 직교", haar_orth(B))]:
        rng = np.random.default_rng(1); r = analyze(X, M)
        print(f"| {dn} | {tn} | {r['crest50']:.2f} / {r['crest90']:.2f} | {r['nlarge']:.2f} | {100*r['all16']:.2f}% | {100*r['flip']:.1f}% | {100*r['four']:.1f}% | {r['sq_e2m1']:.2f} / {r['sq_unif']:.2f} |")
