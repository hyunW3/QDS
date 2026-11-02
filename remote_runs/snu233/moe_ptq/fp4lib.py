"""FP4 block-quantization emulation + metrics for the shrinkage-bias study (arXiv 2606.20381).

All operands are 2D with the GEMM reduction axis LAST; blocks are 1x16 along that axis.
"""
import math
import torch

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.])
UNIF = torch.arange(8, dtype=torch.float32) * 0.25   # E1M2 magnitudes {0..1.75}; == INT4 0..7 up to scale
GRIDS = {"E2M1": E2M1, "UNIF": UNIF}
BLK = 16


def hadamard(n, device):
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def rht(A, signs, H):
    """Random Hadamard on each 16-block of the last axis. signs: [K] +-1 (shared by both GEMM operands)."""
    N, K = A.shape
    return ((A * signs).view(N, K // BLK, BLK) @ H).view(N, K)


def _round_scale(s, scale_fmt):
    if scale_fmt == "fp32":
        return s
    if scale_fmt == "e4m3":   # NVFP4-style: FP32 per-tensor global so that block scales fit E4M3 (max 448)
        g = s.max() / 448.0
        return (s / g).to(torch.float8_e4m3fn).float() * g
    raise ValueError(scale_fmt)


def quantize(A, grid, rounding="rtn", scale_fmt="fp32", gen=None, thr=None):
    """Return (dequantized A_hat, normalized magnitudes t, quantized magnitudes q)."""
    g = grid.to(A.device)
    N, K = A.shape
    Ab = A.float().view(N, K // BLK, BLK)
    s = Ab.abs().amax(-1, keepdim=True) / g[-1]
    s = _round_scale(s, scale_fmt)
    s = torch.where(s == 0, torch.ones_like(s), s)
    t = (Ab.abs() / s).clamp(max=g[-1])           # E4M3 rounding of s can push t slightly above g_max -> clip
    if rounding == "rtn":
        th = (g[1:] + g[:-1]) / 2 if thr is None else thr.to(A.device)
        q = g[torch.bucketize(t, th, right=False)]  # ties: t==th goes to lower level (RTNE-ish; measure-zero)
    elif rounding == "sr":
        i = (torch.bucketize(t, g, right=True) - 1).clamp(0, len(g) - 2)
        lo, hi = g[i], g[i + 1]
        p = (t - lo) / (hi - lo)
        u = torch.rand(t.shape, device=A.device, generator=gen)
        q = torch.where(u < p, hi, lo)
    else:
        raise ValueError(rounding)
    Ah = (torch.sign(Ab) * q * s).view(N, K)
    return Ah, t.reshape(-1), q.reshape(-1)


def tensor_metrics(A, Ah, t, q, grid):
    A = A.float()
    nA = (A * A).sum()
    e = Ah - A
    out = {
        "nmse": ((e * e).sum() / nA).item(),
        "alpha": ((Ah * A).sum() / nA).item(),
        # mean normalized-space error (paper's b_G, measured on the real tensor)
        "bG": (q - t).mean().item(),
    }
    g = grid.to(A.device)
    # bucket occupancy + per-level conditional bias
    idx = torch.bucketize(q, g)  # exact levels
    cnt = torch.bincount(idx, minlength=len(g)).float()
    p = cnt / cnt.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum()
    out["beff"] = (ent.exp() / len(g)).item()
    err = torch.zeros(len(g), device=A.device).index_add_(0, idx, q - t)
    out["mass"] = p.tolist()
    out["lvl_bias"] = (err / cnt.clamp(min=1)).tolist()
    return out


def gemm_metrics(Y, Yh):
    nY = (Y * Y).sum()
    a = ((Yh * Y).sum() / nY).item()
    rel = (((Yh - Y) ** 2).sum() / nY).item()
    rel_c = (((Yh / a - Y) ** 2).sum() / nY).item()
    return {"g_nmse": rel, "g_alpha": a, "g_nmse_acorr": rel_c}


def db(x):
    return -10 * math.log10(max(x, 1e-30))
