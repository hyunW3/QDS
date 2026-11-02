# Stage 1 summary (m7: deployable smoothing + 3rd seeds)

BF16 mean final val 3.7159 over 3 seeds (3.7238, 3.7144, 3.7094).
Δ = final val − BF16 mean. Std = sample std over seeds. s/step = steady-state (steps 100→6000) on one RTX A5000 (4 runs in parallel on the node).

## Per arm

| arm | seeds | final val mean ± std | Δ mean ± std | Δ per seed | s/step | h/run |
|---|---|---|---|---|---|---|
| bf16 | 0,1,2 | 3.7159 ± 0.0073 | -0.0000 ± 0.0073 | s0 +0.0079 / s1 -0.0015 / s2 -0.0064 | 1.15 | 1.9 |
| ufp4 | 0,1,2 | 3.7594 ± 0.0062 | +0.0435 ± 0.0062 | s0 +0.0445 / s1 +0.0368 / s2 +0.0491 | 3.20 | 5.3 |
| e2m1_46_smooth_sr | 0,1,2 | 3.7606 ± 0.0059 | +0.0448 ± 0.0059 | s0 +0.0462 / s1 +0.0383 / s2 +0.0498 | 4.09 | 6.8 |
| e2m1_46_smoothdelay_sr | 0,1,2 | 3.7624 ± 0.0065 | +0.0466 ± 0.0065 | s0 +0.0539 / s1 +0.0412 / s2 +0.0446 | 4.12 | 6.9 |
| e2m1_46_smoothlearn_sr | 0,1,2 | 3.7608 ± 0.0085 | +0.0449 ± 0.0085 | s0 +0.0545 / s1 +0.0382 / s2 +0.0421 | 6.04 | 10.1 |
| nvfp4 | 0,1,2 | 3.7635 ± 0.0059 | +0.0477 ± 0.0059 | s0 +0.0535 / s1 +0.0418 / s2 +0.0478 | 2.89 | 4.8 |
| e2m1_sr | 0,1,2 | 3.7633 ± 0.0047 | +0.0474 ± 0.0047 | s0 +0.0528 / s1 +0.0443 / s2 +0.0450 | 2.37 | 4.0 |
| e2m1_46_sr | 0 | 3.7649 | +0.0491 | s0 +0.0491 | 3.21 | 5.4 |

## Per run

| run | step | final val | Δ vs BF16 mean | train loss (last 200) | s/step | h | smooth_learn L (last 200) | L_learn / L_oracle-formula (mean of diag steps) |
|---|---|---|---|---|---|---|---|---|
| bf16_s0 | 6000 | 3.7238 | +0.0079 | 3.6649 | 1.15 | 1.9 |  |  |
| bf16_s1 | 6000 | 3.7144 | -0.0015 | 3.6600 | 1.15 | 1.9 |  |  |
| bf16_s2 | 6000 | 3.7094 | -0.0064 | 3.6492 | 1.15 | 1.9 |  |  |
| ufp4_s0 | 6000 | 3.7604 | +0.0445 | 3.7048 | 3.21 | 5.3 |  |  |
| ufp4_s1 | 6000 | 3.7527 | +0.0368 | 3.7010 | 3.20 | 5.3 |  |  |
| ufp4_s2 | 6000 | 3.7650 | +0.0491 | 3.7052 | 3.20 | 5.3 |  |  |
| e2m1_46_smooth_sr_s0 | 6000 | 3.7621 | +0.0462 | 3.7031 | 4.14 | 6.9 |  |  |
| e2m1_46_smooth_sr_s1 | 6000 | 3.7542 | +0.0383 | 3.7028 | 4.07 | 6.8 |  |  |
| e2m1_46_smooth_sr_s2 | 6000 | 3.7656 | +0.0498 | 3.7051 | 4.07 | 6.8 |  |  |
| e2m1_46_smoothdelay_sr_s0 | 6000 | 3.7697 | +0.0539 | 3.7107 | 4.11 | 6.8 |  |  |
| e2m1_46_smoothdelay_sr_s1 | 6000 | 3.7571 | +0.0412 | 3.7063 | 4.11 | 6.8 |  |  |
| e2m1_46_smoothdelay_sr_s2 | 6000 | 3.7605 | +0.0446 | 3.7033 | 4.14 | 6.9 |  |  |
| e2m1_46_smoothlearn_sr_s0 | 6000 | 3.7704 | +0.0545 | 3.7136 | 6.12 | 10.2 | 0.01169 | 1.0397 |
| e2m1_46_smoothlearn_sr_s1 | 6000 | 3.7541 | +0.0382 | 3.7016 | 6.03 | 10.1 | 0.01189 | 1.0449 |
| e2m1_46_smoothlearn_sr_s2 | 6000 | 3.7580 | +0.0421 | 3.6973 | 5.97 | 10.0 | 0.01162 | 1.0411 |
| nvfp4_s0 | 6000 | 3.7694 | +0.0535 | 3.7127 | 2.95 | 4.9 |  |  |
| nvfp4_s1 | 6000 | 3.7576 | +0.0418 | 3.7064 | 2.87 | 4.8 |  |  |
| nvfp4_s2 | 6000 | 3.7636 | +0.0478 | 3.7044 | 2.84 | 4.7 |  |  |
| e2m1_sr_s0 | 6000 | 3.7687 | +0.0528 | 3.7129 | 2.39 | 4.0 |  |  |
| e2m1_sr_s1 | 6000 | 3.7602 | +0.0443 | 3.7083 | 2.34 | 3.9 |  |  |
| e2m1_sr_s2 | 6000 | 3.7609 | +0.0450 | 3.7024 | 2.36 | 3.9 |  |  |
| e2m1_46_sr_s0 | 6000 | 3.7649 | +0.0491 | 3.7076 | 3.21 | 5.4 |  |  |

## Paired Δ vs e2m1_46_smooth_sr (same seed)

| arm | per seed | mean ± std |
|---|---|---|
| ufp4 | s0 -0.0017 / s1 -0.0015 / s2 -0.0007 | -0.0013 ± 0.0005 |
| e2m1_46_smoothdelay_sr | s0 +0.0077 / s1 +0.0029 / s2 -0.0051 | +0.0018 ± 0.0065 |
| e2m1_46_smoothlearn_sr | s0 +0.0083 / s1 -0.0001 / s2 -0.0077 | +0.0002 ± 0.0080 |

## Arm definitions (m7)

- e2m1_46_smooth_sr: ORACLE SmoothQuant α=0.5 on every GEMM's reduction axis, s = sqrt(colabsmax(A)/colabsmax(B)) from
  the tensors being quantized (unchanged from earlier runs).
- e2m1_46_smoothdelay_sr: same formula, but colabsmax from the same call site (layer, Linear, FPROP/DGRAD/WGRAD; 252
  sites) in the PREVIOUS optimizer step (elementwise max over that step's 4 grad-accum micro-batches); step 0 uses the
  current stats. Eval forwards use stored stats and do not update them.
- e2m1_46_smoothlearn_sr: s = exp(v) per site, v initialised from the SmoothQuant formula at the site's first call; each
  training call uses the current v, then one Adam step (lr 1e-2, betas 0.9/0.999, own state) on
  L = ||Q(A/s)Q(Bs)^T - AB^T||^2/||AB^T||^2 over <=1024 random rows of each operand (detached, same E2M1 4/6 + SR-on-dY
  quantizer; STE through rounding, per-block absmax kept differentiable). Objective compiled with torch.compile.
  Last column above: mean over the every-100-step diagnostics of L(learned s)/L(oracle formula s) on the same subsample
  (>1 = learned s worse than the oracle per GEMM).
- All three share grid E2M1, 4/6 block scale selection, 1x16 blocks, SR on dY, RTN elsewhere, 6000 steps x 65,536 tokens.
- WGRAD reduces over tokens, so its delayed / learned s is indexed by token position within the 16x1024 micro-batch, not
  by a persistent channel.

