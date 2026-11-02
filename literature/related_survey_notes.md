# Related work check (2026-10-28): survey 2608.25188 and Ordentlich & Polyanskiy 2026a/b

## Jokar, "Transforms for LLM Quantization: The Great Inversion and Format Co-Design" (arXiv 2608.25188, 2026-08)
- Survey, no own benchmarks except one TinyLlama-1.1B layer (Table 5) and illustrative figures.
- Already states our headline as guidance (Sec. 12, Table 16):
  - NVFP4 W4A4: "little or no plain rotation; data-aware block whitening (WUSH) where a fused kernel exists; per-block scale fitting".
  - MXFP4: rotation confined to the scale block helps (E8M0 power-of-two scale), global rotation craters.
  - Lists 4/6 (Cook 2025), SmoothQuant/affine family as the frontier, NVIDIA NVFP4 recipe (RHT on WGRAD only).
- Mechanism it adopts for "rotation can hurt FP": Ordentlich & Polyanskiy's Delta_FP = n sum_i (x_i^2/|x|^2)(y_i^2/|y|^2),
  a JOINT weight-activation concentration; raw operands are anti-aligned (Delta_FP < 1), rotation drives it to 1.
  It explicitly says single-operand concentration is "accommodated for free" by the FP grid.
- Its own Table 5 (G=128 activation groups, E2M1): Hadamard still HELPS E2M1 (+2.8 dB vs +6.2 for INT4); it says it could
  not reproduce the sign flip itself. Consistent with our crossover (E2M1 crosses at G=32).
- Does not cover UFP4 / shrinkage-bias (2606.20381) or FP4 training GEMMs beyond the NVIDIA recipe.

## Ordentlich & Polyanskiy, "High-rate quantized matrix multiplication I" (arXiv 2601.17187)
- Delta_FP comes from the IEIN model: infinite exponent range, relative (multiplicative) error, high rate. Under it the
  per-element error is proportional to x_i^2, so scale and block size drop out.
- Experiment: Llama-3-8B one layer (W_v and its activations), FP8 E4M3 dithered absmax over the WHOLE vector (n=4096),
  INT8 for comparison. Hadamard: INT8 better, FP8 worse; Delta_FP histogram < 1 raw, ~1 rotated.
- NVFP4: only a heuristic upper bound (absmax element exact, small elements overload into {0, 1/2, 1}); compares
  NVINT4+Hadamard vs NVFP4 without rotation (similar), never NVFP4 with vs without rotation.
- So: no claim at FP4 G=16 about why rotation hurts; IEIN cannot produce a block-size crossover.

## Part II (arXiv 2605.13768)
- Weight-only, WaterSIC (waterfilling GPTQ); rotation-immune; GPTQ+random rotation near-optimal at high rate. Not about FP4 W4A4.

## Implication for our framing
- "NVFP4 + RHT is a bad combination, use scaling" is already the survey's recommendation -> not a contribution by itself.
- What is still ours: (1) single-operand level-placement mechanism at G=16 with a quantified split against Delta_FP
  (`dfp_test.py` -> `hf/dfp_tables.md`), (2) UFP4 shrinkage-bias refutation, (3) FP4 training (all three GEMMs).
- Must add WUSH (IST-DASLab/WUSH, ICML 2026, arXiv 2512.00956) as a PTQ baseline.

## Result: Delta_FP vs level placement (`dfp_test.py` -> `hf/dfp_tables.md`, 2026-10-28, gpusvr0908)
Rotation gain (dB, + = rotation helps) split exactly into S (Delta_FP) + nu (single-operand placement) + c (covariance).
pred matches the measured GEMM error within ~0.1 dB (independent-error assumption holds).
- FPROP, G=16, E2M1: total -0.98 / -0.84 / -1.30 (Qwen2.5-0.5B / OLMo-2-1B / Qwen3-1.7B)
  = Delta_FP -0.50 / -0.39 / -1.45 + placement -0.57 / -0.49 / -0.71 + cov +0.09 / +0.06 / +0.83.
  Both terms matter; Delta_FP is NOT negligible (raw operands are anti-aligned: mean Delta_FP 0.74-0.95 -> ~1).
- But Delta_FP is identical for every format and nearly identical for every G (by construction it ignores the grid):
  - Format: same row, INT4 +2.26 vs E2M1 -0.98 (Qwen2.5) -> the whole INT-vs-FP difference is in the placement term.
  - Block size: E2M1 -0.98 (G16) -> -0.10 (G32) -> +2.08 (G128) while Delta_FP stays -0.50/-0.50/-0.52;
    placement -0.57 -> +0.21 -> +2.25. The crossover is entirely placement. Same in all 3 models.
  - Scale format: MX (E8M0) E2M1 at G16 gains +0.92 / +1.04 / +0.41 from rotation; placement +0.93 / +0.68 / +0.95.
- FP8 (the IEIN-favoured format) also loses with block rotation at G16: -2.2 / -1.5 / -2.7 dB; Delta_FP explains only
  -0.5 / -0.4 / -1.45 of it, placement -1.45 / -1.02 / -1.55 (absmax element no longer exact after mixing).
- Exception: OLMo-2 WGRAD, Delta_FP 0.57 -> 1.10 (strong token-axis anti-alignment), S term -4.7 dB dominates.

## Outlier blocks (user question: share of MSE from outliers after RHT; RHT on outlier blocks only)
Outlier blocks = blocks holding a top-1% channel by rms(A_k) rms(B_k): ~14% of blocks, 42-47% of FPROP energy.
- E2M1 FPROP G16 share of GEMM error from outlier blocks: I 40/37/43% -> H 44/45/49% -> oH 47/46/52%.
  H raises outlier-block error by +1.4/+1.7/+1.9 dB and ordinary-block error by +0.6/+0.2/+0.7 dB.
- oH (Hadamard on outlier blocks only) already costs -0.67/-0.75/-0.97 dB, i.e. 68-89% of the full-H loss
  (-0.98/-0.84/-1.30). Most of E2M1's rotation loss comes from mixing the ~14% outlier blocks.
- INT4 is the mirror image: H cuts outlier-block error by -2.8 to -3.0 dB; oH gives +1.2 dB of the +1.5 to +2.3 dB total gain.
- FP8: outlier-block error +3.7 to +4.6 dB under H.
