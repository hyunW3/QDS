# FP4 shrinkage bias (arXiv 2606.20381), Stage 0: real-tensor analysis

Date: 2026-10-21, gpusvr0908 (A5000 x4), lingua venv torch 2.7.

## Setup
- Checkpoints: lingua llama baselines 100M (step 4400), 300M (step 13000), 760M (step 26600, and early step 7980).
- One fwd+bwd pass, DCLM val, batch 4 x 1024 tokens, bf16 autocast. Output head excluded.
- For every block Linear, the three training GEMMs are formed with 1x16 blocks along each GEMM's reduction axis:
  - FPROP: X, W (reduce K)
  - DGRAD: dY, W^T (reduce O)
  - WGRAD: dY^T, X^T (reduce tokens)
- Grids: E2M1 {0,.5,1,1.5,2,3,4,6}; UNIF = E1M2 (equivalent to INT4 under absmax scaling).
- Transform I (none) or H: a 16x16 random Hadamard, with shared signs on both operands.
- Block scale: fp32 (paper Table 1) or E4M3 with a per-tensor fp32 global (NVFP4).
- Rounding: RTN, or SR on the dY operand only.
- Metrics:
  - Tensor level: SQNR = -10 log10 NMSE; alpha = <Q(A),A>/||A||^2; b_G = mean(q - t) in normalized units; B_eff.
  - GEMM output level: SQNR_Y; alpha_Y; SQNR_Y after a per-tensor 1/alpha_Y correction.
- Full tables: `stage0/stage0_tables.md`.

## Findings (consistent across 100M / 300M / 760M / 760M-early)
1. **The paper's SQNR claim reproduces.**
   - Activations and gradients (fwd_x, dY, dY^T): E2M1 wins without RHT by 0.4-1.9 dB.
   - With RHT, E2M1 drops (Delta -0.5 to -1.8 dB) while UNIF gains (+1.5 to +3.1 dB). UNIF then leads by about 1.5-3 dB.
   - Weights: RHT is neutral, and UNIF leads by about 0.8 dB in both cases.
2. **The shrinkage-bias claim is not supported as the driver.**
   - Without RHT, the tensor-level alpha of E2M1 RTN is 0.996-0.998. The normalized b_G is about -0.017, but the 6-level bin (absmax element plus round-up from (5,6)) offsets most of it.
   - RHT does not amplify shrinkage. The mean alpha moves toward or above 1: fwd_x 0.9967 -> 0.9998 (300M); dY^T 0.998 -> 1.003-1.005; w2 input 1.005.
   - Per-layer alpha_H spans 0.995 to 1.035. This matches the "outlier + RHT flips the sign" result from the synthetic study.
3. **At GEMM output, the bias is negligible next to the noise.**
   - E2M1 alpha_Y is 0.992-0.999, a product shrinkage of about 0.5-0.8% for FPROP/DGRAD.
   - A per-tensor 1/alpha_Y correction changes SQNR_Y by only -0.05 to +0.04 dB.
   - The UNIF-over-E2M1 gap after RHT is 1.3-1.9 dB at GEMM output. That gap is resolution (MSE), not bias.
4. **SR on dY costs 1.5-1.9 dB of GEMM SQNR** for both grids, and removes only that ~0.5% bias.
5. **E4M3 block scale vs fp32:** about -0.1 to -0.2 dB. It also adds a small extra shrinkage from clipping after scale rounding, which hits SR too (alpha about 0.99).

## E1M2 vs INT4 (`stage0/int4_vs_e1m2.py`, `stage0/int4_vs_e1m2.md`, fig7)
- Symmetric INT4 (+-7, scale absmax/7) is **bit-identical** to E1M2: max |dY| = 0 over all Linears and configs. Normalized grids are both {0..7}/7, and the scale ratio is exactly 4 = 2^2, so FP32, E4M3 and E8M0 scales all round the same way.
- Full-range INT4 (-8..7, scale absmax/8) clips +absmax from 8 to 7.
  - alpha is 0.96-0.98 (2-4% shrinkage, 5-10x larger than E2M1).
  - After RHT its SQNR still beats E2M1: fwd X 21.86 vs 20.19 dB; FPROP Y 20.73 vs 20.23 dB.
  - A grid with far more bias wins on SQNR, which is further evidence that the gap is resolution, not bias.

## If not shrinkage bias, what? (`stage0/cause_probe.py`, `stage0/cause_probe.md`, fig8/fig9)
1. **Bias is <1% of the error.** Splitting ||e||^2 = (1-alpha)^2 ||A||^2 + ||e_perp||^2, the bias^2 share for E2M1 is 0.1-1.1% per operand and 0.77% on average at GEMM output (max 17.7% in one outlier layer). The other ~99% is noise (variance).
2. **The cause is level placement vs the block-normalized distribution.** Both grids have 8 magnitude levels; they differ only in placement.
   - E2M1 is dense near 0 (step 0.5) and coarse at the top (step 1, then 2).
   - Before RHT, the absmax element (exact, zero error) carries 47-59% of block energy on X and dY^T, and the other elements crowd near 0. The weighted Lloyd-Max optimum is bottom-dense, e.g. [0, .36, .87, 1.51, 2.28, 3.25, 4.45, 6] for dY^T, which is E2M1-like, so E2M1 wins.
   - After RHT, crest p90 drops from 3.0-3.6 to 2.3-2.5, the absmax energy share falls to 23-26%, and the remaining energy ramps toward t=6, into E2M1's coarsest region (4-6). The optimum becomes ~uniform, e.g. [0, .79, 1.59, 2.41, 3.26, 4.16, 5.09, 6].
3. **Distance from the optimum after RHT:** uniform is within 0.05-0.27 dB of Lloyd-Max on X, dY and dY^T; E2M1 is 2.1-3.0 dB away. 65-76% of E2M1's MSE comes from levels 4 and 6 (step-2 bins).
4. **Implication:** fix the placement (uniform grid), or keep E2M1 and avoid the 4-6 gap (e.g. a per-block absmax->4-or-6 choice, Four-Over-Six style). SR and alpha correction cannot close this gap.

## Can a modified rotation suit E2M1? (`rotation/rht_variants.py`, `rotation/rv_agg.py` -> `rotation/rv_tables.md`, fig10/fig11)
- Tested 13 orthogonal transforms on 100M/300M/760M; all are applied to both operands on the reduction axis.
  - Partial or wider Hadamard: H2-H128.
  - Energy compaction: DCT16, Haar16, KLT.
  - Reorder or select: PERM by RMS, ADAPT per-block I/H16, PERM+H16.
  - SIGN16: best of 16 random sign vectors for H16.
  - Data-dependent ones are fitted on a calibration batch and evaluated on another batch. WGRAD is fitted online, since its reduction axis is tokens.
- **No rotation beats no rotation for E2M1.** More mixing means more MSE in levels 4/6 (36% -> 69% from I to H16 at 300M) and lower SQNR, almost linearly (fig10).
  - DCT behaves like H16 and Haar like H2-H4: channel order has no smoothness to compact.
  - KLT costs -0.7 to -2.7 dB.
  - PERM gives +0.2 dB on DGRAD only; ADAPT stays within +-0.07 dB of I.
- **The scale side fixes the top bins.** A per-block min-MSE choice of absmax->6 or ->4 (4/6) gives +0.57-0.70 dB without rotation.
  - Gap to UFP4 shrinks from 1.2-1.9 dB (E2M1+H16) to 0.2-0.6 dB.
  - With PERM on DGRAD, E2M1-4/6 matches or beats UFP4 (+0.0 to +0.07).
  - Adding H16 on top of 4/6 hurts (about -0.3 dB).
- absmax->3 is never chosen: its level set is a subset of the absmax->6 set.
- Fairness check: a per-block clip search (c in {1, .9, .8}) also lifts the uniform grid by +0.15-0.2 dB, so uniform+H16 stays best overall.
- Caveat: this is per-GEMM SQNR only. RHT may matter for training beyond SQNR (e.g. decorrelating WGRAD error), so "E2M1-4/6, no rotation" belongs in Stage 1 as an arm.

## Flipped E2M1 grid (`flip/flip_test.py`, `flip/flip_agg.py` -> `flip/flip_tables.md`, fig12/fig13)
- FLIP quantizes u = absmax - |x| with E2M1, giving magnitude levels {0,2,3,4,4.5,5,5.5,6}: dense near absmax, a 0->2 hole near zero.
- **FLIP alone loses 4-6.5 dB.** 22-25% of elements have t<1 even after RHT, and MSE counts every element equally.
- **A per-block choice helps.** With H16, 13-23% of blocks pick FLIP (2-7% without rotation).
- **The best E2M1 setting is a per-block min-MSE choice among {E2M1@6, E2M1@4, FLIP@6, FLIP@4} with H16.** It matches UFP4 within 0.15 dB on all three models:
  - 300M: +0.08 / -0.03 / -0.08 (FPROP / DGRAD / WGRAD)
  - 100M: +0.03 / -0.06 / -0.13
  - 760M: +0.07 / -0.03 / -0.05
  - Here RHT helps again, because FLIP blocks absorb the near-absmax mass.
- With FLIP on operand A only (B = 4/6), 300M is -0.02 / -0.10 / -0.23 vs UFP4.
- 4/6 is plain NVFP4: the choice lives in the scale value, so the only cost is the 2-candidate encode. FLIP is not: its levels are not a scaled subset of E2M1, so FP4 tensor cores cannot decode it. A scale-sign-bit flag works only in a custom kernel, because native hardware reads that bit as a sign.
- Compute cost: a flipped block decodes to sign*absmax - sign*q*s, which is the sum of two NVFP4 tensors (codes +-1 with scale absmax on flipped blocks; codes +-q with scale s). That means 2 FP4 GEMMs if one operand flips, 4 if both, plus the memory for the extra tensor. Two FP4 GEMMs cost about one FP8 GEMM, and FP8 is about 12.5 dB better (Gaussian, same 1x16 block scale: 33.0 vs 20.5 dB), so FLIP is impractical on native FP4 hardware. A LUT-decoding custom kernel has the same issue: it must expand codes to FP8/BF16 for the MMA, so it runs at FP8 speed (half of FP4 on Blackwell). That gives no gain for compute-bound training GEMMs. It is only near-free when memory-bound (weight-only inference), where any codebook (e.g. NF4) could be used anyway.
- FLIP@6 and E2M1@4 serve the same direction but different blocks (`flip/block_profile.py`, 300M layer 7, H16):
  - E2M1@4 is picked by 41-43% of blocks: moderately top-heavy (about 4 of 16 elements with t>2/3, 5-6 with t<1/4). MSE drops about 31%.
  - FLIP@6 is picked by 4-13% of blocks: strongly top-heavy (5.4-7.3 with t>2/3, 1.7-3.1 with t<1/4). MSE drops 39-61%.
  - Combined gain is 82-93% of the sum of the two gains. With @4 available, about 40% of FLIP blocks move to @4.
  - The only native NVFP4 options are @6 and @4: other mappings give level subsets of these. So no native option serves the strongly top-heavy blocks.
- Two options only, E2M1@6 vs FLIP@6 (`flip/block_profile2.py` -> `flip/block_profile2.json/md`, fig14; 300M, layers 0/7/14, wq/w2, 18 tensors):
  - The deciding factor is the number of near-absmax elements (t>2/3) in the block. P(FLIP) is under 4% for 3 or fewer, about 25% at 5, about 60% at 7, and 90-100% at 9 or more.
  - The curve is nearly the same with and without rotation. RHT raises the FLIP share by producing more top-heavy blocks, not by changing the curve.
  - FLIP share per tensor: 1.5-9% without rotation, 9-31% with H16.
  - MSE drop: 24-62% inside FLIP blocks; per tensor 1-5% without rotation and 5-33% with H16.
  - With H16, 0.5% of blocks have all 16 elements near absmax (Hadamard of a single outlier gives equal magnitudes). They pick FLIP about 100% of the time and are represented almost exactly.
- A per-block choice of E2M1@6 vs FLIP@6 only (no 4/6) is +0.1-0.9 dB over E2M1 but 0.1-0.7 dB below 4/6 without rotation (all models). Since 4/6 costs nothing in the GEMM, it dominates.
- Fair comparison: uniform + per-block clip (H16) is still 0.07-0.27 dB higher at 300M.

## Why do values pile up near absmax after RHT? (`synthetic/rht_norm_check.py` -> `synthetic/rht_norm_check.md`, synthetic, CPU)
- **Not the normalization constant.** H/sqrt(16), un-normalized H and 0.1*H give identical t = |x|/absmax (max diff <= 1.6e-15) and identical E2M1 SQNR: block absmax scaling cancels any scalar.
- **It is Hadamard's flatness.** All entries have equal magnitude, and the transform preserves energy, so one large input is copied with equal magnitude into all 16 outputs.
  - Gaussian data is unchanged by any rotation.
  - On heavy-tail or outlier data, a Haar-random 16x16 orthogonal matrix only Gaussianizes (crest about 2.07, about 3.1 near-absmax elements, FLIP choice about 9%).
  - Hadamard goes beyond Gaussian: crest 1.99-2.02, 3.5-3.9 near-absmax elements, FLIP choice 11-19%, and 0.8-1.5% of blocks with all 16 elements near absmax (0% for random orthogonal).
- This flatness helps the uniform grid (outlier channels: 24.64 vs 21.71 dB for random orthogonal) and hurts E2M1 (18.96 vs 19.91 dB). For E2M1, no rotation is still best on these data.

## Spreading outliers toward zero? Scalar on RHT vs smoothing (`synthetic/smooth_check.py` -> `synthetic/smooth_check.md`, synthetic, CPU)
- **Scaling the RHT (e.g. x 1/32) changes nothing**: identical SQNR, because the block absmax scale cancels any scalar.
- **Identity: mean(t^2) = 1/crest^2.** Spreading an outlier lowers the crest, which pushes t toward absmax. "Spread the outlier" and "keep values near zero" are the same knob turned in opposite directions; an orthogonal transform cannot do both.
- **Per-channel smoothing (SmoothQuant-style) moves outliers out without mixing.** X/s and W*s keep the product unchanged.
  - Channel outliers (1% x20): block shape is kept (mean t^2 0.216 -> 0.227, vs 0.276 for H16) and E2M1 GEMM SQNR rises from 17.85 to 23.35 dB.
  - E2M1-4/6 + smoothing reaches 24.06 dB, above uniform + smoothing + H16 (23.66).
  - Channel-scale spread (lognormal): smoothing helps E2M1 (+1.5 dB), but uniform + smoothing + H16 is best (20.84).
  - Element outliers (not channel-aligned): smoothing does not help; E2M1-4/6 without rotation (19.27) or uniform + H16 (19.59) is best.
- Caveat: synthetic data with Gaussian W. Smoothing scales are static per channel, which fits FPROP/DGRAD; WGRAD reduces over tokens and needs per-batch scales. To do: measure on real HF-model tensors.

## Open-source models (2026-10-22; `hf/`, `hf_common.py`; Qwen2.5-0.5B, OLMo-2-0425-1B, Qwen2.5-1.5B, SmolLM2-1.7B, Qwen3-1.7B)
Same probes (`stage0/stage0_probe.py`, `rotation/rht_variants.py`, `flip/flip_test.py`) run with `--hf`. Models run in bf16 with frozen weights; the input is 4096 DCLM-val tokens per batch (calib / eval), re-tokenized per model. Tables: `hf/stage0_tables.md`, `hf/rv_tables.md`, `hf/flip_tables.md`, `hf/all_layers_46.md`, `hf/all_layers_flip.md`.

**Unchanged vs lingua (all 5 models):**
- RHT hurts E2M1 (fwd X -1.0 to -1.5 dB) and helps uniform (+2.8 to +3.9 dB); weights barely move.
- E2M1 shrinkage is 0.2-0.4% per tensor; after RHT alpha moves to about 1 or above (up to 1.012).
- Alpha correction changes GEMM SQNR by at most +-0.06 dB; SR on dY costs 1.8-1.9 dB.
- No rotation clearly beats no rotation for E2M1.
- 4/6|FLIP4/6 + H16 is within about +-0.1 dB of UFP4 (except OLMo WGRAD, where both are low).

**New: PERM + 4/6 without rotation often beats UFP4.**
- Cells above UFP4: Qwen3 FPROP 20.83 vs 20.18 and DGRAD 18.45 vs 18.18; Qwen2.5-1.5B FPROP/DGRAD; SmolLM2, Qwen2.5-0.5B and OLMo DGRAD.
- PERM gains more than on lingua (up to +0.60 dB on FPROP, +0.57 on DGRAD), consistent with stronger channel outliers in these models. FPROP/DGRAD PERM is fitted on a separate calibration batch.
- OLMo-2 WGRAD: H16 hurts even uniform. E2M1 without rotation (18.34) beats UFP4 (17.88), and +4/6+PERM gives 19.88. Caveat: WGRAD PERM is fitted online on the evaluated batch.
- Uniform + per-block clip + H16 is still best or tied in most cells.

## Piling block energy onto the (exact) absmax with a shared rotation? Block-diagonal KLT (`rotation/bklt_test.py` -> `hf/bklt_tables.md`; Qwen2.5-0.5B, OLMo-2-1B)
- BKLT is a per-16-group PCA rotation fitted on a calibration batch (WGRAD online).
- **It is essentially identity.** Absmax energy share barely moves (Qwen2.5-0.5B FPROP 62.3% -> 61.9%) and SQNR stays within +-0.1 dB of no rotation for E2M1, 4/6 and uniform. PERM+BKLT is no better than no rotation.
- Why: without rotation the absmax element already holds 44-75% of block energy. That concentration comes from per-token outliers, not from channel correlation, so a rotation shared by all tokens has nothing to compact. Raw data is already the most concentrated state; H16 lowers the share to 16-25%.
- Remaining levers for E2M1 are not rotations: 4/6 scale choice, PERM, smoothing.

## Outlier handling without global RHT (`scaling/outlier_test.py` -> `hf/outlier_tables.md`; Qwen2.5-0.5B, OLMo-2-1B, Qwen3-1.7B)
- Channels are scored by rms(A_col)*rms(B_col) and sorted (PERM). FPROP/DGRAD use a calibration batch; WGRAD is online.
- **RHT on the outlier blocks only (oRHT, top 1-3%) hurts E2M1** in every model and GEMM: Hadamard flattens even inside the outlier blocks.
- **Outlier channels in FP8, rest E2M1-4/6, no RHT** beats UFP4 by +0.4-2.0 / +0.2-0.8 / +1.1-3.9 dB (FPROP / DGRAD / WGRAD), with top 1% in FP8. Examples:
  - Qwen3-1.7B: 22.20 / 18.78 / 27.31 vs UFP4 20.18 / 18.18 / 24.21.
  - FP8 is within 0.15 dB of BF16 for the outliers. Going to 3% adds 0.3-0.7 dB.
  - Cost: about 1% of K in an FP8 GEMM (about 2% of FP4 FLOPs-equivalent), plus a gather.
- **SMOOTH (real tensors) + E2M1-4/6 with no extra GEMM** also beats UFP4 on FPROP and WGRAD in all three models. DGRAD is roughly on par (OLMo -0.2 dB).
- Caveat: WGRAD scores and smoothing are fitted online on the evaluated batch.

## Perturbation-based FP8/FP4 selection (`scaling/sens_test.py` -> `hf/sens_tables.md`; Qwen2.5-0.5B, FPROP W4A4, held-out loss)
- Baselines: bf16 2.7749, all-FP8 2.7815, all-FP4 2.9265.
- **Layer level: perturbation works.** Measure dL from making one Linear FP4 (calibration set), then keep the most sensitive Linears in FP8 under a parameter budget.
  - Budget 10%: 2.8698. Budget 25%: 2.8471.
  - That beats first/last layers (2.8957 / 2.8702), lowest SQNR first (2.9058 / 2.8966) and random (2.9135 / 2.8901).
  - Equal-energy Gaussian noise gives a similar ranking (2.8686 / 2.8535; Spearman 0.43 vs the real-FP4 dL).
  - Layer SQNR is unrelated to loss sensitivity (Spearman 0.02).
- **Channel level: criterion barely matters.** With the top k% K-channels per Linear in FP8, magnitude (rms*rms), output-perturbation and gradient-weighted perturbation are within 0.01 (k=1%: 2.874-2.881; k=3%: 2.862-2.867). All beat random by far (2.933).
- Channel 1% is about as good as layer 10%, so channel granularity is far more FP8-efficient.
- Prior work (subagent scan):
  - SNIP (arXiv 2602.01410, ASPLOS'26): noise-injection sensitivity in FP4 training, per Linear x operand, ILP, periodic re-selection. This already covers the layer-level idea.
  - FGMP (2504.14152): block-level FP8/NVFP4 by Fisher x error, PTQ.
  - TetraJet-v2 / CHON: magnitude-based FP8 channels in training.
  - Open: channel/block-level perturbation-based assignment for gradient operands (DGRAD/WGRAD), and a head-to-head comparison of criteria. Our forward-only result suggests magnitude is already enough for activations; gradients are untested.

## Learned per-channel-group transforms, model quality (`scaling/learned_eval.py` -> `hf/learned_eval_tables.md`; FPROP W4A4 PTQ, held-out BPB)
- Held-out BPB increase vs BF16 (Qwen2.5-0.5B / OLMo-2-1B / Qwen2.5-1.5B):

  | config | 0.5B | OLMo | 1.5B |
  |---|---|---|---|
  | E2M1 | +5.47% | +3.22% | +3.66% |
  | E2M1+H16 | +7.68% | +4.66% | +5.74% |
  | UFP4 | +4.63% | +3.09% | +3.23% |
  | E2M1-4/6 | +4.79% | +3.08% | +3.33% |
  | **E2M1-4/6 + learned group diag** | **+3.18%** | +2.02% | **+1.93%** |
  | E2M1-4/6 + learned group inv | +3.24% | **+1.90%** | +2.24% |

- E2M1-4/6 + learned diag cuts the UFP4 loss by 30-40%.
- The full 16x16 inverse adds nothing over the diagonal, so the gain is per-channel scaling (learned smoothing, close to OmniQuant LET).
- Calibration cost: diag 18-57 min per model while sharing the GPU; inv about 2x that.

## FP4-trained model + transform added only after training (`stage1/post_t.py` -> `stage1/runs/post_t.json`)
- e2m1_rtn (trained with plain E2M1 GEMMs): val 3.7883 with its own E2M1 inference.
  - Adding T after training hurts: +0.002 (4/6) to +0.013 (smoothing); learned diag +0.011; UFP4 +0.011.
  - The model has adapted to the training-time quantizer.
- BF16-trained model, PTQ: E2M1 +0.031; E2M1-4/6 + learned diag +0.024 (best); UFP4 +0.025.
- FP4 training itself costs more than FP4 inference here: e2m1_rtn under BF16 inference is 3.7747 vs 3.7238 for the BF16 model. So transforms must act during training (Stage-1 arms e2m1_smooth, e2m1_46_smooth).

## Byte-level vs subword robustness (`byte/byte_vs_subword.py` -> `hf/bvs_tables.md`; 500-byte windows, bs1, BPB)
- BPB increase, FP8 / E2M1 / E2M1-4/6 / UFP4:
  - BLT-1B: +0.17 / +4.75 / +4.24 / +7.36%
  - OLMo-2-1B: +0.25 / +5.66 / +5.06 / +5.68%
  - Qwen2.5-1.5B: +0.18 / +7.14 / +6.22 / +6.51%
  - Qwen2.5-0.5B: +0.26 / +8.68 / +7.92 / +7.40%
  - SmolLM2-1.7B: +0.32 / +13.15 / +12.52 / +16.52%
- BLT is the most robust under E2M1, but weak under UFP4. Its byte-level local encoder/decoder alone goes from +2.55% (E2M1) to +5.64% (UFP4); the global transformer alone is +1.97% / +1.58%.
- Confounded by data, training tokens and size (e.g. SmolLM2 at about 11T tokens is the most fragile). A matched byte/subword pair is needed to attribute this to byte-level modelling.

## SmoothQuant / LET / foldable scales, backward smoothing (`scaling/compare_smooth.py`, `scaling/compare_smooth2.py`, `scaling/bwd_smooth.py`)
- BPB increase vs BF16, E2M1 / E2M1-4/6:

  | config | Qwen2.5-0.5B | OLMo-2-1B |
  |---|---|---|
  | SmoothQuant, best alpha | +4.24/+3.64 | +3.39/+3.32 |
  | DIAG per-Linear | +3.55/+3.15 | +2.31/+1.82 |
  | DIAG+shift | +3.66/+2.98 | +5.70/+5.37 |
  | LET (block, shared, shift) | +3.28/+2.86 | +5.48/+5.24 |

  - Shift is unstable across models: it helps Qwen slightly and hurts OLMo badly. A midrange init blows up (+151%). Recommendation: no shift.
  - SmoothQuant on OLMo is worse than no transform (E2M1 +3.22%).
- Foldable (Qwen2.5-0.5B, shared + GQA tie), E2M1 / E2M1-4/6:
  - DIAG fold: +4.02 / +3.22%
  - LET-noshift fold: +3.65 / +2.96%
  - Both beat UFP4 (+4.63%).
- Real folding into RMSNorm / v_proj / up_proj weights:
  - BF16 BPB 0.9183 -> 0.9184 (same function).
  - Plain E2M1 on folded weights: 0.9536 vs explicit transform 0.9553.
  - Plain E2M1-4/6 on folded weights: 0.9488 vs explicit transform 0.9479.
  - So the scales cost zero inference overhead.
- Backward GEMMs (Qwen2.5-0.5B; forward fixed at E2M1; one backward GEMM quantized; weight-gradient relative error):
  - DGRAD: none 0.241, H16 0.261, UFP4 0.235, smooth 0.227 (with 4/6: 0.206), learned 0.227.
  - WGRAD: none 0.097, H16 0.116, UFP4 0.096, smooth 0.073 (with 4/6: 0.067), learned-alpha 0.072.
  - Per-call closed-form smoothing is enough for backward; most WGRAD Linears pick alpha about 0.4.

## Relation to MR-GPTQ (ICLR 2026) Lemma 2 (`synthetic/crossover_check.py` -> `synthetic/crossover_check.md`; Laplace synthetic)
- Their analysis:
  - Remark 2 / Lemma 1: absmax scaling keeps the top element exact; Hadamard spreads its error over the block.
  - Lemma 2: comparing the mass that escapes the dead zone (|u| < q_min/2), Laplace (native) beats Normal (rotated) at small block size G, and Normal wins at large G. So transforms hurt at small G and help at large G (crossover).
- Crossover G is grid-dependent: E2M1 crosses at G=32 (so NVFP4 G=16 is on the 'rotation hurts' side), while the uniform grid already wins at G=4. This explains the UFP4 observation.
- Error source at G=16 (native -> rotated):
  - E2M1: dead zone 6.8% -> 1.8%; top two bins 47.7% -> 66.6%.
  - Uniform: dead zone 25.0% -> 12.2%; top two bins 4.4% -> 10.4%.
- Reading: Lemma 2's dead-zone mechanism drives the uniform grid's gain (q_min = 1/7). E2M1's loss comes from its coarse top bins plus losing exact top-element preservation (their Remark 2), since its dead zone is small (q_min = 1/12).

## Foldable scales and LET ablation, both models (`scaling/compare_smooth2.py`; BPB increase vs BF16, E2M1-4/6, E2M1 in parentheses)

| variant | Qwen2.5-0.5B | OLMo-2-1B |
|---|---|---|
| DIAG per-Linear | +3.15 (+3.55) | **+1.82** (+2.31) |
| DIAG shared | +3.23 | +1.89 |
| DIAG fold | +3.22 | +1.89 |
| DIAG shared, SQ init | +3.03 | +2.07 |
| LET-noshift unshared | +3.10 | +2.24 |
| LET-noshift fold | **+2.97** | +2.13 |
| SmoothQuant | +3.57 | +4.06 |
| UFP4 | +4.63 | +3.09 |

- LET and DIAG apply the same operation; only the learned values differ. Qwen's LET gain comes mostly from the SmoothQuant init. On OLMo both the SQ init and the block objective hurt.
- Differences of 0.1-0.3pp flip sign across models, so they look like optimisation and init effects, not a better method.
- Recommendation: layer-wise objective, init s=1, shared (foldable) scales. Sharing costs 0.07-0.08pp.

## Inference latency on Blackwell (snu233, RTX PRO 6000 Blackwell Max-Q, sm_120; latency/)
- GEMM only (`torch._scaled_mm`, cuBLAS NVFP4 = E2M1 + E4M3 1x16 scales), per-layer Linear sum, speedup vs BF16:
  NVFP4 1.6-3.4x for M>=512 (FP8 1.3-1.7x); at M<=32 NVFP4 is slower than BF16 for 1.3B/1.5B shapes (0.67-0.80x), 1.2-1.6x for 8B (FP8 ~2x).
- Full forward (random-weight Llama-style decoder, CUDA graph, BF16 KV cache, embedding/lm_head BF16):
  prefill 2k: 8B NVFP4 1.98x (4/6 2.14x, H16 1.79x, GEMM-only 2.61x; FP8 1.46x); 1.3B 1.74x; Qwen1.5B-shape 1.36x (4/6 1.56x).
  decode: 8B B=1 1.50x (FP8 1.67x), B=32 1.35x; small models 0.82-0.92x at B=1, ~1.0-1.15x at larger B (KV-cache attention dominates).
- Overheads: H16 costs 5-10% of prefill vs plain quant; 4/6 measured <= plain (unfused torch.compile artifact); folded smoothing is free.
- Caveats: unfused activation quant (upper bound), no decode-specialized FP4 GEMV kernel, unswizzled scales (timing only), 300 W Max-Q.

## 8B W4A4 PTQ (ptq8b/; Llama-3-8B, Llama-3.1-8B, Qwen3-8B; WikiText-2 ppl + 0-shot Avg5)
- RTN, NVFP4: learned foldable diagonal (+4/6) is best on all 3 (6.69 / 6.76 / 10.04-10.08), beats SQ+4/6 by 0.07-0.12; rotations (E2M1+H, UFP4) worst.
  NVFP4-G32 (E4M3 scales, G=32): same ordering, rotation still worst -> block size alone does not flip the sign.
- MXFP4 (E8M0 scales): rotation helps plain E2M1 (8.31 -> 7.78); diag+H best on all 3 (7.21 / 7.21 / 10.75) -> the scale format (power-of-two
  absmax mismatch), not G, decides whether rotation helps; diagonal and rotation are complementary there.
- GPTQ (131k-token Hessian, no act-order): large gains for rotated methods; NVFP4 UFP4+GPTQ best on the Llamas (6.63 / 6.70) vs diag+4/6+GPTQ
  (6.66 / 6.74); Qwen3 diag+4/6+GPTQ best (10.02 vs 10.13). GPTQ on un-rotated, un-scaled E2M1 is unstable on Llama-3 (6.89 -> 8.77); diagonal
  scaling stabilises it. 8k-token GPTQ (first attempt) overfit: rank-deficient Hessian for K=14336 (held-out down_proj err 0.0169 vs RTN 0.0089).
- Stage 1 update: dY SR helps every arm (~0.02); E2M1-4/6+smooth+SR (+0.039, 2 seeds) == UFP4 (+0.038, 2 seeds).

## Official MR-GPTQ baseline (FP-Quant d2e3092, our eval protocol; ptq8b/mrgptq/)
- MR-GPTQ (GPTQ + act-order + MSE scales + block Hadamard, fineweb-edu 128x2048): NVFP4 ppl 6.871 / 6.892 / 10.235 (Avg5 71.4 / 71.8 / 69.9),
  MXFP4 (H128) 7.177 / 7.223 / 10.530 (Avg5 69.9 / 70.3 / 69.7); H32 variant similar.
- Ours beats it on ppl in every cell: NVFP4 diag+4/6+GPTQ 6.659 / 6.736 / 10.015 (RTN diag+4/6 6.69 / 6.76 / 10.08 also beats it);
  MXFP4 diag+4/6+H+GPTQ 7.040 / 7.083 / 10.310. Avg5 equal or up to +0.6pp (noise-level).
- New variant diag+4/6+H: no gain over diag+4/6 under E4M3 scales (NVFP4), best under E8M0 (MXFP4).
- Caveat: MR-GPTQ uses a static calibrated FP32 activation tensor scale; ours is dynamic. Single seed.

## Motivation checks for channel scaling (report section 22; ana/, ptq8b/records.json, hf/olmo_fold.json, stage1/runs/stage1_summary_m7.md)
- Error anatomy (8B, layer-group means): learned diagonal leaves X/W tensor NMSE ~unchanged (X +1-5%) but cuts GEMM output NMSE by 10-15%;
  rotation removes dead-zone error but pushes it into the 2-6 bins (+3-5% output NMSE on NVFP4); MXFP4 activation error is 21-25% clipping
  (OCP floor exponent), only rotation reduces it (-21..26% output NMSE). corr(log s_learned, log s_SQ) = 0.19-0.27.
- Loss sensitivity (single-group quantization, 48 windows): learned -37..43% total dNLL vs none, rotation +15..21%; concentrated in mlp_in/down
  and the last quarter of layers.
- Seeds (3): NVFP4 diag+4/6 6.692/6.747/10.068 (+-0.01-0.04) vs SQ+4/6 6.767/6.827/10.153; with GPTQ UFP4 6.636/6.711/10.100 vs ours
  6.668/6.736/10.034. Avg5 ties.
- Baselines: AWQ-form per-group alpha search == learned diagonal (6.691 vs 6.694, 6.758 vs 6.760; Qwen3 within noise) -> the ingredient is
  per-layer selection by quantized output error, not gradient learning. LET with shift collapses (Qwen3 10.74; MXFP4 all models).
- Format-aware STE: no gain, worse on MXFP4 (smooth fp32 surrogate is better).
- Shift: damage comes from atypical tokens (attention sink: first-token block absmax x8, error x7-10). Llama-3: skipping the shift on token 0
  recovers 6.81 -> 6.73 (no shift 6.69). Qwen3: several sink tokens + low-norm frequent tokens; dropping the shift 10.74 -> 10.16.
- OLMo-2 (post-norm): one global residual-stream scale is foldable (embed + post-norm weights + one op before the final norm) and matches
  per-layer scales (+2.41% vs +2.42% BPB). Non-foldable in-kernel x/s costs within noise (-4..+6%) vs +20..34% for a fused H16 (RTX PRO 6000).
- FP4 training (66M, 3 seeds): delayed-statistics smoothing == oracle (+0.0018 +- 0.0065), online-learned s no gain and +48% time;
  UFP4 ahead by 0.0013 +- 0.0005 paired.

## Open question for Stage 1
The only way shrinkage could still matter is a *systematic* alpha_Y of about 0.994 per GEMM, repeated through the backward chain across layers. A per-tensor alpha correction removes it for free (FP32 epilogue scale), so the Stage-1 arm "E2M1 + full RHT + alpha correction" directly tests whether it affects loss.

## Figures (`figs/`, regenerate with `report/make_figs.py`)
- `fig1_sqnr_rht.png`: per-operand tensor SQNR before and after RHT (300M). E2M1 drops and uniform rises on X, dY and dY^T; weights do not move.
- `fig2_alpha_per_layer.png`: per-Linear E2M1 alpha, without vs with RHT (300M). Shrinkage is at most 0.5% without RHT; with RHT, alpha spreads to 1.035.
- `fig3_db_budget.png`: GEMM-output dB budget. Alpha correction gives about 0 dB, uniform grid +1.25 to +1.85 dB, SR on dY -1.8 dB (all four checkpoints shown).
- `fig4_level_bias.png`: E2M1 per-level conditional bias and mass x bias (fwd X, 300M). The bins at 2 and 4 shrink, and the top bin (6) offsets them. Count-weighted b_G is about -0.016, while the energy-weighted alpha after RHT is about 1.000.
- `fig5_across_models.png`: the same picture for 100M / 300M / 760M-early / 760M.
- `fig6_synthetic_outliers.png`: synthetic heavy-tail and outlier distributions. Outliers + RHT flip the E2M1 bias to expansion (alpha up to 1.036).

## Files
- `fp4lib.py`: quantizers and metrics.
- `stage0/stage0_probe.py`: the hook-based probe.
- `stage0/stage0_agg.py`: table aggregation.
- `stage0/s0_*.jsonl`: raw records.
- `synthetic/sim*.py`: earlier synthetic-distribution simulations; `synthetic/sim3_results.json` feeds fig6.
- `stage0/cause_probe.py` -> `stage0/cause_probe.json/md`: block-normalized distributions, crest factor, weighted Lloyd-Max optimum, per-level MSE source.
- `rotation/rht_variants.py` -> `rotation/rv_*.jsonl`; `rotation/rv_agg.py` -> `rotation/rv_tables.md`: rotation variants and scale-side variants.
- `flip/flip_test.py` -> `flip/flip_*.jsonl`, `flip/flipA_llama300M.jsonl`; `flip/flip_agg.py` -> `flip/flip_tables.md`: flipped-grid study.
- `report/make_figs.py`: all figures (Korean labels; font in `fonts/`).
- `report/report_template.html` + `report/build_report.py` -> `report.html` (single file, figures embedded).
