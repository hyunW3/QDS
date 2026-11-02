# Is it really shrinkage bias? Re-examining why rotation hurts E2M1 (NVFP4)

A reproduction and extension study of the claim in arXiv 2606.20381 (UFP4) that a random Hadamard transform (RHT)
hurts the E2M1 FP4 grid because it amplifies *shrinkage bias*, and that a uniform 4-bit grid (UFP4) avoids it.

**TL;DR**

- The SQNR effect reproduces: with RHT, E2M1 loses 0.5-1.8 dB on activations/gradients while the uniform grid gains 1.5-3 dB.
- The cause is **not** shrinkage bias. Bias is under 1% of the quantization error (alpha = 0.992-0.999 at GEMM output),
  and a free per-tensor 1/alpha correction changes GEMM SQNR by at most +-0.06 dB.
- The cause is **level placement**. E2M1 is coarse between 4 and 6, and Hadamard flattens each 16-block so that energy
  piles up right there (65-76% of E2M1's MSE after RHT comes from levels 4 and 6). Rotations of any kind
  (partial/wider Hadamard, DCT, Haar, KLT, block-KLT) do not help E2M1.
- Fixes that keep the native NVFP4 format: per-block **4/6 scale choice** (+0.6-0.7 dB, no GEMM cost), channel
  permutation, and **foldable per-channel scaling** (learned diagonal / AWQ-style search). On 8B W4A4 PTQ
  (Llama-3-8B, Llama-3.1-8B, Qwen3-8B), learned diagonal + 4/6 beats rotation baselines and the official MR-GPTQ under NVFP4.
- Under MXFP4 (E8M0 scales) rotation helps again: the scale format, not the block size, decides the sign.

The full write-up with every number is in [`summary.md`](summary.md); per-experiment tables are the `*_tables.md` /
`*.md` files next to each script.

## Layout

Run every script from the repository root (e.g. `python rotation/rht_variants.py ...`). Scripts resolve their inputs and
outputs relative to the root, so results of one study can be read by the aggregation and figure scripts of another.

| Path | What |
|---|---|
| `fp4lib.py` | Shared: FP4 block-quantization emulation (E2M1 / uniform grids, 1x16 blocks, fp32/E4M3 scales, RTN/SR, RHT) and metrics |
| `hf_common.py` | Shared: Hugging Face models behind the probe interface (frozen bf16 weights, dY for every Linear) |
| `moe_tap.py` | Shared: per-expert operand capture for MoE layers |
| `stage0/` | Stage 0 on lingua checkpoints: per-operand / per-GEMM SQNR, alpha, bias for FPROP / DGRAD / WGRAD (`stage0_probe.py`, `s0_*.jsonl`), bias vs noise and Lloyd-Max optimum (`cause_probe.py`), INT4 = E1M2 (`int4_vs_e1m2.py`) |
| `synthetic/` | CPU checks on synthetic data: RHT normalization, smoothing, MR-GPTQ Lemma 2 crossover, 4-bit format zoo, early simulations |
| `rotation/` | Rotation variants (H2-H128, DCT, Haar, KLT, PERM, ADAPT, SIGN; `rht_variants.py`), KLT / block-KLT, decorrelation, Delta_FP decomposition (`dfp_test.py`), block size G, weight dithering |
| `flip/` | Flipped E2M1 grid and per-block grid / scale choice (`flip_test.py`, `block_profile*.py`, `all_layers_choice.py`) |
| `scaling/` | 4/6 and level placement (`placement_test.py`, `scale46_test.py`, `scale_mech.py`), outlier channels in FP8, perturbation-based FP8/FP4 selection, learned / foldable per-channel scaling, SmoothQuant, LET, backward-GEMM smoothing |
| `ptq8b/` | 8B W4A4 PTQ (`ptq8b_b.py`: RTN / GPTQ, NVFP4 / NVFP4-G32 / MXFP4), error anatomy and sensitivity (`ana8b.py`, `cf_ana.py`, `conc_ana.py`, `dist_ana.py`, `shift_ana.py`, `gsize_8b.py`), MR-GPTQ and WUSH baselines, result tables. `ptq8b.py` and `ptq8b_v*.py` are earlier iterations kept for provenance |
| `ana/` | Results of the 8B analyses |
| `moe/` | OLMoE-1B-7B (MoE): `moe_ptq.py` and results |
| `byte/` | Byte-level (BLT-1B) vs subword open models under FP4 (`byte_vs_subword.py` -> `hf/bvs_*`) |
| `stage1/` | Stage 1: small-scale FP4 *training* (66M Llama, 3 seeds) with each arm (`train.py --arm ...`) |
| `latency/` | NVFP4 GEMM and end-to-end latency on Blackwell (`torch._scaled_mm`, sm_120) |
| `hf/` | Raw records and tables of the open-model runs (Qwen2.5-0.5B/1.5B, OLMo-2-1B, SmolLM2-1.7B, Qwen3-1.7B) |
| `report/`, `figs/` | Figure scripts (`make_figs.py`, `comparison_figs.py`, `fig_tnorm.py`, `mrgptq_fig.py`), the Korean single-file HTML report (`report_template.html` + `build_report.py`), and the figures |
| `remote_runs/<host>/<dir>/` | Verbatim copies (results, run scripts, job queues) of the PTQ / MoE / MR-GPTQ / WUSH runs done on other machines; see below |
| `literature/` | Notes on related work |

## Setup

Python 3.12, CUDA 12.8. The numbers were produced with torch 2.7.0+cu128 and transformers 5.14.1.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
source ptq8b/env.sh          # only if torch cannot find its pip-installed CUDA libraries (VENV=... to point elsewhere)
```

Paths are taken from environment variables:

| Variable | Used for | Default |
|---|---|---|
| `DCLM_VAL` | Held-out text (`.jsonl`, one `{"text": ...}` per line); we used a DCLM-baseline-1.0 validation split | `data/dclm_val.jsonl` |
| `LINGUA_ROOT` | Only for the lingua-checkpoint experiments (Stage 0 `--ckpt`, `flip/`, `rotation/rht_variants.py`): a directory holding a [Meta Lingua](https://github.com/facebookresearch/lingua) checkout as `lingua/` and the Llama-3 tokenizer as `tokenizer/llama3/tokenizer.model` | `lingua_root` |
| `LINGUA_CKPTS` | Consolidated lingua Llama checkpoints `llama100M/`, `llama300M/`, `llama760M/` (read by `stage0/cause_probe.py`, `stage0/int4_vs_e1m2.py`, `flip/block_profile*.py`, `flip/all_layers_choice.py`; the probes take `--ckpt`) | `checkpoints` |
| `DCLM_DIR` | DCLM-baseline-1.0 shards that `stage1/prep_data.py` tokenizes for Stage 1 training | `data/dclm_baseline_1.0_2shards_shuffled` |
| `PY` | Python used by the `*.sh` runners | `python` |

The experiments on open Hugging Face models (`--hf <model id>`) only need `DCLM_VAL`. The experiments on our own
lingua Llama checkpoints (100M / 300M / 760M; `--ckpt`, `stage0/stage0_probe.py` without `--hf`, ...) need checkpoints that are
not part of this release.

## Running

Examples (outputs go next to the existing results, so the aggregation scripts pick them up):

```bash
# Stage 0 probe on an open model: per-operand and per-GEMM SQNR / alpha / bias
python stage0/stage0_probe.py --hf Qwen/Qwen2.5-0.5B --tag qwen2.5-0.5b --out hf/s0_qwen2.5-0.5b.jsonl

# Rotation and scale-side variants, flipped grid
python rotation/rht_variants.py --hf Qwen/Qwen2.5-0.5B --tag qwen2.5-0.5b --out hf/rv_qwen2.5-0.5b.jsonl
python flip/flip_test.py        --hf Qwen/Qwen2.5-0.5B --tag qwen2.5-0.5b --out hf/flip_qwen2.5-0.5b.jsonl

# Synthetic checks (CPU)
python synthetic/rht_norm_check.py; python synthetic/smooth_check.py; python synthetic/crossover_check.py

# 8B W4A4 PTQ: one method per process (see the docstring of ptq8b/ptq8b_b.py for the method list)
python ptq8b/ptq8b_b.py --hf meta-llama/Meta-Llama-3-8B --tag llama3-8b --method diag_46 --fmt nvfp4 --out ptq8b/out/llama3-8b_nvfp4_diag_46.json

# Stage 1 FP4 training arm (data prepared by stage1/prep_data.py)
python stage1/train.py --arm e2m1_46_smoothdelay_sr --seed 0 --out stage1/runs/e2m1_46_smoothdelay_sr_s0 --steps 6000
```

Aggregation scripts (`*_agg.py`, `ptq8b/make_r*.py`, `stage1/agg_m7.py`) turn the raw records into the `*_tables.md` files;
`report/make_figs.py` and `report/comparison_figs.py` regenerate the figures.

## Method comparison tables: where they come from

| Table | Produced by | Results in this repo |
|---|---|---|
| Bias vs noise | `stage0/` (`stage0_probe.py`, `cause_probe.py`, `int4_vs_e1m2.py`) | Raw records present; rerunning needs the lingua checkpoints (not released). Open-model values: `hf/s0_*.jsonl` |
| Rotation gain decomposition, levers | `rotation/dfp_test.py` -> `hf/dfp_*.jsonl`, pooled mean as in `report/comparison_figs.py` | Present; numbers reproduce exactly |
| Fixes (GEMM SQNR) | `scaling/placement_test.py` -> `scaling/place_agg.py` -> `hf/place_tables.md`, `moe/place_tables.md` | Present |
| 4/6 mechanism | `scaling/scale46_test.py` -> `scaling/s46_agg.py` -> `hf/s46_tables.md` | Present |
| 8B W4A4 PTQ, QDS x 4/6 | `ptq8b/ptq8b_b.py` (`--rht`, `--gptq`, `--actorder`, `*_opt` methods), `ptq8b/wush/`, MR-GPTQ via `remote_runs/info107/mrgptq/mrgptq_run.py` | `ptq8b/results*/`, `ptq8b/records.json`, `ptq8b/mrgptq/`; the RHT16, UFP4, WUSH, `*_opt` and GPTQ+ (`*_Bao`) rows are in `remote_runs/<host>/fp4_ptq/out/` and `remote_runs/info107/wush/out/` |
| OLMoE | `moe/moe_ptq.py` | `moe/ptq_*.json`; RHT16 variants in `remote_runs/{asri139,snu233}/moe_ptq/out/` |
| FP4 pre-training | `stage1/train.py` -> `stage1/agg_m7.py` | `stage1/runs/*/log.jsonl`, `stage1/runs/stage1_summary_m7.md`; numbers reproduce |

Every PTQ number in these tables is present in the repo. The runs on other machines are kept as copied in
`remote_runs/<host>/<dir>/` (`fp4_ptq`, `fp4_omni`, `moe_ptq`, `mrgptq`, `wush`), including the exact job queues
(`queue*.txt`, `q_*.txt`) and the script versions that ran there; their shell scripts still `cd` into the original
working directories. `ptq8b/results/<host>/` holds earlier pulls of the same `fp4_ptq/out/` directories
(`ptq8b/collect2.py`). MR-GPTQ needs an [FP-Quant](https://github.com/IST-DASLab/FP-Quant) checkout
(`remote_runs/info107/mrgptq/mrgptq_run.py`).

## Not included

- Model checkpoints (Stage 1 `final.pt`, lingua checkpoints), training data, and the DCLM validation text.
- Third-party code used for the baselines: [FP-Quant](https://github.com/IST-DASLab/FP-Quant) (MR-GPTQ) and
  [WUSH](https://github.com/IST-DASLab/WUSH). `ptq8b/wush/wush_run.py` and `remote_runs/info107/mrgptq/mrgptq_run.py` expect
  these checkouts on `PYTHONPATH`.
- Run logs, and the Noto Sans CJK font that `report/make_figs.py` / `report/build_report.py` use for Korean labels: put any
  `*.otf`/`*.ttf` CJK font (e.g. `NotoSansCJKjp-Regular.otf`) in `fonts/` to rebuild the report.

## Caveats

- Stage 0 numbers are per-GEMM SQNR from a single forward+backward pass; Stage 1 training is small scale (66M).
- 8B PTQ results are mostly single-seed (3 seeds for the main NVFP4 comparison); differences of 0.1-0.3 pp flip sign
  across models and should be read as noise.
- Latency numbers use unfused activation quantization (an upper bound on overhead).
