# Literature for the FP4 / RHT / scaling study

## Notes
- `literature_FP4_RHT.md` — NVFP4 rotation literature: NVIDIA NVFP4 pre-training recipe, MR-GPTQ, ARCQuant, nGPT, Transformer Engine / MaxText RHT placement (user note).
- `literature_FP4_scale.md` — NVFP4 scale-side literature: SOAR, ScaleSweep, FOCUS, and the unified PTQ evaluation (Liu et al. 2025) that finds rotation helps INT4 but not MX/NVFP4 (user note).
- `related_survey_notes.md` — check of the transform survey (arXiv 2608.25188) and Ordentlich & Polyanskiy (2601.17187, 2605.13768) against our results, plus the Delta_FP-vs-placement and outlier-block results (`dfp_test.py`).

## Papers (PDFs are not redistributed here; look them up by arXiv id)
- `2608.25188_jokar_transforms_survey.pdf` — Transforms for LLM Quantization: The Great Inversion and Format Co-Design.
- `2601.17187_ordentlich_highrate_matmul_I.pdf` — High-Rate Quantized Matrix Multiplication I (Delta_FP, IEIN model, FP8 experiment).
- `2605.13768_ordentlich_highrate_matmul_II.pdf` — Part II (weight-only, WaterSIC).
- `2512.02010_cook_four_over_six.pdf` — Four Over Six: near-maximal values (scaled ~5, between 66.6% and 100% of the block max) are NVFP4's main error source; 4/6 block scaling; keeps WGRAD RHT in its pre-training recipe; does not relate the near-maximal problem to rotation.
- `2603.04359_federici_concentration_alignment_CAT.pdf` — CAT: concentration/alignment SQNR decomposition for uniform INT4; SmoothQuant criticized for INT4 only.
- `2507.17417_liu_comprehensive_quant_eval.pdf` — A Comprehensive Evaluation on Quantization Techniques for LLMs (Tables 15-16: NVFP4 rotation vs scaling).

