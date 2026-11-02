# Perturbation-based FP8/FP4 selection (Qwen2.5-0.5B, FPROP W4A4, eval loss on 16x1024 held-out tokens)

bf16 2.7749, all-FP8 2.7815, all-FP4 2.9265

Spearman(dL_q, dL_noise) = 0.425; Spearman(dL_q, -SQNR) = 0.023

## Layer level (FP8 budget by parameter count)

| criterion | 10% | 25% |
|---|---|---|
| dL_q | 2.8698 | 2.8471 |
| dL_noise | 2.8686 | 2.8535 |
| first_last | 2.8957 | 2.8702 |
| worst_SQNR | 2.9058 | 2.8966 |
| random_mean | 2.9135 | 2.8901 |

## Channel level (top k% input channels per Linear in FP8)

| criterion | 1% | 3% |
|---|---|---|
| mag | 2.8814 | 2.8619 |
| outerr | 2.8736 | 2.8673 |
| grad | 2.8795 | 2.8667 |
| random | 2.9336 | 2.9334 |
