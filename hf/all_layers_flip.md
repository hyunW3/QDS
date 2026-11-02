Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

## qwen2.5-0.5b (168 Linears, 24 layers)

### I: E2M1|FLIP 2nd-option block share A 2.9%, B 8.7%, all 4.8%
- fprop: GEMM NMSE 13.13 → 13.14 (+0.1%; 18.82 → 18.82 dB), A operand 7.72 → 7.65 (-0.9%)
- dgrad: GEMM NMSE 22.14 → 22.14 (+0.0%; 16.55 → 16.55 dB), A operand 7.92 → 7.78 (-1.8%)
- wgrad: GEMM NMSE 12.06 → 10.80 (-10.5%; 19.18 → 19.67 dB), A operand 6.98 → 6.93 (-0.7%)
- all: GEMM NMSE 15.78 → 15.36 (-2.6%; 18.02 → 18.14 dB), A operand 7.54 → 7.45 (-1.2%)
- per layer GEMM NMSE change: L0 -2.1%, L1 -3.9%, L2 -2.9%, L3 -1.1%, L4 -2.1%, L5 -2.9%, L6 -2.2%, L7 -0.5%, L8 -2.6%, L9 +3.5%, L10 -2.1%, L11 +0.9%, L12 -2.3%, L13 -2.8%, L14 -3.9%, L15 -4.2%, L16 -3.1%, L17 -4.9%, L18 -5.7%, L19 -5.4%, L20 -3.5%, L21 -2.0%, L22 -3.0%, L23 -4.2%
### H16: E2M1|FLIP 2nd-option block share A 20.5%, B 10.7%, all 17.3%
- fprop: GEMM NMSE 16.68 → 13.34 (-20.0%; 17.78 → 18.75 dB), A operand 9.94 → 6.46 (-35.1%)
- dgrad: GEMM NMSE 27.96 → 23.69 (-15.3%; 15.54 → 16.25 dB), A operand 9.35 → 7.59 (-18.8%)
- wgrad: GEMM NMSE 14.45 → 11.67 (-19.3%; 18.40 → 19.33 dB), A operand 9.50 → 6.80 (-28.4%)
- all: GEMM NMSE 19.70 → 16.23 (-17.6%; 17.06 → 17.90 dB), A operand 9.60 → 6.95 (-27.6%)
- per layer GEMM NMSE change: L0 -17.3%, L1 -14.0%, L2 -12.0%, L3 -16.1%, L4 -20.2%, L5 -17.8%, L6 -15.7%, L7 -26.6%, L8 -15.9%, L9 -16.4%, L10 -15.6%, L11 -14.2%, L12 -14.7%, L13 -17.5%, L14 -14.9%, L15 -15.3%, L16 -19.2%, L17 -22.0%, L18 -21.3%, L19 -21.2%, L20 -19.5%, L21 -19.6%, L22 -15.6%, L23 -15.7%

## olmo2-1b (112 Linears, 16 layers)

### I: E2M1|FLIP 2nd-option block share A 3.5%, B 8.1%, all 5.4%
- fprop: GEMM NMSE 9.74 → 9.64 (-1.1%; 20.11 → 20.16 dB), A operand 7.98 → 7.83 (-1.9%)
- dgrad: GEMM NMSE 17.47 → 17.08 (-2.2%; 17.58 → 17.67 dB), A operand 8.12 → 7.93 (-2.3%)
- wgrad: GEMM NMSE 18.50 → 26.95 (+45.7%; 17.33 → 15.69 dB), A operand 5.07 → 5.04 (-0.7%)
- all: GEMM NMSE 15.24 → 17.89 (+17.4%; 18.17 → 17.47 dB), A operand 7.06 → 6.93 (-1.8%)
- per layer GEMM NMSE change: L0 -3.4%, L1 -4.1%, L2 -4.4%, L3 +2.5%, L4 +2.2%, L5 +4.7%, L6 +2.1%, L7 -1.3%, L8 +19.9%, L9 +1.0%, L10 +24.2%, L11 +66.2%, L12 +47.3%, L13 +2.8%, L14 +78.0%, L15 +17.0%
### H16: E2M1|FLIP 2nd-option block share A 18.8%, B 10.5%, all 15.3%
- fprop: GEMM NMSE 12.00 → 9.48 (-21.0%; 19.21 → 20.23 dB), A operand 10.10 → 6.63 (-34.3%)
- dgrad: GEMM NMSE 19.19 → 17.47 (-9.0%; 17.17 → 17.58 dB), A operand 9.21 → 7.94 (-13.8%)
- wgrad: GEMM NMSE 118.68 → 97.07 (-18.2%; 9.26 → 10.13 dB), A operand 8.05 → 4.77 (-40.7%)
- all: GEMM NMSE 49.96 → 41.34 (-17.2%; 13.01 → 13.84 dB), A operand 9.12 → 6.45 (-29.3%)
- per layer GEMM NMSE change: L0 -13.5%, L1 -15.5%, L2 -19.2%, L3 -21.9%, L4 -22.3%, L5 -21.5%, L6 -22.9%, L7 -13.5%, L8 -10.6%, L9 -19.1%, L10 -21.6%, L11 -17.0%, L12 -21.3%, L13 -18.8%, L14 -13.0%, L15 -15.1%

## qwen2.5-1.5b (196 Linears, 28 layers)

### I: E2M1|FLIP 2nd-option block share A 2.8%, B 8.4%, all 5.0%
- fprop: GEMM NMSE 13.36 → 13.40 (+0.3%; 18.74 → 18.73 dB), A operand 7.52 → 7.44 (-1.0%)
- dgrad: GEMM NMSE 20.22 → 20.22 (+0.0%; 16.94 → 16.94 dB), A operand 7.81 → 7.66 (-2.0%)
- wgrad: GEMM NMSE 13.26 → 12.05 (-9.2%; 18.77 → 19.19 dB), A operand 6.76 → 6.71 (-0.7%)
- all: GEMM NMSE 15.62 → 15.22 (-2.5%; 18.06 → 18.17 dB), A operand 7.36 → 7.27 (-1.3%)
- per layer GEMM NMSE change: L0 -2.8%, L1 -2.0%, L2 -2.6%, L3 -2.8%, L4 -0.6%, L5 +1.6%, L6 -3.9%, L7 -2.9%, L8 -1.2%, L9 -6.2%, L10 -2.0%, L11 -2.3%, L12 -2.1%, L13 -0.7%, L14 -1.4%, L15 -2.8%, L16 -2.4%, L17 -2.7%, L18 -2.2%, L19 -2.8%, L20 -3.6%, L21 -4.3%, L22 -2.5%, L23 -1.0%, L24 -3.4%, L25 -2.8%, L26 -3.6%, L27 -4.6%
### H16: E2M1|FLIP 2nd-option block share A 21.5%, B 10.5%, all 17.2%
- fprop: GEMM NMSE 17.58 → 13.70 (-22.1%; 17.55 → 18.63 dB), A operand 10.15 → 6.11 (-39.8%)
- dgrad: GEMM NMSE 25.15 → 21.67 (-13.8%; 16.00 → 16.64 dB), A operand 9.32 → 7.56 (-18.9%)
- wgrad: GEMM NMSE 16.96 → 13.65 (-19.5%; 17.71 → 18.65 dB), A operand 9.55 → 6.53 (-31.6%)
- all: GEMM NMSE 19.89 → 16.34 (-17.9%; 17.01 → 17.87 dB), A operand 9.67 → 6.73 (-30.4%)
- per layer GEMM NMSE change: L0 -15.3%, L1 -12.5%, L2 -20.0%, L3 -26.3%, L4 -23.6%, L5 -20.9%, L6 -19.3%, L7 -15.3%, L8 -17.9%, L9 -21.1%, L10 -18.5%, L11 -20.8%, L12 -16.5%, L13 -17.2%, L14 -13.8%, L15 -15.1%, L16 -13.9%, L17 -15.0%, L18 -19.5%, L19 -17.4%, L20 -14.6%, L21 -15.9%, L22 -18.1%, L23 -18.9%, L24 -16.7%, L25 -18.6%, L26 -18.2%, L27 -16.6%

## smollm2-1.7b (168 Linears, 24 layers)

### I: E2M1|FLIP 2nd-option block share A 3.3%, B 8.5%, all 5.5%
- fprop: GEMM NMSE 11.72 → 11.75 (+0.3%; 19.31 → 19.30 dB), A operand 7.19 → 7.08 (-1.5%)
- dgrad: GEMM NMSE 21.47 → 21.43 (-0.2%; 16.68 → 16.69 dB), A operand 8.09 → 7.94 (-1.8%)
- wgrad: GEMM NMSE 11.57 → 12.64 (+9.3%; 19.37 → 18.98 dB), A operand 6.23 → 6.18 (-0.9%)
- all: GEMM NMSE 14.92 → 15.28 (+2.4%; 18.26 → 18.16 dB), A operand 7.17 → 7.07 (-1.5%)
- per layer GEMM NMSE change: L0 -4.2%, L1 -5.7%, L2 +6.9%, L3 +1.8%, L4 +4.0%, L5 +4.5%, L6 +4.5%, L7 +4.5%, L8 +1.5%, L9 +11.0%, L10 +7.1%, L11 +4.6%, L12 +2.6%, L13 +2.7%, L14 +1.6%, L15 +0.7%, L16 +0.7%, L17 +5.0%, L18 +0.3%, L19 +1.1%, L20 +0.8%, L21 +2.1%, L22 +0.4%, L23 -4.6%
### H16: E2M1|FLIP 2nd-option block share A 19.6%, B 10.4%, all 15.7%
- fprop: GEMM NMSE 16.53 → 11.96 (-27.6%; 17.82 → 19.22 dB), A operand 10.29 → 5.63 (-45.3%)
- dgrad: GEMM NMSE 26.22 → 22.98 (-12.4%; 15.81 → 16.39 dB), A operand 9.40 → 7.74 (-17.6%)
- wgrad: GEMM NMSE 14.87 → 11.88 (-20.1%; 18.28 → 19.25 dB), A operand 8.88 → 6.25 (-29.6%)
- all: GEMM NMSE 19.21 → 15.61 (-18.7%; 17.17 → 18.07 dB), A operand 9.52 → 6.54 (-31.3%)
- per layer GEMM NMSE change: L0 -18.3%, L1 -15.2%, L2 -19.3%, L3 -17.4%, L4 -18.4%, L5 -19.0%, L6 -19.8%, L7 -20.9%, L8 -19.4%, L9 -19.5%, L10 -17.6%, L11 -19.2%, L12 -16.6%, L13 -17.5%, L14 -17.9%, L15 -16.9%, L16 -17.9%, L17 -20.3%, L18 -19.9%, L19 -19.6%, L20 -18.2%, L21 -18.2%, L22 -20.7%, L23 -21.7%

## qwen3-1.7b (196 Linears, 28 layers)

### I: E2M1|FLIP 2nd-option block share A 3.3%, B 8.5%, all 5.5%
- fprop: GEMM NMSE 12.95 → 13.02 (+0.5%; 18.88 → 18.85 dB), A operand 7.43 → 7.35 (-1.1%)
- dgrad: GEMM NMSE 18.35 → 18.17 (-1.0%; 17.36 → 17.41 dB), A operand 7.56 → 7.41 (-2.0%)
- wgrad: GEMM NMSE 5.04 → 4.69 (-7.0%; 22.98 → 23.29 dB), A operand 5.59 → 5.54 (-0.9%)
- all: GEMM NMSE 12.11 → 11.96 (-1.3%; 19.17 → 19.22 dB), A operand 6.86 → 6.77 (-1.4%)
- per layer GEMM NMSE change: L0 -3.3%, L1 -4.2%, L2 -0.9%, L3 -2.1%, L4 -1.7%, L5 -1.8%, L6 -1.6%, L7 -1.8%, L8 -1.8%, L9 -1.9%, L10 -3.2%, L11 -2.8%, L12 -2.6%, L13 -1.5%, L14 -1.7%, L15 -1.3%, L16 -1.3%, L17 -0.7%, L18 -0.5%, L19 -2.0%, L20 +0.8%, L21 -1.4%, L22 +0.4%, L23 +0.2%, L24 +1.2%, L25 -2.3%, L26 +0.5%, L27 +3.3%
### H16: E2M1|FLIP 2nd-option block share A 23.5%, B 10.8%, all 18.2%
- fprop: GEMM NMSE 17.90 → 13.51 (-24.5%; 17.47 → 18.69 dB), A operand 10.27 → 5.83 (-43.2%)
- dgrad: GEMM NMSE 22.15 → 19.18 (-13.4%; 16.55 → 17.17 dB), A operand 9.22 → 7.28 (-21.0%)
- wgrad: GEMM NMSE 7.93 → 6.26 (-21.0%; 21.01 → 22.03 dB), A operand 7.99 → 5.56 (-30.4%)
- all: GEMM NMSE 15.99 → 12.98 (-18.8%; 17.96 → 18.87 dB), A operand 9.16 → 6.22 (-32.1%)
- per layer GEMM NMSE change: L0 -14.3%, L1 -21.7%, L2 -19.3%, L3 -23.9%, L4 -22.7%, L5 -19.3%, L6 -18.9%, L7 -19.2%, L8 -16.8%, L9 -17.9%, L10 -17.4%, L11 -17.5%, L12 -14.6%, L13 -16.2%, L14 -16.6%, L15 -16.5%, L16 -17.0%, L17 -17.9%, L18 -19.5%, L19 -21.0%, L20 -22.1%, L21 -18.8%, L22 -19.5%, L23 -20.6%, L24 -19.0%, L25 -18.9%, L26 -18.7%, L27 -17.2%
