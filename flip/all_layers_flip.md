
## llama100M (98 Linears, 14 layers)

### I: E2M1|FLIP 2nd-option block share A 4.3%, B 8.5%, all 5.6%
- fprop: GEMM NMSE 9.39 → 9.24 (-1.6%; 20.27 → 20.34 dB), A operand 8.11 → 7.93 (-2.2%)
- dgrad: GEMM NMSE 13.71 → 13.36 (-2.6%; 18.63 → 18.74 dB), A operand 8.19 → 7.98 (-2.6%)
- wgrad: GEMM NMSE 12.55 → 11.31 (-9.9%; 19.01 → 19.46 dB), A operand 7.13 → 7.07 (-0.9%)
- all: GEMM NMSE 11.89 → 11.31 (-4.9%; 19.25 → 19.47 dB), A operand 7.81 → 7.66 (-1.9%)
- per layer GEMM NMSE change: L0 -3.4%, L1 -8.4%, L2 -6.2%, L3 -6.4%, L4 -5.0%, L5 -3.9%, L6 -3.6%, L7 -4.3%, L8 -4.3%, L9 -4.0%, L10 -4.3%, L11 -4.3%, L12 -4.4%, L13 -5.8%
### H16: E2M1|FLIP 2nd-option block share A 18.7%, B 11.3%, all 16.4%
- fprop: GEMM NMSE 10.67 → 9.33 (-12.6%; 19.72 → 20.30 dB), A operand 9.32 → 7.34 (-21.2%)
- dgrad: GEMM NMSE 15.07 → 13.71 (-9.0%; 18.22 → 18.63 dB), A operand 9.26 → 8.02 (-13.4%)
- wgrad: GEMM NMSE 14.41 → 11.92 (-17.3%; 18.41 → 19.24 dB), A operand 9.57 → 7.11 (-25.8%)
- all: GEMM NMSE 13.39 → 11.65 (-13.0%; 18.73 → 19.34 dB), A operand 9.38 → 7.49 (-20.2%)
- per layer GEMM NMSE change: L0 -14.0%, L1 -15.3%, L2 -13.2%, L3 -13.8%, L4 -12.9%, L5 -12.5%, L6 -11.9%, L7 -11.4%, L8 -12.1%, L9 -11.9%, L10 -12.8%, L11 -12.9%, L12 -13.0%, L13 -14.0%

## llama300M (105 Linears, 15 layers)

### I: E2M1|FLIP 2nd-option block share A 4.2%, B 8.5%, all 5.8%
- fprop: GEMM NMSE 9.32 → 9.17 (-1.7%; 20.31 → 20.38 dB), A operand 7.90 → 7.73 (-2.1%)
- dgrad: GEMM NMSE 13.86 → 13.52 (-2.4%; 18.58 → 18.69 dB), A operand 8.09 → 7.89 (-2.5%)
- wgrad: GEMM NMSE 12.60 → 10.88 (-13.6%; 19.00 → 19.63 dB), A operand 6.77 → 6.71 (-0.9%)
- all: GEMM NMSE 11.92 → 11.19 (-6.2%; 19.24 → 19.51 dB), A operand 7.59 → 7.44 (-1.9%)
- per layer GEMM NMSE change: L0 -6.7%, L1 -17.1%, L2 -10.2%, L3 -7.0%, L4 -4.8%, L5 -7.3%, L6 -4.3%, L7 -4.4%, L8 -4.5%, L9 -4.5%, L10 -4.3%, L11 -4.0%, L12 -3.9%, L13 -4.7%, L14 -5.4%
### H16: E2M1|FLIP 2nd-option block share A 18.7%, B 10.7%, all 15.9%
- fprop: GEMM NMSE 11.13 → 9.14 (-17.8%; 19.54 → 20.39 dB), A operand 9.66 → 6.73 (-30.4%)
- dgrad: GEMM NMSE 15.36 → 13.87 (-9.7%; 18.14 → 18.58 dB), A operand 9.21 → 7.87 (-14.5%)
- wgrad: GEMM NMSE 14.42 → 11.72 (-18.7%; 18.41 → 19.31 dB), A operand 9.37 → 6.71 (-28.3%)
- all: GEMM NMSE 13.64 → 11.58 (-15.1%; 18.65 → 19.36 dB), A operand 9.42 → 7.10 (-24.5%)
- per layer GEMM NMSE change: L0 -16.9%, L1 -16.5%, L2 -16.3%, L3 -16.3%, L4 -16.8%, L5 -15.3%, L6 -14.1%, L7 -13.8%, L8 -12.9%, L9 -14.0%, L10 -14.1%, L11 -14.1%, L12 -15.1%, L13 -15.6%, L14 -15.3%

## llama760M (168 Linears, 24 layers)

### I: E2M1|FLIP 2nd-option block share A 4.1%, B 8.8%, all 5.9%
- fprop: GEMM NMSE 9.95 → 9.79 (-1.6%; 20.02 → 20.09 dB), A operand 7.90 → 7.75 (-2.0%)
- dgrad: GEMM NMSE 13.77 → 13.46 (-2.2%; 18.61 → 18.71 dB), A operand 8.02 → 7.82 (-2.4%)
- wgrad: GEMM NMSE 12.20 → 11.63 (-4.7%; 19.14 → 19.35 dB), A operand 6.37 → 6.31 (-0.9%)
- all: GEMM NMSE 11.97 → 11.63 (-2.9%; 19.22 → 19.35 dB), A operand 7.43 → 7.29 (-1.8%)
- per layer GEMM NMSE change: L0 -6.7%, L1 -14.3%, L2 -6.5%, L3 -9.3%, L4 -2.2%, L5 -2.3%, L6 -1.9%, L7 +1.4%, L8 +4.3%, L9 +1.7%, L10 -0.4%, L11 -1.9%, L12 +0.6%, L13 -3.3%, L14 -2.3%, L15 -2.9%, L16 -3.6%, L17 -3.2%, L18 -2.6%, L19 -3.1%, L20 -2.7%, L21 -3.6%, L22 -4.6%, L23 -4.4%
### H16: E2M1|FLIP 2nd-option block share A 18.9%, B 10.5%, all 15.7%
- fprop: GEMM NMSE 11.85 → 9.78 (-17.5%; 19.26 → 20.10 dB), A operand 9.56 → 6.83 (-28.6%)
- dgrad: GEMM NMSE 15.45 → 13.78 (-10.8%; 18.11 → 18.61 dB), A operand 9.23 → 7.72 (-16.4%)
- wgrad: GEMM NMSE 15.37 → 12.25 (-20.3%; 18.13 → 19.12 dB), A operand 9.05 → 6.30 (-30.3%)
- all: GEMM NMSE 14.22 → 11.93 (-16.1%; 18.47 → 19.23 dB), A operand 9.28 → 6.95 (-25.1%)
- per layer GEMM NMSE change: L0 -21.9%, L1 -19.3%, L2 -17.5%, L3 -15.9%, L4 -17.3%, L5 -15.7%, L6 -19.5%, L7 -17.1%, L8 -13.5%, L9 -15.1%, L10 -14.5%, L11 -14.4%, L12 -13.8%, L13 -14.7%, L14 -14.6%, L15 -15.7%, L16 -15.5%, L17 -15.9%, L18 -15.6%, L19 -16.2%, L20 -16.6%, L21 -16.0%, L22 -16.3%, L23 -16.2%
