찾아보니 **NVFP4에서 rotation을 직접 다룬 문헌은 학습용 RHT와 추론용 PTQ 연구로 나뉘고, 두 영역의 결론이 다릅니다.** 특히 학습용 결과를 NVFP4 추론 PTQ에 그대로 적용하면 안 됩니다.

### NVFP4에 직접 관련된 핵심 논문

1. **[Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)**  
   NVFP4 사전학습 레시피를 제시한 핵심 논문입니다. RHT로 블록 내 outlier를 완화하고, 2D weight quantization·stochastic rounding·일부 고정밀 연산과 함께 사용합니다. 12B 모델을 10T 토큰으로 학습한 결과를 보고합니다. 다만 RHT만의 효과를 다른 기법과 분리해 해석하긴 어렵습니다. 

2. **[Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202), ICLR 2026**  
   MXFP4와 NVFP4의 **PTQ**를 비교한 핵심 연구입니다. 저자들은 NVFP4의 작은 그룹 크기에서는 기존 outlier 완화 기법의 효과가 제한될 수 있다고 분석하고, 포맷에 맞춘 **block-wise Hadamard rotation + GPTQ**, 즉 MR-GPTQ를 제안합니다. rotation을 무조건 크게 적용하기보다 NVFP4의 16개 원소 단위 scale에 맞춰야 한다는 근거로 읽을 만합니다. 

3. **[ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs](https://arxiv.org/abs/2601.07475), ACL 2026**  
   NVFP4 PTQ 연구로, rotation 계열이 **세밀한 블록 격리를 해칠 수 있다**고 지적하고 잔차 채널을 이용한 보정을 제안합니다. rotation을 직접 제안하는 논문은 아니지만, “rotation이 NVFP4 PTQ에서 항상 이득인가?”를 검토할 때 중요한 반대쪽 근거입니다. 

4. **[Normalized Architectures are Natively 4-Bit](https://arxiv.org/abs/2605.06067)**  
   NVFP4 학습에서 정규화된 nGPT 구조가 RHT 같은 별도 개입 없이도 안정적인 4-bit 학습을 할 수 있다고 주장합니다. RHT가 유용한 처방일 수는 있지만, 모델 구조에 따라 필수는 아닐 수 있다는 비교점입니다. 

### 실제 구현을 확인할 자료

- **[NVIDIA Transformer Engine NVFP4 문서](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)** — 구현에서 RHT는 입력과 gradient의 특정 columnwise 양자화 경로, 특히 WGRAD 입력에 적용됩니다. 정규화한 Hadamard 회전과 sign flip을 사용하고, 문서상 블록 크기는 16입니다. 
- **[MaxText NVFP4 학습 예제](https://docs.nvidia.com/jax-toolbox/frameworks/max-text/nvfp-4-training-example)** — RHT 포함(`te_nvfp4`)과 미포함(`te_nvfp4_no_rht`) 설정을 비교할 수 있습니다. 문서는 수렴이 만족스럽지 않을 때 RHT 포함 설정을 권장합니다. 

### 함께 보면 좋은 배경 논문

- **[QuaRot](https://arxiv.org/abs/2404.00456)** 및 **[SpinQuant](https://arxiv.org/abs/2405.16406)**는 회전 기반 저비트 양자화의 대표 연구입니다. 다만 주된 맥락은 4-bit LLM 양자화 전반이며, 위 NVFP4 연구들과 설정이 같지는 않습니다. NVFP4 근거로 인용할 땐 직접 비교 연구인 MR-GPTQ와 구분해서 보는 게 좋습니다. 

**읽는 순서로는** NVFP4 사전학습 논문 → ICLR 2026의 MR-GPTQ 논문 → ARCQuant를 권합니다. 이 순서로 보면 RHT가 *학습 안정화 기법*으로 채택된 경우와, *추론 PTQ에서 포맷에 맞춰 제한해야 하는 경우*를 비교할 수 있습니다.