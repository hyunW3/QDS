"""Rebuild report section ㉑ (8B PTQ) in report_template.html from ptq8b/results_rows.json."""
import json
from pathlib import Path
D = Path(__file__).resolve().parent.parent
P = json.load(open(D / "ptq8b/results_rows.json"))
MODELS = [("llama3-8b", "Llama-3-8B"), ("llama31-8b", "Llama-3.1-8B"), ("qwen3-8b", "Qwen3-8B")]
METH = [("e2m1", "E2M1"), ("e2m1_46", "E2M1-4/6"), ("e2m1_h", "E2M1 + H"), ("ufp4", "UFP4"), ("sq_46", "SQ + 4/6"),
        ("diag", "대각"), ("diag_46", "대각 + 4/6"), ("diag_h", "대각 + H"), ("diag_46_h", "대각 + 4/6 + H"),
        ("mrgptq", "MR-GPTQ (공식, GPTQ 열만)"), ("mrgptq_h32", "MR-GPTQ, H 32 (공식)")]


def table(fmt, gptq_cols):
    cols = [(m, g) for m, _ in MODELS for g in ([False, True] if gptq_cols else [False])]
    hdr = "<tr><th>방법</th>" + "".join(f"<th>{n} {'GPTQ' if g else 'RTN'}</th>" for (m, g) in cols for mm, n in MODELS if mm == m) + "</tr>"
    get = lambda m, k, g: P[m].get(fmt, {}).get(k + ("+gptq" if g else ""))
    best = {}
    for m, g in cols:
        v = [r["ppl"] for k, _ in METH if (r := get(m, k, g))]
        best[(m, g)] = min(v) if v else None
    out = '<div class="tw"><table>' + hdr
    out += "<tr><td>BF16</td>" + "".join(
        (f'<td class="n">{P[m]["nvfp4"]["bf16"]["ppl"]:.2f}<br><span style="font-size:.8em">{100*P[m]["nvfp4"]["bf16"]["avg5"]:.1f}</span></td>' if not g else '<td class="n">–</td>')
        for m, g in cols) + "</tr>"
    for k, lab in METH:
        if not any(get(m, k, g) for m, g in cols): continue
        out += f"<tr><td>{lab}</td>"
        for m, g in cols:
            r = get(m, k, g)
            if not r: out += '<td class="n">–</td>'; continue
            cls = "n good" if r["ppl"] == best[(m, g)] else "n"
            a5 = f'{100*r["avg5"]:.1f}' if r.get("avg5") is not None else "–"
            out += f'<td class="{cls}">{r["ppl"]:.2f}<br><span style="font-size:.8em;color:var(--ink2,#52514e)">{a5}</span></td>'
        out += "</tr>"
    return out + "</table></div>"


sec = f'''<h2 id="r21">㉑ 8B 모델 W4A4 PTQ: 형식 × 변환 × GPTQ, 공식 MR-GPTQ와 비교</h2>
<p>작은 모델(⑭)에서 본 "회전 없는 E2M1 + 학습형 대각 스케일" 결과를 7~8B 모델(Llama-3-8B, Llama-3.1-8B, Qwen3-8B)과 downstream 과제로 확장했습니다. 블록 크기와 스케일 형식을 바꿔(NVFP4 G16, NVFP4-G32, MXFP4) 각 결론이 어디까지 일반화되는지 확인했고, <b>공식 MR-GPTQ</b>(IST-DASLab FP-Quant)와 비교했습니다(<code>ptq8b.py</code>, <code>ptq8b/</code>).</p>
<ul>
<li><b>설정</b>
<ul>
<li>양자화 범위: 디코더의 모든 Linear를 W4A4로 양자화했습니다.</li>
<li>스케일 형식:
<ul>
<li>NVFP4: 블록 16, E4M3 스케일 + FP32 텐서 스케일</li>
<li>NVFP4-G32: 블록 32, E4M3 스케일. 표준이 아닌 형식으로, 블록 크기만 바꾼 대조군입니다.</li>
<li>MXFP4(OCP MX 표준): 블록 32, E8M0(2의 거듭제곱) 스케일. 4/6에 해당하는 것은 지수 e와 e+1 중 MSE가 작은 쪽을 고르는 것으로 정의했습니다.</li>
</ul></li>
<li>회전 H: 블록 크기와 같은 G×G Hadamard입니다.</li>
<li>대각: 같은 입력을 쓰는 Linear끼리 스케일을 공유하고, RMSNorm·v_proj·up_proj에 접어 넣어 추론 비용이 0입니다. 출력 MSE를 STE로 최소화했고, calibration은 DCLM 8×1024 토큰입니다. +H 변형은 회전 후 양자화를 거치는 STE로 학습했습니다.</li>
<li>GPTQ(우리 구현): Hessian은 DCLM 128×1024 토큰(스케일 학습 데이터와 겹치지 않음), act-order 없음, 비순차 방식입니다.</li>
<li>MR-GPTQ(공식 코드, 논문 권장 설정): GPTQ + act-order + MSE weight 스케일 탐색 + Hadamard(NVFP4 16, MXFP4 128, 32 변형도 측정). calibration은 fineweb-edu 128×2048 토큰입니다.</li>
</ul></li>
<li><b>평가</b>: WikiText-2 PPL(seq 2048)과 lm-eval 0-shot Avg5(HS/ARC-E/ARC-C/PIQA는 acc_norm, WG는 acc)</li>
</ul>
<figure><img src="data:image/png;base64,{{{{fig20}}}}" alt="8B PTQ"><figcaption>그림 20. BF16 대비 WikiText-2 PPL 증가(%). 색은 열(모델 × RTN/GPTQ)마다의 순위입니다(초록 = 좋음).</figcaption></figure>
<h3>NVFP4 (G16, E4M3) — 각 칸: PPL / Avg5, 초록 = 열 최선</h3>
{table("nvfp4", True)}
<h3>NVFP4-G32 (E4M3) — 블록 크기만 바꾼 대조군</h3>
{table("nvfp4_g32", False)}
<h3>MXFP4 (G32, E8M0)</h3>
{table("mxfp4", True)}
<ul>
<li><b>공식 MR-GPTQ보다 우리 방법이 두 형식, 세 모델 모두에서 PPL이 낮습니다.</b>
<ul>
<li>NVFP4: 대각 + 4/6 + GPTQ 6.66 / 6.74 / 10.02 대 MR-GPTQ 6.87 / 6.89 / 10.24</li>
<li>MXFP4: 대각 + 4/6 + H + GPTQ 7.04 / 7.08 / 10.31 대 MR-GPTQ 7.18 / 7.22 / 10.53</li>
<li>차이는 0.10~0.22로, 대각 스케일 학습의 실행 간 편차(약 0.1)보다 큽니다.</li>
<li>Avg5는 같거나 0.6%p 이내로 높아서, 잡음 범위입니다.</li>
<li>GPTQ 없는 RTN만으로도 NVFP4에서는 대각 + 4/6(6.69 / 6.76 / 10.08)이 MR-GPTQ보다 낮습니다.</li>
</ul></li>
<li><b>RTN에서 E4M3 스케일(NVFP4, G16·G32)의 최선은 회전 없는 대각입니다.</b>
<ul>
<li>회전 계열(E2M1 + H, UFP4)은 가장 나쁜 편입니다.</li>
<li>대각 + 4/6에 H를 더해도 이득이 없습니다(Llama 6.69 / 6.77로 같고, Qwen3는 10.21로 오히려 나쁨).</li>
<li>블록을 32로 키운 NVFP4-G32에서도 순서가 같습니다.</li>
</ul></li>
<li><b>스케일 형식이 E8M0(MXFP4)이면 회전이 이득으로 바뀝니다.</b>
<ul>
<li>대각과 회전은 보완적이고, 대각 + 4/6 + H(+GPTQ)가 세 모델 모두에서 최선입니다.</li>
<li>같은 G=32라도 E4M3 스케일에서는 회전이 손해이므로, 회전의 득실을 가르는 것은 블록 크기보다 스케일 형식입니다.</li>
<li>해석(가설): 2의 거듭제곱 스케일은 블록 absmax를 격자 최댓값에 맞추지 못해 범위를 최대 2배까지 낭비하고, 회전이 블록 안의 분포를 고르게 만들어 이 손실을 줄입니다.</li>
</ul></li>
<li><b>GPTQ는 회전 방법을 크게 끌어올립니다.</b>
<ul>
<li>우리 구현의 UFP4 + GPTQ는 Llama 두 모델에서 6.63 / 6.70으로, 대각 + 4/6 + GPTQ(6.66 / 6.74)와 사실상 같습니다.</li>
<li>회전도 대각 스케일도 없는 E2M1에는 GPTQ가 불안정합니다(Llama-3 6.89 → 8.77). 대각 스케일을 먼저 적용하면 안정됩니다.</li>
</ul></li>
</ul>
<div class="note">한계와 주의:
<ul>
<li>대각 스케일 학습은 실행마다 결과가 조금씩 다릅니다(Qwen3 NVFP4-G32 대각: 두 노드에서 10.25와 10.33). 0.1 미만의 차이는 seed 반복 없이 확정할 수 없습니다.</li>
<li>우리 GPTQ는 act-order와 스케일 탐색이 없는 단순 구현입니다. 처음에는 8k 토큰 Hessian으로 돌렸는데, K=14336에서 Hessian이 rank 부족이 되어 과적합했습니다. 표의 값은 131k 토큰 결과이고, 8k 결과는 <code>ptq8b/results_gptq8k/</code>에 보관했습니다.</li>
<li>MR-GPTQ와 우리 방법은 calibration 데이터가 다릅니다(fineweb-edu 대 DCLM). WikiText-2는 두 데이터 모두에 속하지 않습니다.</li>
<li>MR-GPTQ(공식)는 활성값의 FP32 텐서 스케일을 calibration에서 정한 정적 값으로 쓰고, 우리 구현은 호출마다 동적으로 계산합니다. 이 차이는 우리 쪽에 약간 유리할 수 있습니다. 블록(E4M3/E8M0) 스케일은 양쪽 모두 동적입니다.</li>
<li>MR-GPTQ는 공식 코드(FP-Quant d2e3092, 수정 없음)를 우리 평가 방식으로 감싸 돌렸습니다. Qwen3에서는 transformers 4.57.6 호환을 위해 계산을 바꾸지 않는 속성 전달 shim을 넣었습니다. BF16 PPL(6.138 / 6.239 / 9.727)은 같은 경로에서 정확히 재현됐고, seed는 1개입니다.</li>
<li>QuaRot·SpinQuant·FlatQuant 같은 전체 회전 기반 방법과 OmniQuant LET 원본은 아직 비교하지 않았습니다.</li>
</ul></div>

'''
p = D / "report_template.html"; s = open(p).read()
i = s.index('<h2 id="r21">'); j = s.index('<h2 id="concl">')
s = s[:i] + sec + s[j:]
s = s.replace('<a href="#r21">㉑ 8B PTQ</a>', '<a href="#r21">㉑ 8B PTQ</a>')
open(p, "w").write(s)
print("r21 rebuilt")
