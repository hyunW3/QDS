"""Report section ㉒: motivation checks for channel scaling (seeds, baselines, format-aware STE, shift, OLMo folding, overhead, training)."""
import json, statistics as st
from pathlib import Path
D = Path(__file__).resolve().parent.parent
R = json.load(open(D / "ptq8b/records.json"))
M = [("llama3-8b", "Llama-3-8B"), ("llama31-8b", "Llama-3.1-8B"), ("qwen3-8b", "Qwen3-8B")]


def get(m, fmt, meth, g=False, ste="fp32", seed=None, alpha=None):
    v = [r for r in R if r["model"] == m and r["fmt"] == fmt and r["method"] == meth and r["gptq"] == g and r["ste"] == ste
         and (alpha is None or r["alpha"] == alpha)]
    return [r for r in v if seed is None or r["seed"] == seed]


def cell(v, k="ppl", n=3):
    x = [r[k] for r in v if r.get(k) is not None]
    if not x: return "–"
    f = (lambda a: f"{100*a:.1f}") if k == "avg5" else (lambda a: f"{a:.3f}")
    return f(st.mean(x)) + (f" ± {(100 if k == 'avg5' else 1)*st.stdev(x):.{1 if k == 'avg5' else 3}f}" if len(x) > 1 else "")


def tab(rows, head):
    return '<div class="tw"><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join(
        "<tr>" + "".join(("<td>" if j == 0 else '<td class="n">') + f"{c}</td>" for j, c in enumerate(r)) + "</tr>" for r in rows) + "</table></div>"


seeds = tab([[lab] + [cell(get(m, f, me, g, alpha=0.5 if me == "sq_46" else None)) for m, _ in M] for lab, f, me, g in [
    ("NVFP4 대각 + 4/6", "nvfp4", "diag_46", False), ("NVFP4 대각", "nvfp4", "diag", False), ("NVFP4 SmoothQuant + 4/6", "nvfp4", "sq_46", False),
    ("NVFP4 대각 + 4/6 + GPTQ", "nvfp4", "diag_46", True), ("NVFP4 UFP4 + GPTQ", "nvfp4", "ufp4", True),
    ("MXFP4 대각 + 4/6 + H", "mxfp4", "diag_46_h", False), ("MXFP4 대각 + 4/6 + H + GPTQ", "mxfp4", "diag_46_h", True)]],
    ["WikiText-2 PPL (seed 0/1/2 평균 ± 표준편차)"] + [n for _, n in M])
seeds_a5 = tab([[lab] + [cell(get(m, f, me, g), "avg5") for m, _ in M] for lab, f, me, g in [
    ("NVFP4 대각 + 4/6", "nvfp4", "diag_46", False), ("NVFP4 대각 + 4/6 + GPTQ", "nvfp4", "diag_46", True), ("NVFP4 UFP4 + GPTQ", "nvfp4", "ufp4", True)]],
    ["Avg5 (seed 3개)"] + [n for _, n in M])
base_rows = []
for f in ["nvfp4", "mxfp4"]:
    for lab, me, a in [("E2M1-4/6 (변환 없음)", "e2m1_46", None), ("SmoothQuant α=0.5", "sq_46", 0.5), ("SmoothQuant α=0.3", "sq_46", 0.3), ("SmoothQuant α=0.7", "sq_46", 0.7),
                       ("SmoothQuant 형태, 층별 α 탐색", "sqgrid_46", None), ("AWQ 형태, 층별 α 탐색", "awq_46", None),
                       ("LET (shift 없음, 블록 단위)", "letns_46", None), ("LET 원본 (scale + shift, 블록 단위)", "let_46", None),
                       ("대각 + 4/6 (Linear 단위 학습)", "diag_46", None), ("대각 + 4/6 + H", "diag_46_h", None)]:
        if f == "mxfp4" and a in (0.3, 0.7): continue
        base_rows.append([f"{f.upper()} {lab}"] + [cell(get(m, f, me, seed=0, alpha=a)) for m, _ in M])
base = tab(base_rows, ["방법 (seed 0, + 4/6)"] + [n for _, n in M])
ste = tab([[f"{f.upper()} {lab}"] + [f"{cell(get(m, f, me, seed=0))} → {cell(get(m, f, me, ste='fmt'))}" for m, _ in M] for f, me, lab in [
    ("nvfp4", "diag", "대각"), ("nvfp4", "diag_46", "대각 + 4/6"), ("mxfp4", "diag", "대각"), ("mxfp4", "diag_46", "대각 + 4/6"), ("mxfp4", "diag_46_h", "대각 + 4/6 + H")]],
    ["STE: FP32 스케일 → 평가 형식 그대로"] + [n for _, n in M])
OL = json.load(open(D / "hf/olmo_fold.json")); bf = OL["BF16"]
pct = lambda k: f"{100*(OL[k]/bf-1):+.2f}%"
olmo = tab([[n] + [pct(f"{k} | E2M1"), pct(f"{k} | E2M1-4/6")] + [c] for n, k, c in [
    ("변환 없음", "E2M1 (no transform)", "–"), ("층별 s (양자화 커널 안에서 x/s)", "per-layer s (in-kernel x/s, not foldable)", "불가"),
    ("전역 residual s (함께 학습)", "global residual s (learned jointly, foldable)", "가능"), ("전역 s = 층별 s의 기하평균", "global residual s = geo-mean of per-layer s", "가능")]]
    + [["실제로 접은 모델 + 그냥 양자화", pct("fold: plain E2M1 on folded weights"), pct("fold: plain E2M1-4/6 on folded weights"), "BF16 " + f"{OL['fold: BF16 after folding']:.4f} (원래 {bf:.4f})"]],
    ["OLMo-2-1B (DCLM BPB 증가)", "E2M1", "E2M1-4/6", "접기"])
OV = json.load(open(D / "latency/scale_overhead.json"))["rows"]
over = tab([[f"K={r['K']}, M={r['M']}", f"{r['plain']*1e3:.0f}", f"{r['scaled']*1e3:.0f} ({100*(r['scaled']/r['plain']-1):+.1f}%)", f"{r['rot']*1e3:.0f} ({100*(r['rot']/r['plain']-1):+.1f}%)"] for r in OV],
           ["활성값 양자화 커널 (μs, RTX PRO 6000)", "NVFP4 양자화만", "+ 채널별 x/s", "+ 16×16 Hadamard"])
SH = {m: json.load(open(D / f"ana/shift_{m}.json"))["rows"] for m in ["llama3-8b", "qwen3-8b"]}
mean = lambda rows, k: st.mean(r[k] for r in rows if k in r)
shift = tab([[n, f"{mean(SH[m],'first_err_orig_scale'):.4f} → {mean(SH[m],'first_err_orig_shift'):.4f}", f"×{mean(SH[m],'first_absmax_inflation_median'):.1f}",
              f"{mean(SH[m],'rest_err_orig_scale'):.4f} → {mean(SH[m],'rest_err_orig_shift'):.4f}", a, b, c] for m, n, a, b, c in [
    ("llama3-8b", "Llama-3-8B", "6.808", "6.728", "6.692"), ("qwen3-8b", "Qwen3-8B", "10.735", "10.709", "10.163")]],
    ["모델", "첫 토큰 활성값 오차 (shift 전 → 후)", "첫 토큰 블록 absmax", "일반 토큰 활성값 오차", "PPL: shift 전부", "PPL: 첫 토큰만 shift 제외", "PPL: shift 버림 (LET 스케일만)"])
m7 = tab([["UFP4 (균일 + RHT)", "+0.0435 ± 0.0062", "−0.0013 ± 0.0005", "3.20"], ["smoothing, 오라클 통계 (같은 배치)", "+0.0448 ± 0.0059", "–", "4.09"],
          ["smoothing, 지연 통계 (이전 step)", "+0.0466 ± 0.0065", "+0.0018 ± 0.0065", "4.12"], ["smoothing, 온라인 학습 s", "+0.0449 ± 0.0085", "+0.0002 ± 0.0080", "6.04"]],
         ["Stage 1 (66M, seed 3개, E2M1-4/6 + dY SR)", "BF16 대비 val loss Δ", "같은 seed 오라클 대비", "s/step"])

sec = f'''<h2 id="r22">㉒ 채널 스케일의 motivation 검증: 오차 구조, 민감도, 베이스라인, shift, 접기, 학습</h2>
<p>"회전 대신 채널 스케일"이라는 주장을 method comparison으로 정리하기 위해 필요한 근거를 하나씩 확인했습니다. 대상은 Llama-3-8B, Llama-3.1-8B, Qwen3-8B이고(OLMo-2-1B는 접기 검증용), 분석 코드는 <code>ana8b.py</code>, <code>shift_ana.py</code>, <code>olmo_fold.py</code>, 결과는 <code>ana/</code>와 <code>ptq8b/records.json</code>에 있습니다.</p>
<h3>1) 스케일을 적용하면 X와 W의 오차는 어떻게 변하는가</h3>
<figure><img src="data:image/png;base64,{{{{fig21}}}}" alt="오차 구조"><figcaption>그림 21. 레이어·그룹 평균. 왼쪽: 변환 없음 대비 GEMM 출력 오차와 X·W 텐서 NMSE의 변화. 가운데·오른쪽: 활성값 X의 양자화 오차가 정규화 크기 구간별로 어디서 나오는지.</figcaption></figure>
<ul>
<li><b>학습한 대각 스케일은 X·W의 텐서 오차를 거의 바꾸지 않고(X +1~5%, W 0%) GEMM 출력 오차만 10~15% 줄입니다.</b> 오차의 총량이 아니라 위치를 바꾸는 것입니다. 상대 피연산자 값이 큰 채널의 오차를 줄이고, 덜 중요한 채널로 옮깁니다. 그래서 텐서 MSE가 아니라 출력 오차를 기준으로 스케일을 정해야 합니다.</li>
<li>SmoothQuant는 출력 오차를 거의 줄이지 못합니다(Llama +3%, Qwen3 −6%).</li>
<li><b>회전은 dead-zone 오차를 거의 없애지만, 오차를 간격이 큰 2~6 구간으로 몰아넣어 NVFP4에서는 출력 오차가 오히려 늘어납니다</b>(+3~5%). 이는 ⑧·⑱의 메커니즘을 8B 모델에서 다시 확인한 것입니다.</li>
<li><b>MXFP4(E8M0)는 활성값 오차의 21~25%가 클리핑에서 나옵니다.</b> OCP 규칙은 지수를 내림으로 정하므로 블록 최댓값 근처의 값이 6을 넘으면 잘립니다. 회전은 블록 crest를 낮춰 클리핑을 16~17%로 줄이고 출력 오차를 21~26% 줄입니다. 채널 스케일만으로는 클리핑이 줄지 않습니다. 이것이 E8M0에서 회전이 다시 필요한 이유입니다.</li>
<li>학습한 s와 SmoothQuant s의 log 상관은 0.19~0.27에 불과합니다. 학습한 s는 X를 SQ만큼 균등화하지 않고 W 쪽 불균일을 더 허용하는 다른 해입니다.</li>
</ul>
<h3>2) loss 민감도</h3>
<figure><img src="data:image/png;base64,{{{{fig22}}}}" alt="민감도"><figcaption>그림 22. Linear 그룹 하나만 NVFP4-4/6으로 양자화했을 때의 NLL 증가를 층 깊이 4구간으로 합산(평가 창 48×1024 토큰).</figcaption></figure>
<ul>
<li>학습한 대각은 손해의 합을 37~43% 줄이고, 회전은 15~21% 늘립니다.</li>
<li>손해는 gate/up 입력과 down, 그리고 <b>마지막 1/4 층</b>에 몰려 있습니다(Qwen3 35층 gate/up, Llama 31층·1층 down). 학습한 대각은 이 구간의 손해를 크게 줄입니다(Llama 마지막 구간 0.027 → 0.014).</li>
</ul>
<h3>3) seed 반복</h3>
{seeds}
{seeds_a5}
<ul>
<li>SmoothQuant 대비 우위는 seed를 반복해도 확실합니다(Llama 0.07~0.08 = 표준편차의 5배 이상, Qwen3 0.085).</li>
<li>GPTQ를 더하면 Llama에서는 UFP4 + GPTQ가 0.025~0.03 앞서고(표준편차의 약 3배), Qwen3에서는 대각 + 4/6 + GPTQ가 0.07 앞섭니다. Avg5는 세 모델 모두 동률입니다.</li>
</ul>
<h3>4) 베이스라인: "학습"인가 "층별 선택"인가</h3>
{base}
<ul>
<li><b>AWQ 형태(s = mean|x|^α)의 층별 α 탐색이 학습한 대각과 사실상 같습니다</b>(Llama 6.691 대 6.694, 6.758 대 6.760; Qwen3 10.11 대 10.08, seed 편차 안). 성능을 만드는 것은 경사하강 학습이 아니라, <b>양자화된 출력 오차를 기준으로 층별 스케일을 고르는 것</b>입니다.</li>
<li>SmoothQuant 형태의 층별 탐색은 Llama에서 0.05~0.06 나쁩니다. 스케일의 함수 형태도 중요합니다. 전 층 고정 α는 모델마다 최적값이 다릅니다(Llama 0.5, Qwen3 0.3).</li>
<li>OmniQuant식 블록 단위 LET(shift 없음)는 Linear 단위 목표와 비슷하거나 약간 나쁩니다(Llama 6.72 대 6.69).</li>
<li><b>shift를 넣은 LET 원본은 Qwen3 NVFP4에서 붕괴하고(10.74), MXFP4에서는 세 모델 모두 크게 나빠집니다.</b></li>
</ul>
<h3>5) 형식 인지 STE</h3>
{ste}
<p>실제 평가 형식(E4M3 + 4/6, E8M0)을 그대로 통과하는 STE는 일관된 이득이 없고, MXFP4에서는 오히려 나쁩니다(Llama-3.1 대각 8.06 → 8.57). 2의 거듭제곱 스케일은 계단식이라 STE gradient가 부정확해지는 것으로 보입니다. <b>매끄러운 대리 양자화기(FP32 스케일)로 학습하는 편이 낫습니다.</b> "MXFP4에서 대각 단독이 약한 것은 STE 불일치 때문"이라는 가설은 기각되고, 1)의 클리핑이 원인으로 남습니다.</p>
<h3>6) shift는 왜 해로운가</h3>
{shift}
<ul>
<li>q/k/v와 gate/up 입력은 RMSNorm을 거친 뒤라 거대 활성값 토큰이 없습니다. 일반 토큰에서는 shift 전후 오차가 거의 같습니다(+3~8%).</li>
<li><b>망가지는 것은 분포가 다른 소수 토큰입니다.</b> shift는 평균적인 토큰에 맞춘 채널별 상수입니다. 그래서 첫 토큰(attention sink)에서는 x − d ≈ −d가 되어 블록 absmax가 8배 이상 커지고 오차가 7~10배 늘어납니다.</li>
<li><b>인과 확인 (Llama-3)</b>: 첫 토큰만 shift를 빼면 PPL이 6.81 → 6.73으로 회복되고, shift를 버리면 6.69입니다. 손해가 거의 전부 sink 토큰 하나에서 나옵니다.</li>
<li><b>Qwen3</b>: BOS 토큰이 없고 sink 역할을 하는 토큰이 여러 위치에 있습니다(위치 0, 그리고 여러 층에서 반복되는 '{{', ' don' 등). 초기층에서는 크기가 작은 고빈도 토큰(6층의 ' the', 중앙값 노름의 0.4배)도 오차가 10배 커집니다(6층 gate/up 입력에서 토큰의 14%가 3배 이상). 그래서 첫 토큰만 빼도 회복되지 않고(10.71), shift를 버리면 10.16으로 돌아옵니다.</li>
<li>블록 스케일은 이미 토큰마다 범위를 맞추므로, shift가 일반 토큰에 주는 이득은 거의 없습니다. 반대로 sink 같은 비정형 토큰을 망가뜨립니다. 블록 스케일 FP4에서는 shift를 쓰지 않는 것이 맞습니다.</li>
</ul>
<h3>7) OLMo-2처럼 norm이 sublayer 뒤에 있어도 접을 수 있는가</h3>
{olmo}
<p>OLMo-2에서는 q/k/v와 gate/up이 정규화되지 않은 residual stream을 읽기 때문에 층별 s를 RMSNorm에 접을 수 없습니다. 대신 <b>residual stream 전체에 공유 s 하나</b>를 적용하는 방식은 접을 수 있습니다. embedding과 모든 post-norm weight에 1/s를 넣고, q/k/v와 gate/up 열에 s를 곱하며, 최종 norm 앞에 곱셈 하나만 남습니다. 이 방식이 층별 s와 같은 정확도를 냅니다(+2.41% 대 +2.42%, 4/6 사용 시 기하평균 s가 +2.09%로 동일). 실제로 접은 모델의 BF16 BPB는 원래와 같았고, 접은 weight를 BF16으로 반올림하면서 약 0.1%p 손해가 생겼습니다.</p>
<h3>8) 접을 수 없을 때의 비용</h3>
{over}
<p>측정은 RTX PRO 6000(asri139, GPU를 단독으로 사용)에서 했습니다. 활성값 양자화 커널에 채널별 x/s 곱셈을 넣는 비용은 측정 잡음 수준(−4~+6%)입니다. 블록 Hadamard는 같은 커널을 20~34% 느리게 합니다. 따라서 접기가 불가능한 구조에서도 채널 스케일의 비용은 사실상 0입니다. 앞서 snu233에서 잰 값은 다른 사용자의 작업과 GPU를 공유한 상태라 폐기했습니다.</p>
<h3>9) FP4 학습에서는 (M7)</h3>
{m7}
<ul>
<li><b>실제 커널에서 구현 가능한 지연 통계(이전 step의 채널 absmax)로 바꿔도 결과가 유지되고</b>(+0.0018 ± 0.0065, 잡음 범위) 추가 비용도 없습니다.</li>
<li><b>학습 중 s를 학습해도 이득이 없습니다.</b> GEMM 단위로 보면 학습한 s가 공식보다 4% 나쁘고, 시간은 48% 더 듭니다.</li>
<li>UFP4는 모든 seed에서 미세하게 앞섭니다(같은 seed 대비 −0.0013 ± 0.0005). 정확한 표현은 "UFP4와 0.001~0.002 이내"입니다.</li>
<li>WGRAD는 축소 축이 토큰이라, 지연·학습 s가 채널이 아니라 토큰 위치에 붙는다는 한계가 있습니다.</li>
</ul>
<h3>정리: 데이터가 지지하는 주장</h3>
<ol>
<li>블록 스케일 E2M1(E4M3 스케일)에서는 회전 대신 <b>채널 스케일</b>을 써야 합니다. 스케일은 텐서 MSE가 아니라 <b>양자화된 GEMM 출력 오차</b>를 기준으로 층별로 골라야 합니다. 경사하강 학습과 매개변수 1개 탐색(AWQ 형태)의 결과가 같으므로, 핵심은 "학습"이 아니라 "출력 오차 기준의 층별 선택"입니다.</li>
<li>shift는 attention sink 같은 비정형 토큰을 망가뜨리므로 쓰지 않습니다.</li>
<li>스케일 형식이 E8M0(MXFP4)이면 클리핑이 지배적이라 회전이 다시 필요하고, 채널 스케일과 함께 씁니다.</li>
<li>접기는 pre-norm(Llama, Qwen)에서 그대로 되고, post-norm(OLMo-2)에서도 전역 residual 스케일로 됩니다. 접지 못하더라도 비용은 측정 잡음 수준입니다.</li>
<li>FP4 학습에서는 지연 통계 SmoothQuant 공식으로 충분합니다. s를 학습할 필요가 없고, 결과는 UFP4와 0.002 이내입니다.</li>
</ol>

'''
p = D / "report_template.html"; s = open(p).read()
if '<h2 id="r22">' in s: s = s[:s.index('<h2 id="r22">')] + s[s.index('<h2 id="concl">'):]
s = s.replace('<h2 id="concl">', sec + '<h2 id="concl">', 1)
if 'href="#r22"' not in s: s = s.replace('<a href="#r21">㉑ 8B PTQ</a>', '<a href="#r21">㉑ 8B PTQ</a><a href="#r22">㉒ motivation 검증</a>', 1)
if "2026-10-27" not in s:
    s = s.replace('<li>2026-10-26:', '<li>2026-10-27: ㉒ 추가. 오차 구조와 민감도 분석, seed 3개 반복, 베이스라인(SQ α, SQ·AWQ 층별 α 탐색, LET 원본과 shift 없는 버전), 형식 인지 STE, shift 원인 분석, OLMo-2 접기, 커널 비용, FP4 학습의 지연·학습 스케일을 정리했습니다.</li>\n<li>2026-10-26:', 1)
open(p, "w").write(s); print("r22 written")
