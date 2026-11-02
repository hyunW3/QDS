"""Expert-GEMM capture for transformers-5 MoE blocks (OLMoE, Qwen3-MoE, ...).

In transformers 5 the routed experts are one module with 3D parameters gate_up_proj [E, 2I, H] and down_proj [E, H, I]
and call F.linear per expert, so nn.Linear hooks never see them. install() replaces each such module's forward with
the same eager per-expert loop, and taps both expert GEMMs: on the forward pass it keeps (X, W); a gradient hook on the
GEMM output delivers dY, and evaluate(name, gemm, A, B) is then called for FPROP (X, W), DGRAD (dY, W^T) and WGRAD
(dY^T, X^T; tokens trimmed to a multiple of wg_mult so every block size divides the reduction axis).
Only experts with idx % expert_stride == 0 in layers with idx % layer_stride == 0 are tapped (the rest run untapped).
All tapped tensors are moved to `dev` (fp32) before evaluate, so sharded (device_map="auto") models work.
"""
import re, types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def load_auto(name, max_memory):
    from hf_common import HFWrap, HFTok
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa",
                                             device_map="auto", max_memory=max_memory)
    return HFWrap(m).eval(), HFTok(name)


def install(hfmodel, evaluate, dev="cuda:0", expert_stride=8, layer_stride=1, wg_mult=128):
    n_tapped = 0
    for name, mod in hfmodel.named_modules():
        gu = getattr(mod, "gate_up_proj", None)
        if not (isinstance(gu, torch.nn.Parameter) and gu.dim() == 3 and hasattr(mod, "down_proj")):
            continue
        li = int(re.search(r"layers\.(\d+)\.", name).group(1))
        if li % layer_stride:
            continue
        prefix = f"layers.{li}.mlp.experts"

        def tap(lname, x, W, on):
            y = F.linear(x, W)
            if on and y.requires_grad:
                X = x.detach().float().to(dev); Wd = W.detach().float().to(dev)
                def hook(g, X=X, Wd=Wd):
                    dY = g.detach().float().to(dev)
                    with torch.no_grad():
                        evaluate(lname, "fprop", X, Wd)
                        evaluate(lname, "dgrad", dY, Wd.T.contiguous())
                        n = (X.shape[0] // wg_mult) * wg_mult
                        if n:
                            evaluate(lname, "wgrad", dY[:n].T.contiguous(), X[:n].T.contiguous())
                y.register_hook(hook)
            return y

        def forward(self, hidden_states, top_k_index, top_k_weights, prefix=prefix):
            out = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for e in hit:
                e = int(e[0])
                if e == self.num_experts:
                    continue
                pos, tok = torch.where(mask[e])
                on = (e % expert_stride == 0)
                h = tap(f"{prefix}.{e}.gate_up_proj", hidden_states[tok], self.gate_up_proj[e], on)
                gate, up = h.chunk(2, dim=-1)
                h = tap(f"{prefix}.{e}.down_proj", self.act_fn(gate) * up, self.down_proj[e], on)
                out.index_add_(0, tok, (h * top_k_weights[tok, pos, None]).to(out.dtype))
            return out

        mod.forward = types.MethodType(forward, mod)
        n_tapped += 1
    return n_tapped
