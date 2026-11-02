"""Open-source (Hugging Face) models behind the same interface the probe scripts use for lingua checkpoints.

  model(inp, target=tgt, attn_impl=...) -> mean next-token cross-entropy
  model.named_modules() -> decoder-layer submodules renamed to "layers.<i>.<...>" (e.g. layers.7.self_attn.q_proj)
  tok.encode(text, add_bos=True, add_eos=True) -> list[int]

Weights are frozen (no weight gradients are needed); the embedding output is made to require grad so the backward
pass still produces dY for every Linear. The model runs in bf16, so every Linear sees bf16 operands as in a
mixed-precision training step. lm_head is outside "layers." and therefore not probed.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFTok:
    def __init__(self, name):
        self.t = AutoTokenizer.from_pretrained(name)
        self.bos = self.t.bos_token_id
        self.eos = self.t.eos_token_id

    def encode(self, s, add_bos=True, add_eos=True):
        ids = self.t.encode(s, add_special_tokens=False)
        if add_bos and self.bos is not None:
            ids = [self.bos] + ids
        if add_eos and self.eos is not None:
            ids = ids + [self.eos]
        return ids


class HFWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        for p in m.parameters():
            p.requires_grad_(False)
        m.get_input_embeddings().register_forward_hook(lambda mod, i, o: o.requires_grad_(True))

    def forward(self, inp, target=None, attn_impl=None):
        logits = self.m(input_ids=inp).logits
        if target is None:
            return logits
        return F.cross_entropy(logits.float().flatten(0, 1), target.flatten().to(logits.device))

    def named_modules(self, *a, **k):
        for n, mod in self.m.named_modules(*a, **k):
            i = n.find("layers.")
            yield (n[i:] if i >= 0 else "hf." + n), mod


def load_hf(name, dev="cuda"):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
    return HFWrap(m).to(dev).eval(), HFTok(name)
