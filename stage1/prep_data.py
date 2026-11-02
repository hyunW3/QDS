"""Pre-tokenize DCLM text for Stage-1 training (SmolLM2 tokenizer, uint16 .bin; EOS between documents)."""
import os
import json, sys, numpy as np
from pathlib import Path
from transformers import AutoTokenizer
SRC = Path(os.environ.get("DCLM_DIR", "data/dclm_baseline_1.0_2shards_shuffled"))
OUT = Path(__file__).resolve().parent / "data"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
EOS = tok.eos_token_id
assert len(tok) < 65536


def run(src, out, target):
    buf, n, batch = [], 0, []
    with open(out, "wb") as f:
        for line in open(src):
            batch.append(json.loads(line)["text"])
            if len(batch) == 2048:
                for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                    buf.extend(ids); buf.append(EOS)
                batch = []
                arr = np.asarray(buf, dtype=np.uint16); arr.tofile(f); n += len(arr); buf = []
                print(f"{out.name}: {n/1e6:.0f}M tokens", flush=True)
                if n >= target:
                    break
    return n


print("val", run(SRC / "dclm_baseline_1.0_2shards.val.jsonl", OUT / "val.bin", 4_000_000))
print("train", run(SRC / "dclm_baseline_1.0_2shards.chunk.01.jsonl", OUT / "train.bin", 700_000_000))
