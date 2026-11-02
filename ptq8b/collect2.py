"""All 8B PTQ runs as flat records (model, fmt, method, gptq, ste, seed, alpha) -> ptq8b/records.json; seed-aware summary."""
import json, glob, subprocess, sys, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
if "--nosync" not in sys.argv:
    for h in ["info107", "info103", "asri139", "snu188_ansible"]:
        (R / h).mkdir(parents=True, exist_ok=True)
        subprocess.run(f"timeout 180 rsync -a --include='*.json' --exclude='*' {h}:hyun2/fp4_ptq/out/ {R}/{h}/", shell=True, stdin=subprocess.DEVNULL)
recs = {}
for f in glob.glob(str(R / "*" / "*.json")):
    if "/ana_" in f or "/sens" in f or "olmo" in f: continue
    try: r = json.load(open(f))
    except Exception: continue
    if "wikitext2_ppl" not in r or "args" not in r: continue
    a = r["args"]
    if a.get("gptq") and a.get("ngptq") != 128: continue
    k = (a["tag"], a.get("fmt", "nvfp4"), a["method"].replace("e2m1_h16", "e2m1_h"), bool(a.get("gptq")), a.get("ste", "fp32"), a.get("seed", 0), a.get("alpha", 0.5) if a["method"] == "sq_46" else None)
    if k in recs and "avg5" in recs[k] and "avg5" not in r: continue
    recs[k] = dict(model=k[0], fmt=k[1], method=k[2], gptq=k[3], ste=k[4], seed=k[5], alpha=k[6], ppl=r["wikitext2_ppl"], avg5=r.get("avg5"))
out = list(recs.values())
json.dump(out, open(Path(__file__).resolve().parent / "records.json", "w"), indent=1)
print(len(out), "records")
