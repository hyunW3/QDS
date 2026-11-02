import json, glob
for f in sorted(glob.glob("out/*.json")):
    r = json.load(open(f))
    if "avg5" in r: continue
    a = r["args"]; print(a["hf"], a["tag"], a["method"], a.get("fmt", "nvfp4"), 1 if a.get("gptq") else 0)
