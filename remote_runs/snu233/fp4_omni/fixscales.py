import sys, torch
src, dst, dev = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = torch.load(src, map_location="cpu"); S = d["scales"]
for i in [k for k in S if isinstance(k, int)]:
    for g, v in S[i].items():
        l = v.float().log(); l = l - l.mean(); S[i][g] = l.clamp(-dev, dev).exp()      # remove the scale-invariant common factor, bound the spread
torch.save(d, dst); print("fixed", dst)
