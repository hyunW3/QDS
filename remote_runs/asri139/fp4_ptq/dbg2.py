exec(open("dbg_gptq.py").read().split("got = {}")[0])
got = []
m.model.layers[0].mlp.down_proj.register_forward_pre_hook(lambda mm, a: got.append(a[0].reshape(-1, a[0].shape[-1]).float()))
with torch.no_grad():
    for s in x: m(input_ids=s[None])
X = torch.cat(got); W = m.model.layers[0].mlp.down_proj.weight.float(); N, K = W.shape; G = 16; HG = env["HG"]
Hs = X.T @ X
Hr = (Hs.view(K, K // G, G) @ HG).view(K, K); Hr = (HG.T @ Hr.view(K // G, G, K)).view(K, K)
print("sym err", (Hr - Hr.T).abs().max().item(), "diag min", Hr.diag().min().item(), "zero diag", (Hr.diag() == 0).sum().item())
Wr = (W.view(N, K // G, G) @ HG).view(N, K)
amax = Wr.view(N, K // G, G).abs().amax(-1)
print("W rot: zero blocks", (amax == 0).sum().item(), "W orig zero blocks", (W.view(N, K // G, G).abs().amax(-1) == 0).sum().item())
H2 = Hr.clone(); H2 += 0.01 * H2.diag().mean() * torch.eye(K, device="cuda")
L = torch.linalg.cholesky(H2); Hi = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
print("Hinv finite", torch.isfinite(Hi).all().item(), "diag min", Hi.diag().min().item())
gs = env["tensor_gs"](amax, E2M1[-1].cuda(), False); print("gs", gs.item())
c = env["scale_cands"](amax[:, :1, None], E2M1[-1].cuda(), False, gs)[0]; print("scale min", c.min().item(), "zeros", (c == 0).sum().item())
