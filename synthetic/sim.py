import numpy as np
rng=np.random.default_rng(0)
G=np.array([0,.5,1,1.5,2,3,4,6.]); U=np.arange(8)*6/7  # E2M1 vs uniform(INT4-like, same max)
def rtn(t,G,th=None):
    th=(G[1:]+G[:-1])/2 if th is None else th
    return G[np.searchsorted(th,t)]
def sr(t,G):
    i=np.clip(np.searchsorted(G,t,side='right')-1,0,len(G)-2); a,b=G[i],G[i+1]
    p=(t-a)/(b-a); return np.where(rng.random(t.shape)<p,b,a)
X=rng.standard_normal((200000,16)); s=np.abs(X).max(1,keepdims=True)/6
A=X; t=np.abs(X)/s; sg=np.sign(X)
def rep(name,q):
    Ah=sg*q*s; e=Ah-A
    alpha=(Ah*A).sum()/(A*A).sum(); rel=(e**2).sum()/(A*A).sum()
    # per-tensor LS rescale
    c=(Ah*A).sum()/(Ah*Ah).sum(); rel_c=((c*Ah-A)**2).sum()/(A*A).sum()
    print(f"{name:22s} alpha={alpha:.4f}  relMSE={rel:.5f}  SQNR={-10*np.log10(rel):5.2f}dB  | LS-rescaled SQNR={-10*np.log10(rel_c):5.2f}dB")
rep("E2M1 RTN",rtn(t,G)); rep("E2M1 SR",sr(t,G))
rep("INT-uniform RTN",rtn(t,U)); rep("INT-uniform SR",sr(t,U))
# per-bin bias of E2M1 RTN under this distribution
q=rtn(t,G)
for g in G: m=q==g; print(f"  level {g}: mass={m.mean():.3f} E[q-t|bin]={(q[m]-t[m]).mean():+.4f}")
# threshold shift: make each bin's conditional mean error ~0 where feasible (greedy via empirical quantiles)
th=(G[1:]+G[:-1])/2; tt=np.sort(t[t<6].ravel())
for _ in range(200):
    for k in range(1,len(th)):   # skip zero-level threshold
        lo=th[k-1]; hi=th[k] if k<len(th) else 6
    # simple: shift thresholds so mean error of whole tensor (excluding max elements) = 0, uniform shift factor
    break
for d in [0,0.02,0.04,0.06,0.08]:
    th2=th.copy(); th2[3:]-=d*np.diff(G)[3:]  # lower thresholds in asymmetric region (>=1.75)
    rep(f"E2M1 RTN thr-shift {d}",rtn(t,G,th2))
