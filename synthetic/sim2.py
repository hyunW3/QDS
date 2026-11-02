import numpy as np
rng=np.random.default_rng(1)
G=np.array([0,.5,1,1.5,2,3,4,6.])
def rtn(t): return G[np.searchsorted((G[1:]+G[:-1])/2,t)]
def sr(t):
    i=np.clip(np.searchsorted(G,t,side='right')-1,0,6);a,b=G[i],G[i+1]
    return np.where(rng.random(t.shape)<(t-a)/(b-a),b,a)
for B in [16,32,128]:
    X=rng.standard_normal((3_200_000//B,B)); s=np.abs(X).max(1,keepdims=True)/6; t=np.abs(X)/s
    for nm,f in [("RTN",rtn),("SR",sr)]:
        Ah=np.sign(X)*f(t)*s; a=(Ah*X).sum()/(X*X).sum(); r=((Ah-X)**2).sum()/(X*X).sum()
        print(f"B={B:3d} {nm}: alpha={a:.4f} SQNR={-10*np.log10(r):.2f}dB")
