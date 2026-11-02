import numpy as np
from scipy.linalg import hadamard
rng=np.random.default_rng(2)
G=np.array([0,.5,1,1.5,2,3,4,6.]); U=np.arange(8)*6/7
B=16; T,C=8192,1024
def rtn(t,g,th=None):
    th=(g[1:]+g[:-1])/2 if th is None else th; return g[np.searchsorted(th,t)]
def sr(t,g):
    i=np.clip(np.searchsorted(g,t,side='right')-1,0,len(g)-2);a,b=g[i],g[i+1]
    return np.where(rng.random(t.shape)<(t-a)/(b-a),b,a)
TH=(G[1:]+G[:-1])/2; TH2=TH.copy(); TH2[3:]-=0.02*np.diff(G)[3:]
H=hadamard(B)/np.sqrt(B)
def quant(X,f):
    Xb=X.reshape(-1,B); s=np.abs(Xb).max(1,keepdims=True)/6; s[s==0]=1
    return (np.sign(Xb)*f(np.abs(Xb)/s)*s).reshape(X.shape)
def dists():
    yield "Gaussian", rng.standard_normal((T,C))
    yield "Laplace", rng.laplace(size=(T,C))
    yield "Student-t df5", rng.standard_t(5,(T,C))
    yield "Student-t df3", rng.standard_t(3,(T,C))
    X=rng.standard_normal((T,C)); ch=rng.choice(C,C//100,replace=False); X[:,ch]*=20
    yield "Gauss+1%ch x20", X
    X=rng.standard_normal((T,C)); m=rng.random((T,C))<0.001; X[m]*=50
    yield "Gauss+0.1%pt x50", X
meth=[("E2M1 RTN",lambda t:rtn(t,G)),("E2M1 SR",lambda t:sr(t,G)),
      ("E2M1 thr-2%",lambda t:rtn(t,G,TH2)),("UNIF RTN",lambda t:rtn(t,U))]
RES=[]
print(f"{'dist':18s}{'RHT':4s} | "+" | ".join(f"{m:>11s} a / dB " for m,_ in meth)+" | E2M1 RTN+αcorr dB")
for name,X0 in dists():
    for rht in [0,1]:
        if rht:
            D=rng.choice([-1,1],C)
            X=((X0*D).reshape(T,C//B,B)@H).reshape(T,C)
        else: X=X0
        nx=(X*X).sum(); row=[]
        for m,f in meth:
            Ah=quant(X,f); a=(Ah*X).sum()/nx; r=((Ah-X)**2).sum()/nx
            row.append(f"{a:.4f} / {-10*np.log10(r):5.2f}")
            if m=="E2M1 RTN": Ar=Ah
        a=(Ar*X).sum()/nx; rc=((Ar/a-X)**2).sum()/nx
        RES.append(dict(dist=name,rht=bool(rht),rows=row,acorr_db=float(-10*np.log10(rc))))
        print(f"{name:18s}{'yes' if rht else 'no':4s} | "+" | ".join(f"{x:>19s}" for x in row)+f" | {-10*np.log10(rc):5.2f}")
import json, os; json.dump(RES,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"sim3_results.json"),"w"),indent=1)
