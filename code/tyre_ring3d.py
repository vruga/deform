"""Coarse 3D tyre: a stack of M rings (slices) along the tread width W=37 mm (=0.488 diameters), each ring = the 16-node bending ring of
tyre_ring_fast, neighbouring slices tied node-to-node by 3D axial springs (stiffness KA). Nodes have x,y (in the ring plane) and z (axial).
Finger plates press only slices within +-cw*W/2 of the tread centre (cw=1: full-width contact, like the tongs). Overdamped, same constants as the 2D ring.
Observable = silhouette width seen along the axis (top view): max y - min y over ALL nodes.
NOT modelled: out-of-plane bending, tread lugs, friction, true solid wall (a thin shell proxy). With cw=1 the deformation is uniform along z
and the model reduces to the 2D ring (checked in __main__: max difference 1.2e-3 diameters = 0.09 mm, from time step and sampling)."""
import numpy as np
from numba import njit
import tyre_ring_fast as RF
N=RF.N;KS=RF.KS;KC=RF.KC;W=37.0/75.8
@njit(cache=True)
def sim3(g0,kbs,vh,T,t_close,cw,ka,M,X0,L0):
    n=X0.shape[0];h=W/(M-1);X=np.zeros((M,n,3));F=np.zeros((M,n,3))
    for s in range(M):
        for i in range(n):X[s,i,0]=X0[i,0];X[s,i,1]=X0[i,1];X[s,i,2]=-W/2+s*h
    kb=kbs.max();dt=min(0.4/(4*KS+110*kb+100+4*ka),1e-3);n_close=int(t_close/dt);ns=int(T/dt);every=max(1,int(1/240/dt))
    F2=np.zeros((n,2));X2=np.zeros((n,2))
    cnt=ns//every+1;ts=np.empty(cnt);ys=np.empty(cnt);k=0
    for step in range(n_close+ns):
        if step<n_close:g=1.0+(g0-1.0)*min(1.0,(step*dt)/(0.6*t_close))
        else:g=g0+vh*(step-n_close)*dt
        for s in range(M):
            for i in range(n):X2[i,0]=X[s,i,0];X2[i,1]=X[s,i,1]
            RF._forces(X2,F2,kbs[s],1e9,L0)                   # in-plane stretch+bending only (plates disabled by huge gap)
            act=abs(-W/2+s*h)<=cw*W/2+1e-9
            for i in range(n):
                F[s,i,0]=F2[i,0];F[s,i,1]=F2[i,1];F[s,i,2]=0.0
                if act:
                    top=X[s,i,1]-g/2;bot=-g/2-X[s,i,1]
                    if top>0:F[s,i,1]-=KC*top
                    if bot>0:F[s,i,1]+=KC*bot
        for s in range(M-1):
            for i in range(n):
                dx=X[s+1,i,0]-X[s,i,0];dy=X[s+1,i,1]-X[s,i,1];dz=X[s+1,i,2]-X[s,i,2];l=np.sqrt(dx*dx+dy*dy+dz*dz);f=ka*(l-h)/l
                F[s,i,0]+=f*dx;F[s,i,1]+=f*dy;F[s,i,2]+=f*dz;F[s+1,i,0]-=f*dx;F[s+1,i,1]-=f*dy;F[s+1,i,2]-=f*dz
        for s in range(M):
            for i in range(n):
                X[s,i,0]+=dt*F[s,i,0];X[s,i,1]+=dt*F[s,i,1];X[s,i,2]+=dt*F[s,i,2]
        if step>=n_close and (step-n_close)%every==0 and k<cnt:
            y1=-1e9;y0=1e9
            for s in range(M):
                for i in range(n):
                    if X[s,i,1]>y1:y1=X[s,i,1]
                    if X[s,i,1]<y0:y0=X[s,i,1]
            ts[k]=(step-n_close)*dt;ys[k]=y1-y0;k+=1
    return ts[:k],ys[:k]
def profile(t_main=9.5,t_lip=11.8,n_lip=2,M=5):
    """relative bending stiffness per slice ~ (wall/t_lip)^3: n_lip slices at the lip wall, the rest at t_main (mm)."""
    return np.array([1.0]*n_lip+[(t_main/t_lip)**3]*(M-n_lip))
def simulate(g0,kb,vh,T=1.2,t_close=0.3,cw=1.0,ka=100.0,M=5,prof=None):
    prof=np.ones(M) if prof is None else prof
    return sim3(float(g0),float(kb)*np.asarray(prof,float),float(vh),float(T),float(t_close),float(cw),float(ka),int(M),RF.X0,RF.L0)
if __name__=='__main__':
    for a in [(0.5,1.74,2.7),(0.6,3.0,10.0)]:
        t2,y2,_=RF.simulate(*a);t3,y3=simulate(*a,cw=1.0);m=min(len(t2),len(t3));print('cw=1 vs 2D ring',a,'max|dy| %.2e'%np.abs(y2[:m]-y3[:m]).max())
        t4,y4=simulate(*a,cw=0.4);print('   cw=0.4: silhouette at release %.3f vs 2D %.3f ; at 0.3 s %.3f vs %.3f'%(y4[0],y2[0],y4[min(72,len(y4)-1)],y2[min(72,len(y2)-1)]))
