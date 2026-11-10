"""Physical ring model of the tyre. Cross-section = closed chain of N nodes (rest circle, R=0.5), quasi-inextensible stretch springs,
turning-angle bending energy, overdamped viscous dynamics c*v=F (c=1), squeezed between two finger plates that close then retract at speed vh.
Nonlinearity comes from the geometry (large deflection of a ring), not a fitted law. Extruded along the tread axis it is the 3D tube."""
import numpy as np
N=16;R=0.5;KS=100.0
th0=np.linspace(0,2*np.pi,N,endpoint=False);X0=np.stack([R*np.cos(th0),R*np.sin(th0)],1)
def lens(X): return np.linalg.norm(np.roll(X,-1,0)-X,axis=1)
L0=lens(X0)
def perp(a): return np.stack([-a[:,1],a[:,0]],1)
def forces(X,kb,g):
    e=np.roll(X,-1,0)-X                      # edge i: x_i -> x_{i+1}
    l=np.linalg.norm(e,axis=1);u=e/l[:,None]
    fs=KS*(l-L0)[:,None]*u                    # force on node i from edge i (toward i+1 when stretched)
    F=fs-np.roll(fs,1,0)
    e1=np.roll(e,1,0);e2=e                    # at node i: e1=x_i-x_{i-1}, e2=x_{i+1}-x_i
    cr=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0];dt_=e1[:,0]*e2[:,0]+e1[:,1]*e2[:,1];th=np.arctan2(cr,dt_)
    dth=th-(2*np.pi/N);ng=(kb*dth)[:,None]    # energy 0.5*kb*(th-th0)^2 per node
    l1=np.maximum(np.einsum('ij,ij->i',e1,e1),1e-9);l2=np.maximum(np.einsum('ij,ij->i',e2,e2),1e-9)
    g1=perp(e1)/l1[:,None];g2=perp(e2)/l2[:,None]       # dth/dx_{i-1}=g1, dth/dx_{i+1}=g2, dth/dx_i=-(g1+g2)
    Fb=np.zeros_like(X);Fb+= ng*(g1+g2)                 # -kb*dth*dth/dx_i
    Fb-= np.roll(ng*g1,-1,0)                            # node i-1 contribution from node i's angle -> goes to i-1
    Fb-= np.roll(ng*g2,1,0)                             # node i+1
    F=F+Fb
    y=X[:,1];top=y-g/2;bot=-g/2-y
    F[:,1]+= -100.0*np.maximum(top,0)+100.0*np.maximum(bot,0)   # plates (penalty, one-sided)
    return F
def simulate(g0,kb,vh,T=1.2,t_close=0.3,p=0.0):
    """returns t, minor-axis (ring rest diameter = 1) during release. g0 = gap at release (in rest diameters)."""
    X=X0.copy();dt=0.4/(4*KS+110*kb+100);dt=min(dt,1e-3)
    n_close=int(t_close/dt);ts=[];ys=[];ns=int(T/dt);every=max(1,int(1/240/dt))
    for s in range(n_close):
        g=1.0+(g0-1.0)*min(1,(s*dt)/(0.6*t_close));X=X+dt*forces(X,kb,g)
    for s in range(ns):
        g=g0+vh*s*dt
        # p>0: strain-dependent viscosity, recovery slows as the tyre approaches rest (rate ~ (deficit/0.5)^p); p=0 is the plain model
        f=(max(1.0-(X[:,1].max()-X[:,1].min()),0.03)/0.5)**p if p else 1.0
        X=X+dt*f*forces(X,kb,g)
        if s%every==0:ts.append(s*dt);ys.append(X[:,1].max()-X[:,1].min())
    return np.array(ts),np.array(ys),X
if __name__=='__main__':
    import time;t0=time.time();t,y,X=simulate(0.5,0.05,10.0);print('time',round(time.time()-t0,2),'s','y0',y[0].round(3),'y_end',y[-1].round(3),'len',len(y))
    for kb in (0.01,0.05,0.2):
        t,y,_=simulate(0.5,kb,50.0);i=np.argmax(y>0.5+0.63*(y[-1]-0.5)) if y[-1]>0.5 else 0;print('kb',kb,'tau63 ~',round(t[i],3),'y_end',y[-1].round(3))
