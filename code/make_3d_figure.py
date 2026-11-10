"""Held-out event overlay + 3D tyre with MEASURED geometry: outer diameter 75.8 mm (marker/top view), tread width 37 mm (caliper, user-measured),
radial wall 11.8 mm (profile through the standing tyre in tyre_deformation.mov: 38-39 px walls of a 246 px outer diameter)."""
import glob,json,sys,numpy as np
from tyre_fit2 import events
import tyre_ring as T
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
kb,vh,r=json.load(open('../results/tyre_ring_fit.json'))['params']['ring'];D=75.8;W=37.0;WALL=11.8
ALL=[events(f,i+1) for i,f in enumerate(sorted(glob.glob('tyreN_sig_*.npz')))]
e=[x for x in ALL[3] if abs(x['t_abs']-5.7)<0.3][0]
fig=plt.figure(figsize=(17,7));a=fig.add_subplot(2,1,1)
t,y,_=T.simulate(min(.99,e['z0']/r),kb,vh,T=1.2)
a.plot(e['t'],e['z']*D,'k.',ms=3,label='real (video)');a.plot(t,r*y*D,'r',label='ring simulation (fitted on tyre_1, tyre_2 only)')
a.set_xlabel('time since release (s)');a.set_ylabel('tyre width (mm)');a.grid();a.legend();a.set_title('Held-out event: tyre_4 @ %.1fs'%e['t_abs'])
def inner(X,t_mm):
    c=X.mean(0);Xp=np.roll(X,-1,0)-np.roll(X,1,0);n=np.stack([Xp[:,1],-Xp[:,0]],1);n/=np.linalg.norm(n,axis=1)[:,None]
    n*=np.sign(((c-X)*n).sum(1))[:,None];return X+n*t_mm
for i,tt in enumerate([0,0.05,0.15,0.5]):
    _,_,X=T.simulate(min(.99,e['z0']/r),kb,vh,T=max(tt,1e-3)+1e-3);Xo=X*D;Xi=inner(Xo,WALL);n=len(Xo)
    ax=fig.add_subplot(2,4,5+i,projection='3d')
    for j in range(n):
        k=(j+1)%n;q=lambda P,Q,z0,z1:[(P[0],P[1],z0),(Q[0],Q[1],z0),(Q[0],Q[1],z1),(P[0],P[1],z1)]
        for poly,col in ((q(Xo[j],Xo[k],-W/2,W/2),'0.22'),(q(Xi[j],Xi[k],-W/2,W/2),'0.45'),([(Xo[j][0],Xo[j][1],W/2),(Xo[k][0],Xo[k][1],W/2),(Xi[k][0],Xi[k][1],W/2),(Xi[j][0],Xi[j][1],W/2)],'0.35')):
            ax.add_collection3d(Poly3DCollection([poly],facecolor=col,edgecolor='0.55',linewidths=.2))
    g=(min(.99,e['z0']/r)+vh*tt)*D
    for s in(1,-1):ax.add_collection3d(Poly3DCollection([[(-60,s*g/2,-25),(60,s*g/2,-25),(60,s*g/2,25),(-60,s*g/2,25)]],facecolor='tab:orange',alpha=.25))
    ax.set_xlim(-60,60);ax.set_ylim(-60,60);ax.set_zlim(-40,40);ax.set_box_aspect((1,1,.5));ax.view_init(35,-60)
    ax.set_title('t=%.2fs  width %.0f mm'%(tt,Xo[:,1].max()-Xo[:,1].min()),y=.95)
plt.tight_layout();plt.savefig('../figures/heldout_event_and_3d_ring.png',dpi=60);print('ok')
