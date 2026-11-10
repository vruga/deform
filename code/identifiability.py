"""Profile likelihood / identifiability of the ring model from the squeeze videos (all 12 S2 events, 3 free parameters kb, release speed, rest).
Profile = fix one parameter on a grid, re-optimise the other two, record the fit error. A curve that rises sharply pins the parameter; a flat curve means the video
cannot determine it. Noise scale for judging 'flat': the bootstrap spread of the mean error, computed from per-event errors (about 0.001).
Run from ../data:  PYTHONPATH=../code python ../code/identifiability.py  -> ../figures/identifiability.png, ../results/identifiability.json"""
import glob,json,numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
from tyre_fit2 import events
import tyre_ring_fast as RF
ok=lambda e:(not e['held']) and e['dur']>0.3
def pred(e,kb,v,r):
    t,y,_=RF.simulate(min(.99,e['z0']/r),kb,v,T=float(e['t'][-1])+0.02);return r*np.interp(e['t'],t,y)
def cost(E,kb,v,r):return np.concatenate([pred(e,kb,v,r)-e['z'] for e in E])
def rm(x):return float(np.sqrt(np.mean(x**2)))
def prof(a):
    kind,val,E=a;best=None
    if kind=='kb':
        for v0 in (2.7,15.0):
            r=least_squares(lambda q:cost(E,val,np.exp(q[0]),q[1]),[np.log(v0),0.98],bounds=([np.log(.5),.85],[np.log(400),1.1]),loss='soft_l1',f_scale=.05,diff_step=.02,max_nfev=40)
            if best is None or r.cost<best.cost:best=r
        return kind,val,rm(cost(E,val,np.exp(best.x[0]),best.x[1]))
    for k0 in (0.8,1.7,5.0):
        r=least_squares(lambda q:cost(E,np.exp(q[0]),val,q[1]),[np.log(k0),0.98],bounds=([np.log(.2),.85],[np.log(80),1.1]),loss='soft_l1',f_scale=.05,diff_step=.02,max_nfev=40)
        if best is None or r.cost<best.cost:best=r
    return kind,val,rm(cost(E,np.exp(best.x[0]),val,best.x[1]))
if __name__=='__main__':
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    files=sorted(glob.glob('tyreN_sig_*.npz'));E=[e for i,f in enumerate(files) for e in events(f,i+1) if ok(e)]
    kbs=np.exp(np.linspace(np.log(0.4),np.log(12),22));vs=np.exp(np.linspace(np.log(1.0),np.log(300),20))
    jobs=[('kb',float(k),E) for k in kbs]+[('v',float(v),E) for v in vs]
    with ProcessPoolExecutor() as ex:out=list(ex.map(prof,jobs))
    K=np.array([(v,c) for k,v,c in out if k=='kb']);V=np.array([(v,c) for k,v,c in out if k=='v'])
    # bootstrap noise of the mean error: events resampled, error of a mid model
    base=min(K[:,1].min(),V[:,1].min());pe=np.array([rm(cost([e],1.74,2.72,.979)) for e in E]);rng=np.random.RandomState(0);bs=np.std([rng.choice(pe,len(pe)).mean() for _ in range(2000)])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].semilogx(K[:,0],K[:,1]-base,'o-',color='#c0392b');ax[0].axhspan(0,bs,color='gray',alpha=.25,label='noise (bootstrap s.d., %.4f)'%bs);ax[0].set_xlabel('bending stiffness / drag, kb (ring units)');ax[0].set_ylabel('fit error above best (RMSE of width)');ax[0].set_title('Relaxation scale (stiffness : damping): pinned');ax[0].legend();ax[0].grid(alpha=.3)
    ax[1].semilogx(V[:,0],V[:,1]-base,'o-',color='#2c3e50');ax[1].axhspan(0,bs,color='gray',alpha=.25);ax[1].set_xlabel('plate release speed (diameters/s)');ax[1].set_title('Release speed: not identified above ~2.5');ax[1].grid(alpha=.3)
    plt.tight_layout();plt.savefig('../figures/identifiability.png',dpi=110)
    ok_k=K[K[:,1]-base<bs][:,0];ok_v=V[V[:,1]-base<bs][:,0]
    res=dict(noise=float(bs),best=float(base),kb_within_noise=[float(ok_k.min()),float(ok_k.max())],speed_within_noise=[float(ok_v.min()),float(ok_v.max())],kb_profile=K.tolist(),speed_profile=V.tolist())
    json.dump(res,open('../results/identifiability.json','w'),indent=1);print('best',base,'noise',bs);print('kb within noise',res['kb_within_noise'],' speed within noise',res['speed_within_noise'])
