"""All tyre recovery models, one protocol, fast. Run from experiment_1/data:  PYTHONPATH=../code python ../code/tyre_models_fast.py
Models (z = width / resting width; every parameter shared by all events; an event supplies only its own measured start level z0):
  first    : z = r - (r - z0) exp(-t/tau)                                    params tau, r
  nlfirst  : strain-dependent viscosity, x' = -x (x/0.5)^p / tau              params tau, r, p
  ringA    : nonlinear ring + finger plates, shared release speed             params kb, speed, r
  ringB    : release speed = exp(a + b*depth)                                 params kb, a, b, r
  ringC    : ring + strain-dependent drag (p)                                 params kb, speed, r, p
  ringE    : fingers accelerate to the release speed (time constant ta)       params kb, speed, r, ta
Protocol: train on clips 1-2, test on clips 3-4 (events S2: no flicks < 0.3 s, no hand-gripped event);
leave-one-clip-out (train on 3 clips, test on the 4th, rotating); paired bootstrap over the held-out events vs 'first'."""
import glob,json,sys,time,numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
from tyre_fit2 import events
import tyre_ring_fast as RF
ok=lambda e:(not e['held']) and e['dur']>0.3
rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
def pred(m,e,q):
    if m=='first':return q[1]-(q[1]-e['z0'])*np.exp(-e['t']/q[0])
    if m=='nlfirst':
        tau,r,p=q;x0=max(r-e['z0'],1e-3)
        x=x0*np.exp(-e['t']/tau) if p<1e-4 else (x0**(-p)+p*e['t']/(tau*0.5**p))**(-1/p);return r-x
    kb=np.exp(q[0]);T=float(e['t'][-1])+0.02
    ta=0.0
    if m=='ringA':v,r,p=q[1],q[2],0.0
    elif m=='ringE':v,r,p,ta=q[1],q[2],0.0,q[3]
    elif m=='ringB':v,r,p=np.exp(q[1]+q[2]*(1-e['z0'])),q[3],0.0
    else:v,r,p=q[1],q[2],q[3]
    t,y,_=RF.simulate(min(.99,e['z0']/r),kb,v,T=T,p=p,ta=ta);return r*np.interp(e['t'],t,y)
SPEC={ # starts (list of q0), lower, upper
 'first':([[0.14,0.98],[0.3,0.97]],[0.005,0.85],[3,1.1]),
 'nlfirst':([[0.14,0.98,0.3],[0.1,0.98,1.5],[0.3,0.97,0.05]],[0.005,0.85,0],[3,1.1,4]),
 'ringA':([[np.log(k),v,0.98] for k in (0.7,1.7,4,10) for v in (3,20)],[np.log(.3),1,.85],[np.log(60),300,1.1]),
 'ringB':([[np.log(k),a,b,0.98] for k in (0.7,1.7,4) for a,b in ((1,0),(0,6))],[np.log(.3),-2,-10,.85],[np.log(60),6,30,1.1]),
 'ringE':([[np.log(k),v,0.98,ta] for k in (0.7,1.7,4) for v in (5,30) for ta in (0.03,0.15)],[np.log(.3),1,.85,0.005],[np.log(60),500,1.1,0.6]),
 'ringC':([[np.log(k),v,0.98,p] for k in (0.7,1.7,4) for v in (3,20) for p in (0.3,1.5)],[np.log(.3),1,.85,0],[np.log(60),300,1.1,4])}
def fit(m,Es):
    S,lb,ub=SPEC[m];f=lambda q:np.concatenate([pred(m,e,q)-e['z'] for e in Es]);best=None
    for q0 in S:
        r=least_squares(f,q0,bounds=(lb,ub),loss='soft_l1',f_scale=0.05,diff_step=0.02,max_nfev=60)
        if best is None or r.cost<best.cost:best=r
    return best.x
def task(args):
    m,trn,tst=args;q=fit(m,trn);return m,q,[rmse(pred(m,e,q),e['z']) for e in tst],[rmse(pred(m,e,q),e['z']) for e in trn]
if __name__=='__main__':
    t0=time.time();files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[[e for e in events(f,i+1) if ok(e)] for i,f in enumerate(files)]
    tr=ALL[0]+ALL[1];te=ALL[2]+ALL[3];M=list(SPEC);jobs=[]
    for m in M:
        jobs.append((m,'main',tr,te))
        for h in range(4):jobs.append((m,h,[e for k in range(4) if k!=h for e in ALL[k]],ALL[h]))
    with ProcessPoolExecutor() as ex:out=list(ex.map(task,[(m,a,b) for m,_,a,b in jobs]))
    res={m:{'loco_events':[]} for m in M}
    for (m,tag,_,_),(_,q,te_err,tr_err) in zip(jobs,out):
        if tag=='main':res[m].update(train=float(np.mean(tr_err)),test=float(np.mean(te_err)),test_med=float(np.median(te_err)),params=[float(x) for x in q],per_event_test=te_err)
        else:res[m]['loco_events']+=te_err
    for m in M:res[m]['loco']=float(np.mean(res[m]['loco_events']))
    # paired bootstrap of the LOCO error difference vs 'first' (resampling events)
    rng=np.random.RandomState(0);base=np.array(res['first']['loco_events'])
    for m in M:
        d=np.array(res[m]['loco_events'])-base
        bs=[d[rng.randint(0,len(d),len(d))].mean() for _ in range(4000)];res[m]['loco_diff_vs_first']=[float(d.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
    res['n']=dict(train=len(tr),test=len(te),loco_events=len(base),seconds=time.time()-t0)
    json.dump(res,open('tyre_models_fast.json','w'),indent=1)
    print('events: train %d test %d LOCO %d | %.0f s'%(len(tr),len(te),len(base),time.time()-t0))
    print('%-8s %6s %6s %7s %6s   LOCO error minus first-order [95%% CI]   params'%('model','train','test','t-med','LOCO'))
    for m in M:
        r=res[m];d=r['loco_diff_vs_first'];print('%-8s %.4f %.4f %.4f %.4f   %+.4f [%+.4f, %+.4f]   %s'%(m,r['train'],r['test'],r['test_med'],r['loco'],d[0],d[1],d[2],np.round(r['params'],3)))
