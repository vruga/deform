"""Release-speed ablation. Same protocol as tyre_models_fast.py (train clips 1-2 / test 3-4 / leave-one-clip-out, events S2).
ringI  : ring with INSTANT release (plates vanish at t=0, speed fixed at 300 diam/s = no release parameter)  params kb, rest
ringS  : plates retract at a FIXED speed v (not fitted), kb and rest fitted, for v in a grid
ringA  : the fitted-speed ring from tyre_models_fast.py (kb, speed, rest), for reference.
Run from ../data:  PYTHONPATH=../code python ../code/tyre_no_release_param.py"""
import glob,json,numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
from tyre_fit2 import events
import tyre_ring_fast as RF
ok=lambda e:(not e['held']) and e['dur']>0.3
rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
def pred(e,q,v):
    t,y,_=RF.simulate(min(.99,e['z0']/q[1]),float(np.exp(q[0])),v,T=float(e['t'][-1])+0.02);return q[1]*np.interp(e['t'],t,y)
def fit(Es,v):
    f=lambda q:np.concatenate([pred(e,q,v)-e['z'] for e in Es]);best=None
    for q0 in [[np.log(k),0.98] for k in (0.5,1.5,4,10)]:
        r=least_squares(f,q0,bounds=([np.log(.05),.85],[np.log(80),1.1]),loss='soft_l1',f_scale=0.05,diff_step=0.02,max_nfev=60)
        if best is None or r.cost<best.cost:best=r
    return best.x
def task(a):
    v,tag,trn,tst=a;q=fit(trn,v);return v,tag,q,[rmse(pred(e,q,v),e['z']) for e in tst],[rmse(pred(e,q,v),e['z']) for e in trn]
if __name__=='__main__':
    files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[[e for e in events(f,i+1) if ok(e)] for i,f in enumerate(files)]
    tr=ALL[0]+ALL[1];te=ALL[2]+ALL[3];jobs=[]
    for v in (300.0,20.0,8.0,4.0,2.72,1.5):
        jobs.append((v,'main',tr,te))
        for h in range(4):jobs.append((v,h,[e for k in range(4) if k!=h for e in ALL[k]],ALL[h]))
    with ProcessPoolExecutor() as ex:out=list(ex.map(task,jobs))
    R={}
    for v,tag,q,tee,trr in out:
        d=R.setdefault(str(v),{'loco_events':[]})
        if tag=='main':d.update(train=float(np.mean(trr)),test=float(np.mean(tee)),kb=float(np.exp(q[0])),rest=float(q[1]))
        else:d['loco_events']+=tee
    print('release speed (diam/s, FIXED)   train   held-out  LOCO    kb     rest')
    for k,d in R.items():d['loco']=float(np.mean(d['loco_events']));print('%-10s %.4f  %.4f   %.4f  %.2f  %.3f'%(k,d['train'],d['test'],d['loco'],d['kb'],d['rest']))
    json.dump(R,open('../results/tyre_no_release_param.json','w'),indent=1)
