"""Fit the stepped 3D tyre (tyre_ring3d.py) with the SAME protocol as tyre_models_fast.py (train clips 1-2, test 3-4, leave-one-clip-out; events S2).
Free parameters: kb (lip-zone bending stiffness), release speed, rest level. Wall profile is NOT fitted: t_lip = 11.8 mm measured, t_main is a sensitivity value.
Run from ../data:  PYTHONPATH=../code python ../code/tyre_ring3d_fit.py"""
import glob,json,sys,numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
from tyre_fit2 import events
import tyre_ring3d as T3
ok=lambda e:(not e['held']) and e['dur']>0.3
rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
def pred(e,q,prof):
    kb=np.exp(q[0]);t,y=T3.simulate(min(.99,e['z0']/q[2]),kb,q[1],T=float(e['t'][-1])+0.02,prof=prof);return q[2]*np.interp(e['t'],t,y)
def fit(Es,prof):
    f=lambda q:np.concatenate([pred(e,q,prof)-e['z'] for e in Es]);best=None
    for q0 in [[np.log(k),v,0.98] for k in (1.0,2.5,6) for v in (3,20)]:
        r=least_squares(f,q0,bounds=([np.log(.3),1,.85],[np.log(60),300,1.1]),loss='soft_l1',f_scale=0.05,diff_step=0.02,max_nfev=40)
        if best is None or r.cost<best.cost:best=r
    return best.x
def task(a):
    tm,tag,trn,tst=a;prof=T3.profile(tm) if tm else np.ones(5);q=fit(trn,prof)
    return tm,tag,q,[rmse(pred(e,q,prof),e['z']) for e in tst],[rmse(pred(e,q,prof),e['z']) for e in trn]
if __name__=='__main__':
    files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[[e for e in events(f,i+1) if ok(e)] for i,f in enumerate(files)]
    tr=ALL[0]+ALL[1];te=ALL[2]+ALL[3];jobs=[]
    for tm in (0,11.8,9.5,7.0):   # 0 = uniform (=2D ring); 11.8 = uniform wall (check); others stepped
        jobs.append((tm,'main',tr,te))
        for h in range(4):jobs.append((tm,h,[e for k in range(4) if k!=h for e in ALL[k]],ALL[h]))
    with ProcessPoolExecutor() as ex:out=list(ex.map(task,jobs))
    R={}
    for tm,tag,q,tee,trr in out:
        d=R.setdefault(str(tm),{'loco_events':[]})
        if tag=='main':d.update(train=float(np.mean(trr)),test=float(np.mean(tee)),params=[float(x) for x in q],kb=float(np.exp(q[0])))
        else:d['loco_events']+=tee
    print('t_main(mm)  train   held-out  LOCO    kb      speed   rest')
    for k,d in R.items():
        d['loco']=float(np.mean(d['loco_events']));print('%-10s %.4f  %.4f   %.4f  %.2f   %.2f   %.3f'%(k,d['train'],d['test'],d['loco'],d['kb'],d['params'][1],d['params'][2]))
    json.dump(R,open('../results/tyre_ring3d.json','w'),indent=1)
