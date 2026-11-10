import glob,json,sys,numpy as np
from scipy.optimize import least_squares
from tyre_fit2 import events,fit,rm,MOD
import tyre_ring as T
files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[events(f,i+1) for i,f in enumerate(files)]
ok=lambda e:(not e['held']) and e['dur']>0.3          # a real press-hold-release (not a <0.3 s flick, not hand still gripping)
print('durations of squeezes:',np.round([e['dur'] for A in ALL for e in A],2))
def ring_pred(e,q):
    kb,vh,r=np.exp(q[0]),q[1],q[2]
    t,y,_=T.simulate(min(0.99,e['z0']/r),kb,vh,T=float(e['t'][-1])+0.02);return r*np.interp(e['t'],t,y)
def ring_fit(Es):
    f=lambda q:np.concatenate([ring_pred(e,q)-e['z'] for e in Es]);best=None
    for q0 in ([np.log(1.0),10,0.98],[np.log(4.0),40,0.98]):
        r=least_squares(f,q0,bounds=([np.log(0.3),1,0.85],[np.log(30),300,1.1]),loss='soft_l1',f_scale=0.05,diff_step=0.05,max_nfev=40)
        if best is None or r.cost<best.cost:best=r
    return best.x
R=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
tr=[e for e in ALL[0]+ALL[1] if ok(e)];te=[e for e in ALL[2]+ALL[3] if ok(e)]
print('train',len(tr),'test',len(te))
qr=ring_fit(tr);print('ring params: kb=%.2f vh=%.1f /s r=%.3f'%(np.exp(qr[0]),qr[1],qr[2]))
m='1st-order (1 tau)';q1=fit(tr,m,False)
from tyre_fit2 import pred
rows={'null':lambda e:R(e['z0'],e['z']),'first':lambda e:R(pred(e,m,q1[:1],q1[1],False),e['z']),'ring':lambda e:R(ring_pred(e,qr),e['z'])}
res={k:dict(train=float(np.mean([f(e) for e in tr])),test=float(np.mean([f(e) for e in te])),med=float(np.median([f(e) for e in te]))) for k,f in rows.items()}
# leave-one-clip-out
loco={'first':[],'ring':[]}
for h in range(4):
    trn=[e for k in range(4) if k!=h for e in ALL[k] if ok(e)];tst=[e for e in ALL[h] if ok(e)]
    qa=fit(trn,m,False);qb=ring_fit(trn)
    loco['first']+=[R(pred(e,m,qa[:1],qa[1],False),e['z']) for e in tst];loco['ring']+=[R(ring_pred(e,qb),e['z']) for e in tst]
res['loco']={k:float(np.mean(v)) for k,v in loco.items()};res['params']=dict(ring=[float(np.exp(qr[0])),float(qr[1]),float(qr[2])],first=[float(x) for x in q1])
json.dump(res,open('tyre_ring_fit.json','w'),indent=1);print(json.dumps(res,indent=1))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,4,figsize=(17,7))
for a,e in zip(ax.flat,te):
    a.plot(e['t'],e['z'],'k.',ms=2,label='real');a.plot(e['t'],ring_pred(e,qr),'r',label='ring sim');a.plot(e['t'],pred(e,m,q1[:1],q1[1],False),'b--',lw=1,label='1st-order');a.grid();a.set_title('tyre_%d @%.1fs rmse %.3f'%(e['clip'],e['t_abs'],R(ring_pred(e,qr),e['z'])))
ax.flat[0].legend();plt.tight_layout();plt.savefig(sys.argv[1],dpi=60)
