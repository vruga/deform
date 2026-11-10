"""Diagnostic: ring stiffness fixed at the training value, fit the finger release speed separately for each event.
Run from experiment_1/code (needs tyre_fit2.py, tyre_ring.py and ../results/tyre_ring_fit.json, ../data/tyreN_sig_*.npz)."""
import glob,json,numpy as np
from scipy.optimize import least_squares
from tyre_fit2 import events
import tyre_ring as T
kb,vh0,r=json.load(open('tyre_ring_fit.json'))['params']['ring']
files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[events(f,i+1) for i,f in enumerate(files)];out=[]
for A in ALL:
    for e in A:
        if e['held'] or e['dur']<=0.3: continue
        pr=lambda v:r*np.interp(e['t'],*T.simulate(min(.99,e['z0']/r),kb,v,T=float(e['t'][-1])+0.02)[:2])
        s=least_squares(lambda q:pr(np.exp(q[0]))-e['z'],[np.log(5)],bounds=([np.log(0.3)],[np.log(300)]),diff_step=0.1)
        v=np.exp(s.x[0]);a=np.sqrt(np.mean((pr(vh0)-e['z'])**2));b=np.sqrt(np.mean(s.fun**2));out.append((1-e['z0'],v,a,b))
        print(e['clip'],'%5.1f'%e['t_abs'],'depth %.2f'%(1-e['z0']),'speed %.2f diam/s (%.0f mm/s)'%(v,v*75.8),'rmse shared %.3f -> own %.3f'%(a,b))
o=np.array(out);print('corr(depth, log speed) = %.2f'%np.corrcoef(o[:,0],np.log(o[:,1]))[0,1])
