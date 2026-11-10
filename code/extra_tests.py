"""Extra checks on the tyre model. Run from ../data:  PYTHONPATH=../code python ../code/extra_tests.py
 A. second observable: the ring's predicted LENGTH (extent along the unsqueezed axis) against the measured major axis, on the held-out events, frozen parameters
 B. bootstrap over events: confidence intervals for the relaxation time tau (first-order) and the ring stiffness kb
 C. fractional (Mittag-Leffler) recovery, z = r - (r-z0) E_a(-(t/tau)^a): does a heavy tail help?   (alpha = 1 is the plain exponential)
 D. depth dependence of the recovery time: data vs what the ring itself predicts (instant release, and fitted release speed)"""
import glob,json,numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
from scipy.signal import medfilt
from tyre_fit2 import events
import tyre_ring_fast as RF
import tyre_models_fast as TM
ok=lambda e:(not e['held']) and e['dur']>0.3
KB,VH,REST=json.load(open('../results/tyre_ring_fit.json'))['params']['ring'];TAU,REST1=json.load(open('../results/tyre_ring_fit.json'))['params']['first']
def ring_traces(z0,T,speed=VH):
    g0=min(.99,z0/REST);n=RF.N;X=RF.X0.copy();F=np.zeros((n,2));dt=min(0.4/(4*RF.KS+110*KB+100),1e-3);t_close=0.3
    for s in range(int(t_close/dt)):
        g=1.0+(g0-1.0)*min(1.0,(s*dt)/(0.6*t_close));RF._forces(X,F,KB,g,RF.L0);X+=dt*F
    ts=[];wy=[];wx=[];nxt=0.0;s=0
    while s*dt<=T:
        tt=s*dt
        if tt>=nxt-1e-12:ts.append(tt);wy.append(X[:,1].max()-X[:,1].min());wx.append(X[:,0].max()-X[:,0].min());nxt+=1/240.
        RF._forces(X,F,KB,g0+speed*tt,RF.L0);X+=dt*F;s+=1
    return np.array(ts),np.array(wy),np.array(wx)
def major_window(e,files):
    d=np.load(files[e['clip']-1]);t=d['t'];y=d['major'];okk=np.isfinite(y);y=np.interp(t,t[okk],y[okk]);y=medfilt(y,13);z=y/np.percentile(y,85);on=int(round(e['t_abs']*240));return z[on:on+len(e['t'])]
def taskB(a):
    kind,idx,E=a;rng=np.random.RandomState(idx);S=[E[i] for i in rng.randint(0,len(E),len(E))]
    q=TM.fit(kind,S);return kind,(float(q[0]) if kind=='first' else float(np.exp(q[0])))
_ML={}
def ml_table(alphas,xs):
    import mpmath as mp;mp.mp.dps=60;out=np.zeros((len(alphas),len(xs)))
    for i,a in enumerate(alphas):
        for j,x in enumerate(xs):out[i,j]=float(mp.nsum(lambda k:(-x)**int(k)/mp.gamma(a*k+1),[0,mp.inf])) if x>0 else 1.0
    return out
if __name__=='__main__':
    files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[[e for e in events(f,i+1) if ok(e)] for i,f in enumerate(files)];E=[e for A in ALL for e in A];TE=ALL[2]+ALL[3];R={}
    # ---- A: length
    rows=[]
    for e in TE:
        ts,wy,wx=ring_traces(e['z0'],float(e['t'][-1])+0.02);m=major_window(e,files);L=np.interp(e['t'],ts,wx)/1.0;w=np.interp(e['t'],ts,wy)*REST
        L0=m[0];rest_len=np.median(m[-30:]);Lpred=L/L[-1]*rest_len   # model length normalised to the event's own settled length
        rows.append(dict(clip=e['clip'],t_abs=e['t_abs'],z0=float(e['z0']),len_start=float(m[0]),len_end=float(rest_len),rmse_model=float(np.sqrt(np.mean((Lpred-m)**2))),rmse_const=float(np.sqrt(np.mean((rest_len-m)**2))),rmse_start=float(np.sqrt(np.mean((m[0]-m)**2))),model_len_start=float(Lpred[0])))
    R['A_length']=rows;print('A: length on held-out events (major axis / resting): model vs measured')
    for r in rows:print('  clip %d t=%.1f z0=%.2f  measured start %.3f -> settled %.3f | model start %.3f | rmse model %.3f  const(settled) %.3f  const(start) %.3f'%(r['clip'],r['t_abs'],r['z0'],r['len_start'],r['len_end'],r['model_len_start'],r['rmse_model'],r['rmse_const'],r['rmse_start']))
    print('  mean rmse: model %.3f  settled-constant %.3f  start-constant %.3f'%tuple(np.mean([r[k] for r in rows]) for k in ('rmse_model','rmse_const','rmse_start')))
    # ---- B: bootstrap
    jobs=[(k,i,E) for k in ('first','ringA') for i in range(100)]
    with ProcessPoolExecutor() as ex:out=list(ex.map(taskB,jobs))
    for k in ('first','ringA'):
        v=np.array([o[1] for o in out if o[0]==k]);R['B_'+k]=dict(median=float(np.median(v)),ci95=[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]);print('B: %s %s median %.3f 95%% CI [%.3f, %.3f]'%(k,'tau (s)' if k=='first' else 'kb',np.median(v),*np.percentile(v,[2.5,97.5])))
    # ---- C: Mittag-Leffler
    al=np.linspace(0.35,1.0,14);xs=np.concatenate([[0],np.geomspace(1e-3,80,160)]);T=ml_table(al,xs)
    def Ea(a,x):
        i=np.clip(np.searchsorted(al,a)-1,0,len(al)-2);w=(a-al[i])/(al[i+1]-al[i]);row=(1-w)*T[i]+w*T[i+1];return np.interp(x,xs,row)
    def predML(e,q):tau,a,r=q;return r-(r-e['z0'])*Ea(a,(e['t']/tau)**a)
    def fitML(Es):
        f=lambda q:np.concatenate([predML(e,q)-e['z'] for e in Es]);best=None
        for q0 in ([0.14,1.0,0.98],[0.1,0.7,0.98],[0.3,0.5,0.97]):
            r=least_squares(f,q0,bounds=([0.005,0.35,0.85],[3,1.0,1.1]),loss='soft_l1',f_scale=0.05,max_nfev=80)
            if best is None or r.cost<best.cost:best=r
        return best.x
    rm=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    tr=ALL[0]+ALL[1];q=fitML(tr);te_ml=np.mean([rm(predML(e,q),e['z']) for e in TE]);qf=TM.fit('first',tr);te_f=np.mean([rm(TM.pred('first',e,qf),e['z']) for e in TE])
    R['C_fractional']=dict(tau=float(q[0]),alpha=float(q[1]),rest=float(q[2]),heldout=float(te_ml),heldout_first=float(te_f));print('C: Mittag-Leffler  tau %.3f alpha %.2f rest %.3f  held-out %.4f  (first-order %.4f)'%(q[0],q[1],q[2],te_ml,te_f))
    # ---- D: depth dependence
    def tau_free(t,z,z0):
        f=lambda q:REST1-(REST1-z0)*np.exp(-t/q[0])-z;return min((least_squares(f,[s],bounds=([0.005],[3])) for s in (.05,.15,.4)),key=lambda r:r.cost).x[0]
    z0s=np.array([e['z0'] for e in E]);td=np.array([tau_free(e['t'],e['z'],e['z0']) for e in E])
    def sim_tau(speed):
        out=[]
        for e in E:
            ts,wy,_=ring_traces(e['z0'],1.2,speed);tt=np.arange(0,1.2,1/240.);y=REST*np.interp(tt,ts,wy);out.append(tau_free(tt,y,y[0]))
        return np.array(out)
    ti=sim_tau(300.0);tf=sim_tau(VH)
    c=lambda a,b:float(np.corrcoef(a,b)[0,1])
    R['D_depth']=dict(corr_data=c(z0s,td),corr_ring_instant=c(z0s,ti),corr_ring_fitted_speed=c(z0s,tf),tau_range_data=[float(td.min()),float(td.max())],tau_range_instant=[float(ti.min()),float(ti.max())],tau_range_fitted=[float(tf.min()),float(tf.max())])
    print('D: correlation of recovery time with start width: data %.2f | ring instant release %.2f | ring fitted release %.2f'%(c(z0s,td),c(z0s,ti),c(z0s,tf)))
    print('   tau range (s): data %.3f-%.3f | ring instant %.3f-%.3f | ring fitted %.3f-%.3f'%(td.min(),td.max(),ti.min(),ti.max(),tf.min(),tf.max()))
    json.dump(R,open('../results/extra_tests.json','w'),indent=1)
