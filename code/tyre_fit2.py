import glob,sys,json,numpy as np
from scipy.signal import medfilt,find_peaks
from scipy.optimize import least_squares
def events(p,clip):
    d=np.load(p);t=d['t'];y=d['minor'];ok=np.isfinite(y);y=np.interp(t,t[ok],y[ok]);y=medfilt(y,13);z=y/np.percentile(y,85)
    pk,_=find_peaks(-z,prominence=0.25,distance=240,height=-0.8);E=[]
    for k in pk:
        zm=z[k];on=k+np.max(np.where(z[k:k+240]<=zm+0.03)[0]);w=slice(on,min(on+288,len(z)))
        if w.stop-w.start<72: continue                       # need >=0.3 s of recovery
        tt=t[w]-t[on];zz=z[w];i5=min(len(zz)-1,120)
        held=(zz[i5]-zz[0])<0.5*(1-zz[0])*1.0 and len(zz)>120    # hand still gripping at 0.5 s (not a free recovery)
        i0=k
        while i0>0 and z[i0]<0.85: i0-=1
        E.append(dict(dur=float(t[on]-t[i0]),clip=clip,t=tt,z=zz,z0=zz[0],held=bool(held),t_abs=float(t[on])))
    return E
# models: z(t)= r-(r-z0)*g(t)
def g_exp(t,p): return np.exp(-t/p[0])
def g_sls(t,p): return p[1]*np.exp(-t/p[0])+(1-p[1])*np.exp(-t/p[2])
def g_osc(t,p):
    wn,ze=p[0],p[1];s=np.sqrt(complex(ze**2-1+0j));r1=wn*(-ze+s);r2=wn*(-ze-s)
    return np.real((r2*np.exp(r1*t)-r1*np.exp(r2*t))/(r2-r1))
MOD={'1st-order (1 tau)':(g_exp,[0.15],[0.005],[2.0]),
     '2nd-order spring-damper':(g_osc,[40,2.0],[2,0.2],[400,12]),
     'SLS / Prony (2 tau)':(g_sls,[0.08,0.6,0.5],[0.005,0.0,0.05],[0.4,1.0,5.0])}
def pred(E,m,p,r,fixr):
    g=MOD[m][0];rr=1.0 if fixr else r
    return rr-(rr-E['z0'])*g(E['t'],p)
def fit(Es,m,fixr):
    g,p0,lb,ub=MOD[m];n=len(p0)
    f=lambda q:np.concatenate([pred(E,m,q[:n],q[n] if not fixr else 1.0,fixr)-E['z'] for E in Es])
    x0=list(p0)+([] if fixr else [0.97]);lo=list(lb)+([] if fixr else [0.8]);hi=list(ub)+([] if fixr else [1.1])
    r=least_squares(f,x0,bounds=(lo,hi),loss='soft_l1',f_scale=0.05);return r.x
rm=lambda E,m,q,fixr:float(np.sqrt(np.mean((pred(E,m,q[:len(MOD[m][1])],1.0 if fixr else q[-1],fixr)-E['z'])**2)))
if __name__=='__main__':
    files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[events(f,i+1) for i,f in enumerate(files)]
    flat=[e for A in ALL for e in A];print('events extracted:',[len(A) for A in ALL],'| flagged hand-held:',sum(e['held'] for e in flat))
    tr=[e for e in ALL[0]+ALL[1] if not e['held']];te=[e for e in ALL[2]+ALL[3] if not e['held']];te_all=ALL[2]+ALL[3]
    out={}
    out['null (no recovery)']=dict(held=np.mean([np.sqrt(np.mean((e['z0']-e['z'])**2)) for e in te]),heldall=np.mean([np.sqrt(np.mean((e['z0']-e['z'])**2)) for e in te_all]))
    rows=[]
    for m in MOD:
        for fixr in (True,False):
            q=fit(tr,m,fixr);n=len(MOD[m][1])
            tr_r=np.mean([rm(e,m,q,fixr) for e in tr]);te_r=[rm(e,m,q,fixr) for e in te];ta=np.mean([rm(e,m,q,fixr) for e in te_all])
            # leave-one-clip-out (all 4 clips)
            cv=[]
            for h in range(4):
                trn=[e for k in range(4) if k!=h for e in ALL[k] if not e['held']];qq=fit(trn,m,fixr)
                cv+= [rm(e,m,qq,fixr) for e in ALL[h] if not e['held']]
            rows.append(dict(model=m+('' if fixr else ' + fitted rest'),params=[round(float(x),3) for x in q],train=float(tr_r),test=float(np.mean(te_r)),test_med=float(np.median(te_r)),test_incl_held=float(ta),loco=float(np.mean(cv)),ntr=len(tr),nte=len(te)))
    json.dump(dict(rows=rows,null=out,ntr=len(tr),nte=len(te)),open('tyre_fit2.json','w'),indent=1,default=float)
    print('train events',len(tr),'test events',len(te),'null RMSE',round(out['null (no recovery)']['held'],3))
    for r in rows: print('%-44s train %.3f  test %.3f (med %.3f)  test+held %.3f  LOCO %.3f  p=%s'%(r['model'],r['train'],r['test'],r['test_med'],r['test_incl_held'],r['loco'],r['params']))
