import cv2,numpy as np,os,glob
from scipy.signal import medfilt,find_peaks
def run(path,fps=240.):
    name=os.path.basename(path)[:-4].replace('-','_'); c=cv2.VideoCapture(path); rows=[]
    while True:
        ok,f=c.read()
        if not ok: break
        s=cv2.resize(f,(480,270)); v=cv2.cvtColor(s,cv2.COLOR_BGR2HSV)[...,2]
        m=(v<0.3*np.percentile(v,80)).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cnt=[q for q in cnt if cv2.contourArea(q)>400]
        if not cnt: rows.append([np.nan]*4); continue
        q=max(cnt,key=cv2.contourArea); (cx,cy),(w,h),_=cv2.minAreaRect(q); rows.append([min(w,h),max(w,h),cv2.contourArea(q),cx])
    r=np.array(rows); t=np.arange(len(r))/fps
    np.savez(f'tyreN_sig_{name}.npz',t=t,minor=r[:,0],major=r[:,1],area=r[:,2],fps=fps); print(name,len(r),flush=True)
for p in sorted(glob.glob(os.path.expanduser('~/Documents/tyre*.mov'))): run(p)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(4,1,figsize=(16,12))
for a,p in zip(ax,sorted(glob.glob('tyreN_sig_*.npz'))):
    d=np.load(p);t=d['t'];y=d['minor'];ok=np.isfinite(y);y=np.interp(t,t[ok],y[ok]);y=medfilt(y,13);z=y/np.percentile(y,85)
    pk,_=find_peaks(-z,prominence=0.25,distance=240,height=-0.8)
    a.plot(t,z,'k',lw=.8);a.plot(t[pk],z[pk],'rv');a.set_title(f'{p}: {len(pk)} squeezes');a.grid();print(p,len(pk),np.round(t[pk],1))
plt.tight_layout();plt.savefig('$S/f/sigN.png',dpi=60)
