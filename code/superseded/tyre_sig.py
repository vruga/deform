"""Per-frame tyre shape signal: filled dark blob -> ellipse/rect minor axis, major axis, area. Saves tyre_sig_<name>.npz"""
import cv2, numpy as np, sys, glob, os
FPS = {'tyre1_wood_240fps':240.,'tyre2_white_240fps':240.,'tyre3_white_60fps':60.,'tyre4_white_240fps':240.}
def run(path):
    name=os.path.basename(path)[:-4]; fps=FPS[name]; c=cv2.VideoCapture(path); rows=[]
    while True:
        ok,f=c.read()
        if not ok: break
        s=cv2.resize(f,(480,270)) if f.shape[1]>f.shape[0] else cv2.resize(f,(270,480))
        v=cv2.cvtColor(s,cv2.COLOR_BGR2HSV)[...,2]
        m=(v<0.3*np.percentile(v,80)).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt=[q for q in cnt if cv2.contourArea(q)>400]
        if not cnt: rows.append([np.nan]*5); continue
        q=max(cnt,key=cv2.contourArea); (cx,cy),(w,h),_=cv2.minAreaRect(q)
        rows.append([min(w,h),max(w,h),cv2.contourArea(q),cx,cy])
    r=np.array(rows); np.savez(f'tyre_sig_{name}.npz',t=np.arange(len(r))/fps,minor=r[:,0],major=r[:,1],area=r[:,2],cx=r[:,3],cy=r[:,4],fps=fps)
    print(name,len(r),'frames',round(len(r)/fps,1),'s',flush=True)
if __name__=='__main__':
    for p in sorted(glob.glob('tyre_videos/*.mov')): run(p)
