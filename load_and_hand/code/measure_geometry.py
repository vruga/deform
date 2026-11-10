"""Geometry and frame-rate checks for the side-view clips (the exact analyses run for README section A2). Videos are read from ~/Documents."""
import cv2,numpy as np,os
A=cv2.aruco;det=A.ArucoDetector(A.getPredefinedDictionary(A.DICT_4X4_50),A.DetectorParameters())
V=lambda n:cv2.VideoCapture(os.path.expanduser(f'~/Documents/{n}.mov'))
def flicker(n):                      # true capture rate from the 100 Hz mains flicker
    c=V(n);m=[]
    while True:
        ok,f=c.read()
        if not ok:break
        m.append(cv2.resize(f,(96,54)).mean())
    m=np.array(m);m=m-np.convolve(m,np.ones(31)/31,'same');F=np.abs(np.fft.rfft(m[50:-50]));fr=np.fft.rfftfreq(len(m)-100);k=np.argmax(F[5:])+5
    print(n,'flicker %.4f cyc/frame -> %.0f fps (peak/median %.0f)'%(fr[k],100/fr[k],F[k]/np.median(F[5:])))
def tyre6_scale():                   # marker scale vs tyre width / tread height, rest frames of tyre_6
    c=V('tyre_6');out=[]
    for i in range(0,330,10):
        c.set(1,i);ok,f=c.read()
        if not ok:break
        v=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)[...,2].astype(int);m=(v<0.3*np.percentile(v,80)).astype(np.uint8);m[:int(.25*f.shape[0])]=0;m[int(.9*f.shape[0]):]=0
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8));n,l,st,_=cv2.connectedComponentsWithStats(m)
        if n<2:continue
        j=1+np.argmax(st[1:,4]);x,y,w,h,a=st[j];c4,ids,_=det.detectMarkers(f)
        if ids is not None and len(ids):q=c4[0].reshape(4,2);out.append((w,h,np.mean([np.linalg.norm(q[k]-q[(k+1)%4]) for k in range(4)])))
    o=np.array(out);print('tyre_6: width %.0f px, marker %.1f px/10 mm -> tyre width %.1f mm'%(np.median(o[:,0]),np.median(o[:,2]),10*np.median(o[:,0])/np.median(o[:,2])))
def wall_thickness():                # brightness profile through the standing tyre (tyre_deformation), unloaded frames
    c=V('tyre_deformation');R=[]
    for i in range(20,200,8):
        c.set(1,i);ok,f=c.read()
        if not ok:break
        g=cv2.GaussianBlur(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(float),(5,5),0);best=None
        for y in range(430,700,4):
            row=g[y,300:620];thr=0.55*np.percentile(g[420:700,300:620],90);dark=row<thr;runs=[];s=None
            for k,d in enumerate(dark):
                if d and s is None:s=k
                if (not d) and s is not None:
                    if k-s>=6:runs.append((s,k-1))
                    s=None
            if s is not None and len(dark)-s>=6:runs.append((s,len(dark)-1))
            if len(runs)==2 and runs[0][1]<runs[1][0]-30:
                o=runs[1][1]-runs[0][0]
                if best is None or o>best[0]:best=(o,runs[1][0]-runs[0][1],runs[0][1]-runs[0][0],runs[1][1]-runs[1][0])
        if best:R.append(best)
    o,h,t1,t2=[np.median(np.array(R)[:,k]) for k in range(4)];print('outer %.0f hole %.0f walls %.0f/%.0f px -> wall = %.1f mm at 75.8 mm outer'%(o,h,t1,t2,75.8*((o-h)/2)/o))
if __name__=='__main__':
    for n in ('tyre_deformation','tyre_5','tyre_6','tyre_7'):flicker(n)
    tyre6_scale();wall_thickness()
