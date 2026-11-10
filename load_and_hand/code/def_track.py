import cv2,numpy as np,os,sys
A=cv2.aruco;det=A.ArucoDetector(A.getPredefinedDictionary(A.DICT_4X4_50),A.DetectorParameters())
c=cv2.VideoCapture(os.path.expanduser(os.environ.get('TYRE_VIDEOS','~/Documents')+'/tyre_deformation.mov'));rows=[]
while True:
    ok,f=c.read()
    if not ok:break
    s=cv2.resize(f,(360,640));hsv=cv2.cvtColor(s,cv2.COLOR_BGR2HSV);H,S,V=[hsv[...,i].astype(int) for i in range(3)]
    corners,ids,_=det.detectMarkers(s);my=mx=ms=np.nan
    if ids is not None and len(ids):
        q=corners[0].reshape(4,2);my,mx=q[:,1].mean(),q[:,0].mean();ms=np.mean([np.linalg.norm(q[i]-q[(i+1)%4]) for i in range(4)])
    blue=((H>=98)&(H<=130)&(S>70)&(V>25)&(V<200)).astype(np.uint8);blue[:150]=0
    nb=blue.sum();bb=np.nan
    if nb>1500:
        ys,xs=np.nonzero(blue);bb=np.percentile(ys,99.5)
    dark=(V<0.32*np.percentile(V[250:],80)).astype(np.uint8);dark[:200]=0;dark[430:]=0;dark[:,:100]=0    # floor band, away from tray
    dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));n,l,st,_=cv2.connectedComponentsWithStats(dark)
    if n>1:
        j=1+np.argmax(st[1:,4]);x,y,w,h,a=st[j];tb=y+h;tt=y;tw=w;tx=x+w/2
    else:tb=tt=tw=tx=np.nan
    rows.append([nb,bb,tt,tb,tw,tx,my,mx,ms])
r=np.array(rows);np.save('def_track.npy',r);print(r.shape)
