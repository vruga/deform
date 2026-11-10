import cv2,numpy as np,os,glob
for p in sorted(glob.glob(os.path.expanduser('~/Documents/tyre*.mov'))):
    name=os.path.basename(p)[:-4].replace('-','_');c=cv2.VideoCapture(p);rows=[]
    while True:
        ok,f=c.read()
        if not ok:break
        s=cv2.resize(f,(240,135));h=cv2.cvtColor(s,cv2.COLOR_BGR2HSV);H,S,V=[h[...,i].astype(int) for i in range(3)]
        sat_ref=np.percentile(S,50)
        m=((H<22)|(H>170))&(S>35)&(S<190)&(V>70)&(V>0.45*np.percentile(V,80))
        rows.append(m.mean())
    np.save(f'hand_{name}.npy',np.array(rows));print(name,flush=True)
