"""Demo: real held-out squeeze vs the fitted model's prediction, side by side, as a 3x slow-motion video and a GIF.
Model = the ring fitted ONLY on clips 1-2 (kb, release speed, rest in ../results/tyre_ring_fit.json). The only thing given to the model for each event
is its own measured squeezed width z0. Panels: real video (top view), simulated tube (the 2D ring extruded over the 37 mm tread; with full-width contact
the 3D stack in tyre_ring3d.py gives the same motion), and the width traces (real black, model red).
Run from ../data:  PYTHONPATH=../code python ../code/make_demo.py   (needs ffmpeg, opencv, matplotlib, numba; videos in ~/Documents)"""
import glob,json,os,subprocess,numpy as np,cv2
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from tyre_fit2 import events
import tyre_ring_fast as RF
KB,VH,REST=json.load(open('../results/tyre_ring_fit.json'))['params']['ring'];W=37.0/75.8;FPS_SIM=60;SLOW=3
def ring_frames(z0,T=1.2,t_close=0.3):
    g0=min(.99,z0/REST);n=RF.N;X=RF.X0.copy();F=np.zeros((n,2));dt=min(0.4/(4*RF.KS+110*KB+100),1e-3);out=[]
    for s in range(int(t_close/dt)):
        g=1.0+(g0-1.0)*min(1.0,(s*dt)/(0.6*t_close));RF._forces(X,F,KB,g,RF.L0);X+=dt*F
    nxt=0.0;s=0
    while s*dt<=T:
        tt=s*dt
        if tt>=nxt-1e-12:out.append((tt,X.copy()));nxt+=1.0/FPS_SIM
        RF._forces(X,F,KB,g0+VH*tt,RF.L0);X+=dt*F;s+=1
    return out
def render(clip,e,fidx,cap,fig,axs,tag):
    z0=e['z0'];fr=ring_frames(z0);ts=np.array([t for t,_ in fr]);wid=np.array([REST*(X[:,1].max()-X[:,1].min()) for _,X in fr])
    t_abs=e['t_abs'];pre=int(0.25*FPS_SIM);frames=[]
    # real-frame centre: tyre blob at release frame
    cap.set(1,int(t_abs*240));_,f0=cap.read();g=cv2.cvtColor(f0,cv2.COLOR_BGR2GRAY);m=(g<0.3*np.percentile(g,80)).astype(np.uint8);cn,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);q=max(cn,key=cv2.contourArea);M=cv2.moments(q);cx,cy=int(M['m10']/M['m00']),int(M['m01']/M['m00'])
    for k in range(-pre,len(fr)):
        t=k/FPS_SIM;cap.set(1,int(round((t_abs+t)*240)));ok,f=cap.read()
        if not ok:continue
        im=cv2.cvtColor(f[max(cy-190,0):cy+190,max(cx-190,0):cx+190],cv2.COLOR_BGR2RGB)
        X=fr[max(k,0)][1];j=max(k,0)
        axs[0].clear();axs[0].imshow(im);axs[0].axis('off');axs[0].set_title('real video (%s, held out)'%tag,fontsize=10)
        ax=axs[1];ax.clear();ax.set_axis_off();th=np.linspace(0,1,2)
        P=np.vstack([X,X[:1]]);zz=np.array([-W/2,W/2])
        Xs=np.outer(np.ones(2),P[:,0]).reshape(2,-1);Ys=np.outer(np.ones(2),P[:,1]).reshape(2,-1);Zs=np.outer(zz,np.ones(len(P)))
        ax.plot_surface(Xs,Zs,Ys,color='#444',edgecolor='#222',linewidth=.3,shade=True,alpha=.95)
        ax.plot(P[:,0],np.full(len(P),-W/2),P[:,1],'k',lw=1.6);ax.plot(P[:,0],np.full(len(P),W/2),P[:,1],'k',lw=1.6)
        ax.set_xlim(-.75,.75);ax.set_ylim(-.75,.75);ax.set_zlim(-.75,.75);ax.set_box_aspect((1,1,1));ax.view_init(elev=48,azim=-72);ax.set_title('model (fitted on clips 1-2)',fontsize=10)
        a=axs[2];a.clear();a.plot(e['t'],e['z'],'k',lw=1.4,label='real');a.plot(ts,wid,'r',lw=1.6,label='model');a.axvline(max(t,0),color='gray',lw=1)
        a.set_ylim(0.35,1.05);a.set_xlim(-0.05,1.2);a.set_xlabel('time after release (s, real)');a.set_ylabel('width / resting width');a.legend(loc='lower right',fontsize=8);a.grid(alpha=.3)
        rm=np.sqrt(np.mean((np.interp(e['t'],ts,wid)-e['z'])**2));a.set_title('held-out error %.3f'%rm,fontsize=10)
        fig.canvas.draw();frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
    return frames
if __name__=='__main__':
    files=sorted(glob.glob('tyreN_sig_*.npz'));ALL=[[e for e in events(f,i+1) if (not e['held']) and e['dur']>0.3] for i,f in enumerate(files)]
    picks=[(3,ALL[2][0],'tyre_3, t=%.1f s'),(4,ALL[3][-1],'tyre_4, t=%.1f s')];names={3:'tyre-3.mov',4:'tyre_4.mov'}
    fig=plt.figure(figsize=(12,4.2),dpi=80);axs=[fig.add_subplot(1,3,1),fig.add_subplot(1,3,2,projection='3d'),fig.add_subplot(1,3,3)];fig.subplots_adjust(left=.03,right=.98,wspace=.15,bottom=.16,top=.9)
    allf=[]
    for c,e,tg in picks:
        cap=cv2.VideoCapture(os.path.expanduser('~/Documents/'+names[c]));allf+=render(c,e,0,cap,fig,axs,tg%e['t_abs']);print(c,len(allf),flush=True)
    os.makedirs('../figures/demo_frames',exist_ok=True)
    for i,fr in enumerate(allf):cv2.imwrite('../figures/demo_frames/f%04d.png'%i,cv2.cvtColor(fr,cv2.COLOR_RGB2BGR))
    fps=FPS_SIM/SLOW
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i','../figures/demo_frames/f%04d.png','-c:v','libx264','-pix_fmt','yuv420p','-crf','20','../figures/demo.mp4'],check=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i','../figures/demo_frames/f%04d.png','-vf','scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4','../figures/demo.gif'],check=True)
    import shutil;shutil.rmtree('../figures/demo_frames');print('done',len(allf),'frames')
