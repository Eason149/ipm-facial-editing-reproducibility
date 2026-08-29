#!/usr/bin/env python
"""Compute preregistered geometric, appearance and identity stimulus metrics."""
from pathlib import Path
import csv, hashlib, json, math, os
import cv2, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get("IPM_DATA_ROOT",REPO_ROOT/"data")).resolve()
OUT=ROOT/"ipm_visual_information_gate"
STIM=Path(os.environ.get("IPM_STIMULUS_ROOT",ROOT/"stimuli")).resolve()
MAP=OUT/"image_condition_mapping.csv"
MODEL=OUT/"models"
LOG=OUT/"logs"/"visual_metrics.log"
FIG=OUT/"figures"; TAB=OUT/"tables"
FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)

EYE=set([33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173,263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398])
MOUTH=set([61,146,91,181,84,17,314,405,321,375,291,185,40,39,37,0,267,269,270,409,78,95,88,178,87,14,317,402,318,324,308])
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

def log(s):
    with LOG.open("a",encoding="utf-8") as f:f.write(s+"\n")
    print(s,flush=True)
def rms_scale(x):
    x=np.asarray(x,float); d=np.sqrt(np.mean(x*x)); return x/d if d>0 else x
def ssim_gray(a,b,mask):
    x=a[mask].astype(float); y=b[mask].astype(float)
    c1=(.01*255)**2;c2=(.03*255)**2
    return ((2*x.mean()*y.mean()+c1)*(2*np.cov(x,y,ddof=0)[0,1]+c2))/((x.mean()**2+y.mean()**2+c1)*(x.var()+y.var()+c2))
def lbp_hist(gray,mask):
    c=gray[1:-1,1:-1]; code=np.zeros_like(c,np.uint8)
    neigh=[gray[:-2,:-2],gray[:-2,1:-1],gray[:-2,2:],gray[1:-1,2:],gray[2:,2:],gray[2:,1:-1],gray[2:,:-2],gray[1:-1,:-2]]
    for k,n in enumerate(neigh): code|=((n>=c).astype(np.uint8)<<k)
    h=np.bincount(code[mask[1:-1,1:-1]].ravel(),minlength=256).astype(float); return h/max(h.sum(),1)
def chi2(a,b): return .5*np.sum((a-b)**2/(a+b+1e-12))
def polygon_mask(shape,pts):
    m=np.zeros(shape[:2],np.uint8);cv2.fillPoly(m,[np.asarray(pts,np.int32)],255);return m.astype(bool)
def boxvec(a): return np.array([a['x1'],a['y1'],a['x2'],a['y2']],float)
def points(a): return np.array([[p['x'],p['y']] for p in a],float)

def main():
    if LOG.exists():LOG.unlink()
    log("Stage 3 started: G/A/I metrics.")
    m=pd.read_csv(MAP)
    m=m[(m.identity.isin(['F_1','F_2','M_1','M_2'])) & ((m.formal_factorial_condition==True)|(m.original_flag==1))].copy()
    m=m.drop_duplicates(['identity','picture_filename'])
    aoi_rows=json.loads((STIM/'aoi_output'/'aoi_all_images.json').read_text(encoding='utf-8'))
    aoi={r['image']:r for r in aoi_rows}
    missing=sorted(set(m.picture_filename)-set(aoi))
    if missing: raise RuntimeError(f"Saved MediaPipe AOI coordinates missing: {missing}")
    det=cv2.FaceDetectorYN.create(str(MODEL/'face_detection_yunet_2023mar.onnx'),"",(320,320),.7,.3,5000)
    rec=cv2.FaceRecognizerSF.create(str(MODEL/'face_recognition_sface_2021dec.onnx'),"")
    cache={}
    for _,r in m.iterrows():
        fn=r.picture_filename;p=STIM/fn; im=cv2.imread(str(p)); rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        geo=aoi[fn]
        det.setInputSize((im.shape[1],im.shape[0])); _,faces=det.detect(im)
        if faces is None: raise RuntimeError(f"YuNet no face: {fn}")
        f=faces[np.argmax(faces[:,2]*faces[:,3])]
        aligned=rec.alignCrop(im,f); emb=rec.feature(aligned).reshape(-1).astype(float);emb/=np.linalg.norm(emb)
        cache[fn]=(im,geo,emb)
    log(f"All {len(cache)} experiment-referenced face images have saved MediaPipe AOIs and passed recognition-model detection.")
    rows=[]
    for ident,g in m.groupby('identity'):
        base=g[g.original_flag==1].iloc[0]; bi,bgeo,be=cache[base.picture_filename]
        beo=bgeo['aois']['Eye']; bmo=bgeo['aois']['Mouth']; bno=bgeo['aois']['Nose']; bc=points(bgeo['skin']['face_contour_points'])
        iod=max(beo['x2']-beo['x1'],1)
        bgray=cv2.cvtColor(bi,cv2.COLOR_BGR2GRAY); blab=cv2.cvtColor(bi,cv2.COLOR_BGR2LAB).astype(float)
        bmask=polygon_mask(bi.shape,bc); beye=np.zeros(bi.shape[:2],bool); beye[beo['y1']:beo['y2']+1,beo['x1']:beo['x2']+1]=1; bmouth=np.zeros_like(beye);bmouth[bmo['y1']:bmo['y2']+1,bmo['x1']:bmo['x2']+1]=1;bskin=bmask&~beye&~bmouth
        bh=lbp_hist(bgray,bskin)
        for _,r in g[g.formal_factorial_condition==True].iterrows():
            im,geo,emb=cache[r.picture_filename]; gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float)
            eo=geo['aois']['Eye'];mo=geo['aois']['Mouth'];no=geo['aois']['Nose'];c=points(geo['skin']['face_contour_points'])
            mask=polygon_mask(im.shape,c); eye=np.zeros(im.shape[:2],bool);eye[eo['y1']:eo['y2']+1,eo['x1']:eo['x2']+1]=1;mouth=np.zeros_like(eye);mouth[mo['y1']:mo['y2']+1,mo['x1']:mo['x2']+1]=1;skin=mask&~eye&~mouth&bskin
            contour_disp=np.linalg.norm(c-bc,axis=1)/iod
            edge0=cv2.Canny(bgray,80,160);edge=cv2.Canny(gray,80,160)
            row=dict(identity=ident,picture_filename=r.picture_filename,cond_id=int(r.cond_id),fslim=int(r.fslim_binary),eye=int(r.eye_binary),mouth=int(r.mouth_binary),skin=int(r.skin_binary),
              g_total=np.mean([np.linalg.norm(boxvec(eo)-boxvec(beo)),np.linalg.norm(boxvec(mo)-boxvec(bmo)),np.linalg.norm(boxvec(no)-boxvec(bno)),np.mean(np.linalg.norm(c-bc,axis=1))])/iod,
              g_eye=np.linalg.norm(boxvec(eo)-boxvec(beo))/iod,g_mouth=np.linalg.norm(boxvec(mo)-boxvec(bmo))/iod,g_contour=contour_disp.mean(),
              eye_ratio_change=abs(((eo['x2']-eo['x1'])/max(eo['y2']-eo['y1'],1))-((beo['x2']-beo['x1'])/max(beo['y2']-beo['y1'],1))),
              mouth_ratio_change=abs(((mo['x2']-mo['x1'])/max(mo['y2']-mo['y1'],1))-((bmo['x2']-bmo['x1'])/max(bmo['y2']-bmo['y1'],1))),
              face_ratio_change=abs(((c[:,0].ptp())/max(c[:,1].ptp(),1))-((bc[:,0].ptp())/max(bc[:,1].ptp(),1))),
              a_deltaE=np.linalg.norm(lab[skin].mean(0)-blab[skin].mean(0)),a_luminance=abs(lab[skin,0].mean()-blab[skin,0].mean()),
              a_contrast=abs(gray[skin].std()-bgray[skin].std()),a_texture=chi2(lbp_hist(gray,skin),bh),
              a_edge=abs((edge[skin]>0).mean()-(edge0[skin]>0).mean()),a_one_minus_ssim=1-ssim_gray(bgray,gray,skin),
              identity_cosine=float(np.dot(be,emb)),identity_distance=float(1-np.dot(be,emb)))
            rows.append(row)
    d=pd.DataFrame(rows)
    gc=['g_total','g_eye','g_mouth','g_contour','eye_ratio_change','mouth_ratio_change','face_ratio_change']
    ac=['a_deltaE','a_luminance','a_contrast','a_texture','a_edge','a_one_minus_ssim']
    d['G_raw']=np.mean(np.column_stack([rms_scale(d[c]) for c in gc]),axis=1)
    d['A_raw']=np.mean(np.column_stack([rms_scale(d[c]) for c in ac]),axis=1)
    d['I_raw']=d.identity_distance
    for c in ['G_raw','A_raw','I_raw']:d[c[0]]= (d[c]-d[c].mean())/d[c].std(ddof=0)
    d.to_csv(OUT/'image_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'file':str(MODEL/x),'sha256':hashlib.sha256((MODEL/x).read_bytes()).hexdigest()} for x in ['face_detection_yunet_2023mar.onnx','face_recognition_sface_2021dec.onnx']]).to_csv(TAB/'model_hashes.csv',index=False)
    log("Stage 4 started: construct-validity tests.")
    tests=[]
    X=np.column_stack([np.ones(len(d)),d[['fslim','eye','mouth','skin']].to_numpy(float),pd.get_dummies(d.identity,drop_first=True).to_numpy(float)])
    names=['Intercept','fslim','eye','mouth','skin']+[f'identity_{x}' for x in sorted(d.identity.unique())[1:]]
    for y in ['G','A','I']:
        yy=d[y].to_numpy(float); inv=np.linalg.inv(X.T@X); beta=inv@X.T@yy; resid=yy-X@beta; h=np.sum((X@inv)*X,axis=1)
        meat=X.T@np.diag((resid/(1-h))**2)@X; cov=inv@meat@inv; se=np.sqrt(np.diag(cov)); dof=len(d)-X.shape[1]
        r2=1-np.sum(resid**2)/np.sum((yy-yy.mean())**2)
        for term in ['fslim','eye','mouth','skin']:
            j=names.index(term);tv=beta[j]/se[j]
            tests.append(dict(outcome=y,term=term,beta=beta[j],se=se[j],t=tv,p=2*stats.t.sf(abs(tv),dof),r2=r2,n=len(d)))
    t=pd.DataFrame(tests);t['p_holm_within_outcome']=t.groupby('outcome').p.transform(lambda x: np.maximum.accumulate(np.sort(x)*np.arange(len(x),0,-1))[np.argsort(np.argsort(x))])
    t.to_csv(TAB/'construct_validity_factor_tests.csv',index=False,encoding='utf-8-sig')
    d.groupby(['fslim','eye','mouth','skin'])[['G','A','I']].mean().reset_index().to_csv(TAB/'construct_validity_cell_means.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(1,3,figsize=(10,3.2))
    for a,y,title in zip(ax,['G','A','I'],['Geometry G','Appearance A','Identity alteration I']):
        means=[d.groupby(q)[y].mean().iloc[1]-d.groupby(q)[y].mean().iloc[0] for q in ['fslim','eye','mouth','skin']]
        a.bar(['FSlim','Eye','Mouth','Skin'],means,color=['#3b82f6','#6366f1','#8b5cf6','#f59e0b']);a.axhline(0,color='black',lw=.7);a.set_title(title);a.set_ylabel('standardized mean difference')
    fig.tight_layout();fig.savefig(FIG/'construct_validity.png',dpi=300);fig.savefig(FIG/'construct_validity.pdf');fig.savefig(FIG/'construct_validity.svg');plt.close(fig)
    q=d[['G','A','I']].corr();q.to_csv(TAB/'metric_correlations.csv',encoding='utf-8-sig')
    summary={'n_images':len(d),'n_identities':d.identity.nunique(),'landmark_model':'MediaPipe Face Mesh 468','identity_model':'OpenCV SFace 2021dec with YuNet 2023mar','metric_correlations':q.to_dict(),'identity_cosine_range':[float(d.identity_cosine.min()),float(d.identity_cosine.max())]}
    (OUT/'visual_metrics_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    log("Stages 3-4 complete.")

if __name__=='__main__':main()
