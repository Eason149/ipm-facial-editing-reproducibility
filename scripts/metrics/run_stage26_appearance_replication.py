from pathlib import Path
import sys,os
REPO_ROOT=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('IPM_DATA_ROOT',REPO_ROOT/'data')).resolve();STIM=Path(os.environ.get('IPM_STIMULUS_ROOT',ROOT/'stimuli')).resolve();BASE=ROOT/'ipm_visual_information_gate';OUT=ROOT/'ipm_stage_2_6';LM=OUT/'landmarks_468'
import cv2,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent));from face_aoi_mediapipe import compute_aois
def rms(x):x=np.asarray(x,float);return x/np.sqrt(np.mean(x*x))
def js(p,q):
 p=p/p.sum();q=q/q.sum();m=(p+q)/2
 return .5*np.sum(np.where(p>0,p*np.log((p+1e-12)/(m+1e-12)),0))+.5*np.sum(np.where(q>0,q*np.log((q+1e-12)/(m+1e-12)),0))
def lm(fn):
 d=pd.read_csv(LM/(Path(fn).stem+'.csv'));return d[['x_px','y_px']].to_numpy()
def mask_from_cheeks(shape,a):
 m=np.zeros(shape[:2],np.uint8)
 for key in ['Cheek_L','Cheek_R']:cv2.fillPoly(m,[np.array([[p['x'],p['y']] for p in a[key]['points']],np.int32)],255)
 return m.astype(bool)
def main():
 m=pd.read_csv(BASE/'image_condition_mapping.csv');m=m[m.identity.isin(['F_1','F_2','M_1','M_2'])&((m.formal_factorial_condition==True)|(m.original_flag==1))].drop_duplicates(['identity','picture_filename']);rows=[]
 for ident,g in m.groupby('identity'):
  br=g[g.original_flag==1].iloc[0];bim=cv2.imread(str(STIM/br.picture_filename));bl=lm(br.picture_filename);bp={i:tuple(map(int,p)) for i,p in enumerate(bl)};ba=compute_aois(bp,bim.shape[1],bim.shape[0]);mask=mask_from_cheeks(bim.shape,ba);blab=cv2.cvtColor(bim,cv2.COLOR_BGR2LAB);bgray=cv2.cvtColor(bim,cv2.COLOR_BGR2GRAY)
  hist0=cv2.calcHist([blab],[1,2],mask.astype(np.uint8),[16,16],[0,256,0,256]).ravel().astype(float)
  gabor=[cv2.getGaborKernel((21,21),4,th,8,.5,0,ktype=cv2.CV_32F) for th in [0,np.pi/4,np.pi/2,3*np.pi/4]];tex0=np.array([np.mean(np.abs(cv2.filter2D(bgray,cv2.CV_32F,k)[mask])) for k in gabor])
  stable=[33,263,1,61,291]
  for _,r in g[g.formal_factorial_condition==True].iterrows():
   im=cv2.imread(str(STIM/r.picture_filename));moved=lm(r.picture_filename);T,_=cv2.estimateAffinePartial2D(moved[stable].astype(np.float32),bl[stable].astype(np.float32),method=cv2.LMEDS);aligned=cv2.warpAffine(im,T,(bim.shape[1],bim.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
   lab=cv2.cvtColor(aligned,cv2.COLOR_BGR2LAB);gray=cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY);hist=cv2.calcHist([lab],[1,2],mask.astype(np.uint8),[16,16],[0,256,0,256]).ravel().astype(float);tex=np.array([np.mean(np.abs(cv2.filter2D(gray,cv2.CV_32F,k)[mask])) for k in gabor])
   rows.append(dict(identity=ident,picture_filename=r.picture_filename,cond_id=int(r.cond_id),fslim=int(r.fslim_binary),eye=int(r.eye_binary),mouth=int(r.mouth_binary),skin=int(r.skin_binary),a2_chroma_js=js(hist0,hist),a2_luminance=abs(lab[:,:,0][mask].mean()-blab[:,:,0][mask].mean()),a2_texture=np.mean(abs(tex-tex0)/(abs(tex0)+1e-9)),alignment_rmse=np.sqrt(np.mean(np.sum((cv2.transform(moved[stable][None].astype(np.float32),T)[0]-bl[stable])**2,axis=1)))))
 d=pd.DataFrame(rows);d['A2_raw']=np.mean(np.column_stack([rms(d.a2_chroma_js),rms(d.a2_luminance),rms(d.a2_texture)]),axis=1);d['A2']=(d.A2_raw-d.A2_raw.mean())/d.A2_raw.std(ddof=0)
 old=pd.read_csv(BASE/'image_metrics.csv')[['identity','picture_filename','A','G','I']];d=d.merge(old,on=['identity','picture_filename'],validate='one_to_one');d.to_csv(OUT/'appearance_metric_sensitivity.csv',index=False,encoding='utf-8-sig')
 import statsmodels.formula.api as smf
 fit=smf.ols('A2 ~ fslim + eye + mouth + skin + alignment_rmse + C(identity)',d).fit(cov_type='HC3');tests=[]
 for term in ['fslim','eye','mouth','skin','alignment_rmse']:tests.append(dict(test='construct',group='all',term=term,estimate=fit.params[term],se=fit.bse[term],p=fit.pvalues[term]))
 for held in sorted(d.identity.unique()):
  q=d[d.identity!=held];f=smf.ols('A2 ~ fslim + eye + mouth + skin + alignment_rmse + C(identity)',q).fit(cov_type='HC3')
  for term in ['fslim','skin','alignment_rmse']:tests.append(dict(test='leave_one_identity_out',group=f'exclude_{held}',term=term,estimate=f.params[term],se=f.bse[term],p=f.pvalues[term]))
 pd.DataFrame(tests).to_csv(OUT/'appearance_metric_agreement.csv',index=False,encoding='utf-8-sig')
 corr=d[['A','A2','G','I','alignment_rmse']].corr(method='pearson');spear=d[['A','A2','G','I','alignment_rmse']].corr(method='spearman');corr.to_csv(OUT/'appearance_metric_correlations_pearson.csv');spear.to_csv(OUT/'appearance_metric_correlations_spearman.csv')
 report=f'''# Appearance Metric Replication\n\n## Frozen independent A2\n\nBefore examining A2 outcomes, A2 was fixed as the equal-weight RMS-scaled composite of three quantities measured only in the two cheek AOIs after five-landmark similarity alignment: (1) Lab a/b histogram Jensen–Shannon divergence, (2) absolute L* change, and (3) four-orientation Gabor texture-response change. It does not reuse the Stage 2 A skin mask or its SSIM/LBP/edge components. No alternative A2 variants were screened.\n\n## Agreement and construct checks\n\n- A–A2 Pearson r={corr.loc['A','A2']:.4f}; Spearman rho={spear.loc['A','A2']:.4f}.\n- Skin effect on A2: beta={fit.params['skin']:.4f}, p={fit.pvalues['skin']:.4g}.\n- FSlim cross-loading on A2: beta={fit.params['fslim']:.4f}, p={fit.pvalues['fslim']:.4g}.\n- Residual geometric-alignment association: beta={fit.params['alignment_rmse']:.4f}, p={fit.pvalues['alignment_rmse']:.4g}.\n- A2 correlations with G and I: r={corr.loc['A2','G']:.4f} and r={corr.loc['A2','I']:.4f}.\n\nLeave-one-identity-out coefficients and p values are in `appearance_metric_agreement.csv`. Because only four identities exist, stability is descriptive and does not establish new-stimulus generalization.\n''';(OUT/'appearance_metric_replication.md').write_text(report,encoding='utf-8')
if __name__=='__main__':main()
