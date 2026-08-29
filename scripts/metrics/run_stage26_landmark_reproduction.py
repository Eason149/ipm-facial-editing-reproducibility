from pathlib import Path
import sys,json,hashlib,os
REPO_ROOT=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('IPM_DATA_ROOT',REPO_ROOT/'data')).resolve();STIM=Path(os.environ.get('IPM_STIMULUS_ROOT',ROOT/'stimuli')).resolve();BASE=ROOT/'ipm_visual_information_gate';OUT=ROOT/'ipm_stage_2_6';LMOUT=OUT/'landmarks_468';LMOUT.mkdir(parents=True,exist_ok=True)
import cv2,mediapipe as mp,numpy as np,pandas as pd
from scipy import stats
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent));from face_aoi_mediapipe import compute_aois
CONTOUR=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
def boxvec(a):return np.array([a['x1'],a['y1'],a['x2'],a['y2']],float)
def ptsarr(a):return np.array([[p['x'],p['y']] for p in a],float)
def rms(x):x=np.asarray(x,float);return x/np.sqrt(np.mean(x*x))
def main():
 m=pd.read_csv(BASE/'image_condition_mapping.csv');m=m[m.identity.isin(['F_1','F_2','M_1','M_2'])&((m.formal_factorial_condition==True)|(m.original_flag==1))].drop_duplicates(['identity','picture_filename'])
 mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.5);det=mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=.5)
 geo={};status=[]
 for _,r in m.iterrows():
  fn=r.picture_filename;im=cv2.imread(str(STIM/fn));rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);dr=det.process(rgb);mr=mesh.process(rgb);conf=float(dr.detections[0].score[0]) if dr.detections else np.nan
  if not mr.multi_face_landmarks:
   status.append(dict(image=fn,success=False,detection_confidence=conf,error='FaceMesh no face'));continue
  xyz=np.array([[p.x,p.y,p.z] for p in mr.multi_face_landmarks[0].landmark],float);pix={i:(int(p[0]*im.shape[1]),int(p[1]*im.shape[0])) for i,p in enumerate(xyz)};a=compute_aois(pix,im.shape[1],im.shape[0]);geo[fn]=(xyz,a)
  pd.DataFrame({'landmark_index':np.arange(468),'x_normalized':xyz[:,0],'y_normalized':xyz[:,1],'z_normalized':xyz[:,2],'x_px':xyz[:,0]*im.shape[1],'y_px':xyz[:,1]*im.shape[0]}).to_csv(LMOUT/(Path(fn).stem+'.csv'),index=False,encoding='utf-8-sig')
  status.append(dict(image=fn,success=True,detection_confidence=conf,error=''))
 pd.DataFrame(status).to_csv(LMOUT/'detection_status.csv',index=False,encoding='utf-8-sig')
 if len(geo)!=68:raise RuntimeError(f'Only {len(geo)}/68 landmark detections succeeded')
 rows=[]
 for ident,g in m.groupby('identity'):
  b=g[g.original_flag==1].iloc[0];_,ba=geo[b.picture_filename];be=ba['Eye'];bm=ba['Mouth'];bn=ba['Nose'];bc=ptsarr(ba['Skin']['face_contour_points']);iod=max(be['x2']-be['x1'],1)
  for _,r in g[g.formal_factorial_condition==True].iterrows():
   _,a=geo[r.picture_filename];e=a['Eye'];mo=a['Mouth'];no=a['Nose'];c=ptsarr(a['Skin']['face_contour_points'])
   rows.append(dict(identity=ident,picture_filename=r.picture_filename,cond_id=int(r.cond_id),fslim=int(r.fslim_binary),eye=int(r.eye_binary),mouth=int(r.mouth_binary),skin=int(r.skin_binary),g_total=np.mean([np.linalg.norm(boxvec(e)-boxvec(be)),np.linalg.norm(boxvec(mo)-boxvec(bm)),np.linalg.norm(boxvec(no)-boxvec(bn)),np.mean(np.linalg.norm(c-bc,axis=1))])/iod,g_eye=np.linalg.norm(boxvec(e)-boxvec(be))/iod,g_mouth=np.linalg.norm(boxvec(mo)-boxvec(bm))/iod,g_contour=np.mean(np.linalg.norm(c-bc,axis=1))/iod,eye_ratio_change=abs((e['x2']-e['x1'])/max(e['y2']-e['y1'],1)-(be['x2']-be['x1'])/max(be['y2']-be['y1'],1)),mouth_ratio_change=abs((mo['x2']-mo['x1'])/max(mo['y2']-mo['y1'],1)-(bm['x2']-bm['x1'])/max(bm['y2']-bm['y1'],1)),face_ratio_change=abs(np.ptp(c[:,0])/max(np.ptp(c[:,1]),1)-np.ptp(bc[:,0])/max(np.ptp(bc[:,1]),1))))
 d=pd.DataFrame(rows);comps=['g_total','g_eye','g_mouth','g_contour','eye_ratio_change','mouth_ratio_change','face_ratio_change'];d['G_new_raw']=np.mean(np.column_stack([rms(d[c]) for c in comps]),axis=1);d['G_new']=(d.G_new_raw-d.G_new_raw.mean())/d.G_new_raw.std(ddof=0)
 old=pd.read_csv(BASE/'image_metrics.csv')[['identity','picture_filename','G_raw','G']];z=d.merge(old,on=['identity','picture_filename'],validate='one_to_one');z['difference_z']=z.G_new-z.G;z['mean_z']=(z.G_new+z.G)/2
 overall=pd.DataFrame([dict(group_type='overall',group='all',n=len(z),pearson=stats.pearsonr(z.G,z.G_new).statistic,spearman=stats.spearmanr(z.G,z.G_new).statistic,mean_absolute_difference=np.mean(abs(z.difference_z)),mean_difference=np.mean(z.difference_z),sd_difference=np.std(z.difference_z,ddof=1),loa_low=np.mean(z.difference_z)-1.96*np.std(z.difference_z,ddof=1),loa_high=np.mean(z.difference_z)+1.96*np.std(z.difference_z,ddof=1))])
 more=[]
 for typ,col in [('identity','identity'),('operation','fslim'),('operation','eye'),('operation','mouth'),('operation','skin')]:
  for val,q in z.groupby(col):more.append(dict(group_type=typ,group=f'{col}={val}',n=len(q),pearson=stats.pearsonr(q.G,q.G_new).statistic,spearman=stats.spearmanr(q.G,q.G_new).statistic,mean_absolute_difference=np.mean(abs(q.difference_z)),mean_difference=np.mean(q.difference_z),sd_difference=np.std(q.difference_z,ddof=1),loa_low=np.mean(q.difference_z)-1.96*np.std(q.difference_z,ddof=1),loa_high=np.mean(q.difference_z)+1.96*np.std(q.difference_z,ddof=1)))
 agree=pd.concat([overall,pd.DataFrame(more)],ignore_index=True);agree.to_csv(OUT/'geometric_metric_agreement.csv',index=False,encoding='utf-8-sig');z.to_csv(OUT/'geometric_metric_reproduced_values.csv',index=False,encoding='utf-8-sig')
 fig,ax=plt.subplots(figsize=(5,4));ax.scatter(z.mean_z,z.difference_z,c='#2563eb');ax.axhline(z.difference_z.mean(),c='black');ax.axhline(z.difference_z.mean()+1.96*z.difference_z.std(ddof=1),c='red',ls='--');ax.axhline(z.difference_z.mean()-1.96*z.difference_z.std(ddof=1),c='red',ls='--');ax.set(xlabel='Mean of old and reproduced G (z)',ylabel='Reproduced − old G (z)',title='Bland–Altman agreement');fig.tight_layout();fig.savefig(OUT/'geometric_metric_bland_altman.png',dpi=300);fig.savefig(OUT/'geometric_metric_bland_altman.pdf');plt.close(fig)
 o=overall.iloc[0];report=f'''# Geometric Metric Reproduction\n\n## Environment and frozen formula\n\nA clean environment was locked to Python 3.9, MediaPipe 0.10.14, NumPy 1.26.4 and OpenCV 4.11.0. All 68 core images were rerun. The seven pre-existing G components, RMS scaling and equal-weight mean were retained without reference to behavioral or EEG outcomes.\n\n## Detection\n\n- Successful full 468-point detections: {sum(x['success'] for x in status)}/68.\n- Failures: {sum(not x['success'] for x in status)}.\n- Face-detection confidence range: {np.nanmin([x['detection_confidence'] for x in status]):.4f}–{np.nanmax([x['detection_confidence'] for x in status]):.4f}.\n- Raw per-image coordinates and detection status are in `landmarks_468/`.\n\n## Agreement with Stage 2 G\n\n- Pearson r={o.pearson:.4f}.\n- Spearman rho={o.spearman:.4f}.\n- Mean absolute standardized difference={o.mean_absolute_difference:.4f}.\n- Mean difference={o.mean_difference:.4f}; 95% limits of agreement [{o.loa_low:.4f}, {o.loa_high:.4f}].\n\nIdentity- and operation-specific agreement is reported in `geometric_metric_agreement.csv`. This reproduction is algorithmically independent of the behavioral and EEG outcomes but uses the frozen Stage 2 formula.\n''';(OUT/'geometric_metric_reproduction.md').write_text(report,encoding='utf-8')
 (OUT/'landmark_environment.json').write_text(json.dumps({'python':sys.version,'mediapipe':mp.__version__,'opencv':cv2.__version__,'numpy':np.__version__,'formula_components':comps,'weights':'equal after RMS scaling','n_images':68},indent=2),encoding='utf-8')
if __name__=='__main__':main()
