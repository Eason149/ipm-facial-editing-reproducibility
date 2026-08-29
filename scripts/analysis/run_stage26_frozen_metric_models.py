from pathlib import Path
import os,sys
REPO_ROOT=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('IPM_DATA_ROOT',REPO_ROOT/'data')).resolve();BASE=ROOT/'ipm_visual_information_gate';OUT=ROOT/'ipm_stage_2_6'
sys.path.insert(0,str(Path(__file__).resolve().parent));pkg=ROOT/'RSA_time_resolved_analysis'/'.python-packages'
if pkg.exists():sys.path.append(str(pkg))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.codex_matplotlib_cache'))
import numpy as np,pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from run_time_resolved_factor_decoding import Context,setup_dirs,parse_eprime_condition_map,find_eeg_files,load_subject,assign_analysis_channels,extract_subject_matrix
from run_erp_dynamics_persistence import smooth_timecourse,nearest_time_indices,time_grid
SEED=20260818;NP=10000;IDS=['F_1','F_2','M_1','M_2']
def clusters(t,crit):
 mask=np.abs(t)>=crit;out=[];s=None
 for i,v in enumerate(mask):
  if v and s is None:s=i
  if s is not None and (not v or i==len(mask)-1):e=i if v and i==len(mask)-1 else i-1;out.append(np.arange(s,e+1));s=None
 return out
def joint(df,metrics,label,seed,nperm=NP):
 piv={m:df[df.metric==m].pivot(index='subj',columns='time_ms',values='beta_uV').sort_index(axis=1) for m in metrics};subs=sorted(set.intersection(*[set(x.index) for x in piv.values()]));times=piv[metrics[0]].columns.to_numpy(float);a=np.stack([piv[m].reindex(subs).to_numpy() for m in metrics]);n=len(subs);crit=stats.t.ppf(.975,n-1);sq=np.sum(a*a,1);sm=np.sum(a,1);tv=(sm/n)/np.sqrt(((sq-sm*sm/n)/(n-1))/n);obs=[]
 for j,m in enumerate(metrics):
  for c in clusters(tv[j],crit):obs.append((m,c,float(np.abs(tv[j,c]).sum()),tv[j]))
 rng=np.random.default_rng(seed);null=np.zeros(nperm)
 for st in range(0,nperm,250):
  sg=rng.choice([-1.,1.],(min(250,nperm-st),n));ss=np.einsum('bn,mnt->bmt',sg,a);tt=(ss/n)/np.sqrt(((sq[None]-ss*ss/n)/(n-1))/n)
  for b in range(len(sg)):
   null[st+b]=max([float(np.abs(tt[b,j,c]).sum()) for j in range(len(metrics)) for c in clusters(tt[b,j],crit)]+[0])
 rows=[]
 for m,c,mass,t in obs:
  pk=c[np.argmax(np.abs(t[c]))];rows.append(dict(model=label,metric=m,n_participants=n,cluster_start_ms=times[c[0]],cluster_end_ms=times[c[-1]],cluster_mass_abs_t=mass,joint_familywise_p=(1+(null>=mass).sum())/(nperm+1),peak_time_ms=times[pk],peak_t=t[pk],n_permutations=nperm))
 return rows
def main():
 old=pd.read_csv(BASE/'image_metrics.csv');g=pd.read_csv(OUT/'geometric_metric_reproduced_values.csv')[['identity','picture_filename','G_new']];a2=pd.read_csv(OUT/'appearance_metric_sensitivity.csv')[['identity','picture_filename','A2']];im=old.merge(g,on=['identity','picture_filename']).merge(a2,on=['identity','picture_filename']);im['AR']=(im.A+im.A2);im['AR']=(im.AR-im.AR.mean())/im.AR.std(ddof=0)
 # G construct validation.
 fitg=smf.ols('G_new ~ fslim+eye+mouth+skin+C(identity)',im).fit(cov_type='HC3');pd.DataFrame([dict(term=t,estimate=fitg.params[t],se=fitg.bse[t],p=fitg.pvalues[t]) for t in ['fslim','eye','mouth','skin']]).to_csv(OUT/'geometric_construct_validation.csv',index=False,encoding='utf-8-sig')
 # Behavior: frozen original-A, A2 and robust composite models.
 beh=pd.read_csv(ROOT/'ipm_stage2_analysis'/'02_behavior_primary'/'behavior_trialwise_long.csv').merge(im[['identity','fslim','eye','mouth','skin','G_new','A','A2','AR','I']],left_on=['Identity','FSlim','Eye','Mouth','Skin'],right_on=['identity','fslim','eye','mouth','skin'],validate='many_to_one');specs={'original_A':'G_new+A+I','A2':'G_new+A2+I','robust_A_AR':'G_new+AR+I'};br=[]
 for name,rhs in specs.items():
  f=f'Rating ~ RatingDimension_Beauty*({rhs})+C(Identity)';fit=smf.mixedlm(f,beh,groups=beh.subj,re_formula='~RatingDimension_Beauty').fit(reml=False,method='lbfgs',maxiter=1000)
  for t in fit.params.index:br.append(dict(model=name,term=t,estimate=fit.params[t],se=fit.bse[t],p=fit.pvalues[t],aic=fit.aic,bic=fit.bic))
 pd.DataFrame(br).to_csv(OUT/'behavior_frozen_metric_models.csv',index=False,encoding='utf-8-sig')
 # Identity-specific fixed-effect coefficients (descriptive forest plot).
 fr=[]
 for ident,q in beh.groupby('Identity'):
  f=smf.ols('Rating ~ RatingDimension_Beauty*(G_new+A2+I)',q).fit(cov_type='cluster',cov_kwds={'groups':q.subj})
  for t in ['G_new','A2','I','RatingDimension_Beauty:G_new','RatingDimension_Beauty:A2','RatingDimension_Beauty:I']:
   fr.append(dict(identity=ident,term=t,estimate=f.params[t],se=f.bse[t],ci_low=f.params[t]-1.96*f.bse[t],ci_high=f.params[t]+1.96*f.bse[t],p=f.pvalues[t]))
 forest=pd.DataFrame(fr);forest.to_csv(OUT/'identity_level_behavior_coefficients.csv',index=False,encoding='utf-8-sig')
 fig,axs=plt.subplots(2,3,figsize=(11,6),sharey=True)
 for ax,(t,q) in zip(axs.ravel(),forest.groupby('term',sort=False)):
  y=np.arange(len(q));ax.errorbar(q.estimate,y,xerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt='o',color='#2563eb');ax.axvline(0,color='black',lw=.7);ax.set_yticks(y,q.identity);ax.set_title(t,fontsize=9)
 fig.tight_layout();fig.savefig(OUT/'identity_level_coefficient_forest.png',dpi=300);fig.savefig(OUT/'identity_level_coefficient_forest.pdf');plt.close(fig)
 # EEG first-level regressions for each frozen metric set, then unified N=28.
 ctx=Context(ROOT,OUT,SEED,100,False,setup_dirs(ROOT,'ipm_stage_2_6/logs/_metric_loader'));cmap=parse_eprime_condition_map(ROOT,ctx);ss=[load_subject(p,ctx,cmap) for p in find_eeg_files(ROOT)];ss=[s for s in ss if s is not None];assign_analysis_channels(ss,'n170_lpp_roi',ctx);centers=time_grid(ss[0].times,0,1000,20);allrows=[]
 sets={'original_A':['G_new','A','I'],'A2':['G_new','A2','I'],'robust_A_AR':['G_new','AR','I']}
 for s in ss:
  meta=s.metadata[(~s.metadata.is_control)&(~s.metadata.attention_check)&s.metadata.raw_cond_id.isin(range(2,18))].merge(im[['identity','cond_id','G_new','A','A2','AR','I']],left_on=['identity','raw_cond_id'],right_on=['identity','cond_id'],validate='many_to_one');dat=smooth_timecourse(extract_subject_matrix(s,meta,s.times).mean(2),s.times,30);Y=dat[:,nearest_time_indices(s.times,centers)];ident=pd.get_dummies(meta.identity,drop_first=True,dtype=float).to_numpy()
  for name,metrics in sets.items():
   X=np.column_stack([np.ones(len(meta)),meta[metrics].to_numpy(),ident]);B=np.linalg.pinv(X)@Y
   for j,v in enumerate(metrics,1):
    for tm,beta in zip(centers,B[j]):allrows.append(dict(model=name,subj=s.subj,metric=v,time_ms=tm,beta_uV=beta))
 bet=pd.DataFrame(allrows);bet.to_csv(OUT/'frozen_metric_subject_time_betas.csv',index=False,encoding='utf-8-sig');primary=bet[(bet.subj!='s18')];cr=[];sens=[]
 for k,(name,metrics) in enumerate(sets.items()):
  q=primary[primary.model==name];cr+=joint(q,metrics,name,SEED+1000+k)
  for h,held in enumerate(IDS):sens+=joint(q[q.subj.notna()].copy().merge(pd.DataFrame(),how='left') if False else q,metrics,f'{name}_LOIO_{held}',SEED+2000+k*10+h,nperm=1)
 # Correct LOIO requires refitting after holding identity; perform from stored raw subject objects.
 # The placeholder one-permutation rows above are discarded; dedicated refits follow.
 sens=[]
 for held in IDS:
  lr=[]
  for s in ss:
   if s.subj=='s18':continue
   meta=s.metadata[(~s.metadata.is_control)&(~s.metadata.attention_check)&s.metadata.raw_cond_id.isin(range(2,18))&(s.metadata.identity!=held)].merge(im[['identity','cond_id','G_new','A2','I']],left_on=['identity','raw_cond_id'],right_on=['identity','cond_id'],validate='many_to_one');dat=smooth_timecourse(extract_subject_matrix(s,meta,s.times).mean(2),s.times,30);Y=dat[:,nearest_time_indices(s.times,centers)];X=np.column_stack([np.ones(len(meta)),meta[['G_new','A2','I']].to_numpy(),pd.get_dummies(meta.identity,drop_first=True,dtype=float).to_numpy()]);B=np.linalg.pinv(X)@Y
   for j,v in enumerate(['G_new','A2','I'],1):
    for tm,beta in zip(centers,B[j]):lr.append(dict(subj=s.subj,metric=v,time_ms=tm,beta_uV=beta))
  sens+=joint(pd.DataFrame(lr),['G_new','A2','I'],f'A2_LOIO_{held}',SEED+3000+IDS.index(held))
 pd.DataFrame(cr).to_csv(OUT/'frozen_metric_joint_eeg_clusters.csv',index=False,encoding='utf-8-sig');pd.DataFrame(sens).to_csv(OUT/'frozen_metric_identity_sensitivity.csv',index=False,encoding='utf-8-sig')
 # Append model result synopsis to metric reports.
 sig=pd.DataFrame(cr);summary=sig[sig.joint_familywise_p<.05].to_dict('records');
 with (OUT/'geometric_metric_reproduction.md').open('a',encoding='utf-8') as f:f.write('\n## Frozen-model reruns\n\nG construct validation, behavior coefficients and N=28 joint EEG reruns are in `geometric_construct_validation.csv`, `behavior_frozen_metric_models.csv`, and `frozen_metric_joint_eeg_clusters.csv`. Significant corrected clusters: '+str(summary)+'\n')
 with (OUT/'appearance_metric_replication.md').open('a',encoding='utf-8') as f:f.write('\n## Frozen-model reruns\n\nOriginal A, A2, and the prespecified robust composite AR=(standardized A+A2) were each entered in a full three-predictor joint model with G_new and I. Results are in `behavior_frozen_metric_models.csv`, `frozen_metric_joint_eeg_clusters.csv`, and `frozen_metric_identity_sensitivity.csv`.\n')
if __name__=='__main__':main()
