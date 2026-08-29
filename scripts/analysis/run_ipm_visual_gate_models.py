#!/usr/bin/env python
from pathlib import Path
import os,sys,json,logging
REPO_ROOT=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('IPM_DATA_ROOT',REPO_ROOT/'data')).resolve();OUT=ROOT/'ipm_visual_information_gate'
sys.path.insert(0,str(Path(__file__).resolve().parent));pkg=ROOT/'RSA_time_resolved_analysis'/'.python-packages'
if pkg.exists():sys.path.append(str(pkg))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.codex_matplotlib_cache'))
import numpy as np,pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm,ols
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from run_time_resolved_factor_decoding import Context,setup_dirs,parse_eprime_condition_map,find_eeg_files,load_subject,assign_analysis_channels,extract_subject_matrix
from run_erp_dynamics_persistence import smooth_timecourse,nearest_time_indices,time_grid,find_clusters_two_sided

SEED=20260818;N_PERM=10000;RNG=np.random.default_rng(SEED); METRICS=['G','A','I']; IDS=['F_1','F_2','M_1','M_2']
for x in ['behavior','eeg','sensitivity','negative_controls','figures','logs']: (OUT/x).mkdir(parents=True,exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'logs'/'gate_models.log',encoding='utf-8'),logging.StreamHandler()])
def save(d,p):d.to_csv(p,index=False,encoding='utf-8-sig');logging.info('wrote %s %s',p,d.shape)
def joint_clusters(betas,nperm=N_PERM,seed=SEED):
    piv={m:betas[betas.metric==m].pivot(index='subj',columns='time_ms',values='beta_uV').sort_index(axis=1) for m in METRICS};subs=sorted(set.intersection(*[set(x.index) for x in piv.values()]));times=piv['G'].columns.to_numpy(float);arr={m:piv[m].reindex(subs).to_numpy() for m in METRICS}
    obs=[]
    for m,a in arr.items():
        tv,pv=stats.ttest_1samp(a,0,axis=0)
        for c in find_clusters_two_sided(tv,pv):obs.append((m,c,float(np.abs(tv[c]).sum()),tv,a))
    rng=np.random.default_rng(seed);null=np.zeros(nperm)
    for k in range(nperm):
        signs=rng.choice([-1.,1.],(len(subs),1));mx=0
        for a in arr.values():
            tv,pv=stats.ttest_1samp(a*signs,0,axis=0);cs=find_clusters_two_sided(tv,pv);mx=max(mx,max([float(np.abs(tv[c]).sum()) for c in cs],default=0))
        null[k]=mx
    rows=[];step=np.median(np.diff(times))
    for m,c,mass,tv,a in obs:
        pk=c[np.argmax(np.abs(tv[c]))];p=(1+(null>=mass).sum())/(nperm+1)
        rows.append(dict(metric=m,cluster_start_ms=times[c[0]],cluster_end_ms=times[c[-1]],duration_ms=times[c[-1]]-times[c[0]]+step,cluster_mass_abs_t=mass,joint_familywise_p=p,peak_time_ms=times[pk],peak_t=tv[pk],peak_mean_beta_uV=a[:,pk].mean(),n_subjects=len(subs),n_permutations=nperm))
    return pd.DataFrame(rows).sort_values('joint_familywise_p') if rows else pd.DataFrame(columns=['metric','cluster_start_ms','cluster_end_ms','joint_familywise_p'])

def behavior(metrics):
    logging.info('Stage 5-6 behavior models')
    d=pd.read_csv(ROOT/'ipm_stage2_analysis'/'02_behavior_primary'/'behavior_trialwise_long.csv')
    mm=metrics[['identity','fslim','eye','mouth','skin','G','A','I']].drop_duplicates()
    d=d.merge(mm,left_on=['Identity','FSlim','Eye','Mouth','Skin'],right_on=['identity','fslim','eye','mouth','skin'],validate='many_to_one')
    formulas={
      'factor':'Rating ~ RatingDimension_Beauty*(FSlim+Eye+Mouth+Skin)+C(Identity)',
      'information':'Rating ~ RatingDimension_Beauty*(G+A+I)+C(Identity)',
      'combined':'Rating ~ RatingDimension_Beauty*(FSlim+Eye+Mouth+Skin+G+A+I)+C(Identity)'}
    rows=[]
    for name,f in formulas.items():
        fit=mixedlm(f,d,groups=d.subj,re_formula='~RatingDimension_Beauty').fit(reml=False,method='lbfgs',maxiter=1000)
        for term in fit.params.index:
            rows.append(dict(model=name,term=term,estimate=fit.params[term],se=fit.bse[term],z=fit.tvalues[term],p=fit.pvalues[term],aic=fit.aic,bic=fit.bic,llf=fit.llf,n=len(d)))
    save(pd.DataFrame(rows),OUT/'behavior'/'mixed_model_coefficients.csv')
    # Five-fold participant-group CV; prediction uses only fixed effects, appropriate for unseen participants.
    subs=np.array(sorted(d.subj.unique()));rng=np.random.default_rng(SEED);rng.shuffle(subs);folds=np.array_split(subs,5);cv=[]
    import patsy
    for k,testsubs in enumerate(folds):
        tr=d[~d.subj.isin(testsubs)];te=d[d.subj.isin(testsubs)]
        for name,f in formulas.items():
            y,x=patsy.dmatrices(f,tr,return_type='dataframe');fit=np.linalg.lstsq(x,y,rcond=None)[0];xt=patsy.build_design_matrices([x.design_info],te,return_type='dataframe')[0];pred=np.asarray(xt@fit).ravel();actual=te.Rating.to_numpy();
            cv.append(dict(fold=k+1,model=name,n=len(te),rmse=np.sqrt(np.mean((actual-pred)**2)),mae=np.mean(np.abs(actual-pred)),r2=1-np.sum((actual-pred)**2)/np.sum((actual-actual.mean())**2)))
    cv=pd.DataFrame(cv);save(cv,OUT/'behavior'/'participant_group_cv.csv');save(cv.groupby('model')[['rmse','mae','r2']].mean().reset_index(),OUT/'behavior'/'participant_group_cv_summary.csv')
    lo=[]
    for held in IDS:
        sub=d[d.Identity!=held]
        fit=ols(formulas['information'],sub).fit(cov_type='HC3')
        for term in ['G','A','I','RatingDimension_Beauty:G','RatingDimension_Beauty:A','RatingDimension_Beauty:I']:
            lo.append(dict(held_out_identity=held,term=term,estimate=fit.params.get(term,np.nan),p=fit.pvalues.get(term,np.nan)))
    save(pd.DataFrame(lo),OUT/'sensitivity'/'behavior_leave_one_identity_out.csv')
    return d

def eeg(metrics):
    logging.info('Stage 7-10 EEG information models')
    ctx=Context(ROOT,OUT,SEED,100,False,setup_dirs(ROOT,'ipm_visual_information_gate/logs/_loader'));cmap=parse_eprime_condition_map(ROOT,ctx)
    ss=[load_subject(p,ctx,cmap) for p in find_eeg_files(ROOT)];ss=[s for s in ss if s is not None];assign_analysis_channels(ss,'n170_lpp_roi',ctx)
    centers=time_grid(ss[0].times,0,1000,20);pre=time_grid(ss[0].times,-200,0,20);mm=metrics[['identity','cond_id','G','A','I']]
    rows=[];prerows=[];lo=[];integrated=[];shuffle_inputs=[]
    for s in ss:
        meta=s.metadata[(~s.metadata.is_control)&(~s.metadata.attention_check)&s.metadata.raw_cond_id.isin(range(2,18))].copy().merge(mm,left_on=['identity','raw_cond_id'],right_on=['identity','cond_id'],validate='many_to_one')
        dat=extract_subject_matrix(s,meta,s.times).mean(axis=2);dat=smooth_timecourse(dat,s.times,30)
        ident=pd.get_dummies(meta.identity,drop_first=True,dtype=float).to_numpy();X=np.column_stack([np.ones(len(meta)),meta[METRICS].to_numpy(),ident]);Y=dat[:,nearest_time_indices(s.times,centers)];B=np.linalg.pinv(X)@Y
        for j,m in enumerate(METRICS,1):
            for tm,v in zip(centers,B[j]):rows.append(dict(subj=s.subj,metric=m,time_ms=tm,beta_uV=v))
        Yp=dat[:,nearest_time_indices(s.times,pre)];Bp=np.linalg.pinv(X)@Yp
        for j,m in enumerate(METRICS,1):
            for tm,v in zip(pre,Bp[j]):prerows.append(dict(subj=s.subj,metric=m,time_ms=tm,beta_uV=v))
        yi=Y.mean(axis=1);bi=np.linalg.pinv(X)@yi
        shuffle_inputs.append((s.subj,meta[METRICS].to_numpy(),ident,yi))
        for j,m in enumerate(METRICS,1):integrated.append(dict(subj=s.subj,metric=m,beta_uV=bi[j]))
        for held in IDS:
            q=meta.identity!=held;Xi=np.column_stack([np.ones(q.sum()),meta.loc[q,METRICS].to_numpy(),pd.get_dummies(meta.loc[q,'identity'],drop_first=True,dtype=float).to_numpy()]);Bb=np.linalg.pinv(Xi)@Y[q]
            for j,m in enumerate(METRICS,1):
                for tm,v in zip(centers,Bb[j]):lo.append(dict(held_out_identity=held,subj=s.subj,metric=m,time_ms=tm,beta_uV=v))
    b=pd.DataFrame(rows);pb=pd.DataFrame(prerows);ld=pd.DataFrame(lo);save(b,OUT/'eeg'/'subject_time_resolved_GAI_betas.csv');save(pb,OUT/'negative_controls'/'prestimulus_GAI_betas.csv');save(ld,OUT/'sensitivity'/'eeg_leave_one_identity_out_betas.csv')
    cl=joint_clusters(b);pcl=joint_clusters(pb,seed=SEED+1);save(cl,OUT/'eeg'/'joint_GAI_cluster_results.csv');save(pcl,OUT/'negative_controls'/'prestimulus_joint_cluster_results.csv')
    lor=[]
    for held,q in ld.groupby('held_out_identity'):
        z=joint_clusters(q,nperm=2000,seed=SEED+100+IDS.index(held));z.insert(0,'held_out_identity',held);lor.append(z)
    save(pd.concat(lor,ignore_index=True),OUT/'sensitivity'/'eeg_leave_one_identity_out_clusters.csv')
    # Genuine shuffled-stimulus-label negative control: permute trialwise G/A/I labels,
    # refit every participant, then use the family maximum across the three group t tests.
    integ=pd.DataFrame(integrated);wide=integ.pivot(index='subj',columns='metric',values='beta_uV');obs=max(abs(stats.ttest_1samp(wide,0,axis=0).statistic));null=[];rng=np.random.default_rng(SEED+9)
    for _ in range(1000):
        pb=[]
        for subj,vals,ident,yi in shuffle_inputs:
            perm=vals[rng.permutation(len(vals))];Xp=np.column_stack([np.ones(len(vals)),perm,ident]);pb.append((np.linalg.pinv(Xp)@yi)[1:4])
        null.append(max(abs(stats.ttest_1samp(np.asarray(pb),0,axis=0).statistic)))
    save(pd.DataFrame([dict(observed_family_max_abs_t=obs,shuffle_signflip_p=(1+(np.asarray(null)>=obs).sum())/1001,n_shuffles=1000)]),OUT/'negative_controls'/'shuffled_label_integrated_test.csv')
    fig,ax=plt.subplots(figsize=(8,4));
    for m,c in zip(METRICS,['#2563eb','#f59e0b','#dc2626']):
        p=b[b.metric==m].pivot(index='subj',columns='time_ms',values='beta_uV');mu=p.mean();se=p.sem();ax.plot(mu.index,mu,label=m,color=c);ax.fill_between(mu.index,mu-se,mu+se,color=c,alpha=.18)
    ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7);ax.set(xlabel='Time from face onset (ms)',ylabel='beta (uV per SD)',title='Continuous visual-information EEG coefficients');ax.legend();fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'eeg_GAI_timecourses.{ext}',dpi=300 if ext=='png' else None)
    plt.close(fig);return cl,pcl

def main():
    metrics=pd.read_csv(OUT/'image_metrics.csv');behavior(metrics);cl,pcl=eeg(metrics)
    logging.info('Stages 5-10 complete')
if __name__=='__main__':main()
