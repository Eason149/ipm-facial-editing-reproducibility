#!/usr/bin/env python
"""Stage 2.6 sample reconciliation and frozen-model sample sensitivity."""
from pathlib import Path
import os,sys,json,logging
REPO_ROOT=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('IPM_DATA_ROOT',REPO_ROOT/'data')).resolve(); BASE=ROOT/'ipm_visual_information_gate'; OUT=ROOT/'ipm_stage_2_6'
sys.path.insert(0,str(Path(__file__).resolve().parent));pkg=ROOT/'RSA_time_resolved_analysis'/'.python-packages'
if pkg.exists():sys.path.append(str(pkg))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.codex_matplotlib_cache'))
import numpy as np,pandas as pd
from scipy import stats
from run_time_resolved_factor_decoding import Context,setup_dirs,parse_eprime_condition_map,find_eeg_files,load_subject
from run_erp_dynamics_persistence import find_clusters_two_sided
OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'logs'/'eeg_sample_validation.log',encoding='utf-8'),logging.StreamHandler()])
METRICS=['G','A','I'];SEED=20260818;NPERM=10000

def clusters_1d(t):
    mask=np.abs(t)>=stats.t.ppf(.975,DF_GLOBAL)
    out=[];start=None
    for i,v in enumerate(mask):
        if v and start is None:start=i
        if start is not None and (not v or i==len(mask)-1):
            end=i if v and i==len(mask)-1 else i-1;out.append(np.arange(start,end+1));start=None
    return out

def joint(df,label,seed):
    global DF_GLOBAL
    piv={m:df[df.metric==m].pivot(index='subj',columns='time_ms',values='beta_uV').sort_index(axis=1) for m in METRICS}
    subs=sorted(set.intersection(*[set(x.index) for x in piv.values()]));times=piv['G'].columns.to_numpy(float);a=np.stack([piv[m].reindex(subs).to_numpy() for m in METRICS])
    n=len(subs);DF_GLOBAL=n-1;sumsq=np.sum(a*a,axis=1);sm=np.sum(a,axis=1);tv=(sm/n)/np.sqrt(((sumsq-sm*sm/n)/(n-1))/n)
    obs=[]
    for j,m in enumerate(METRICS):
        for c in clusters_1d(tv[j]):obs.append((m,c,float(np.abs(tv[j,c]).sum()),tv[j]))
    rng=np.random.default_rng(seed);null=np.zeros(NPERM);chunk=250
    for st in range(0,NPERM,chunk):
        signs=rng.choice(np.array([-1.,1.]),size=(min(chunk,NPERM-st),n));signed=np.einsum('bn,mnt->bmt',signs,a);tt=(signed/n)/np.sqrt(((sumsq[None,:,:]-signed*signed/n)/(n-1))/n)
        for b in range(len(signs)):
            mx=0.
            for j in range(3):
                for c in clusters_1d(tt[b,j]):mx=max(mx,float(np.abs(tt[b,j,c]).sum()))
            null[st+b]=mx
    rows=[]
    for m,c,mass,t in obs:
        pk=c[np.argmax(np.abs(t[c]))]
        rows.append(dict(analysis=label,excluded_participant=label.replace('LOPO_','') if label.startswith('LOPO_') else '',n_participants=n,participants=';'.join(subs),metric=m,cluster_start_ms=times[c[0]],cluster_end_ms=times[c[-1]],cluster_mass_abs_t=mass,joint_familywise_p=(1+(null>=mass).sum())/(NPERM+1),peak_time_ms=times[pk],peak_t=t[pk],n_permutations=NPERM,cluster_forming_p=.05,seed=seed))
    if not rows:rows=[dict(analysis=label,excluded_participant='',n_participants=n,participants=';'.join(subs),metric='',cluster_start_ms=np.nan,cluster_end_ms=np.nan,cluster_mass_abs_t=np.nan,joint_familywise_p=np.nan,peak_time_ms=np.nan,peak_t=np.nan,n_permutations=NPERM,cluster_forming_p=.05,seed=seed)]
    logging.info('%s complete n=%d clusters=%d',label,n,len(obs));return rows

def main():
    b=pd.read_csv(BASE/'eeg'/'subject_time_resolved_GAI_betas.csv');allsubs=sorted(b.subj.unique(),key=lambda x:int(x[1:]));n28=[s for s in allsubs if s!='s18']
    rows=[];rows+=joint(b,'N29',SEED);rows+=joint(b[b.subj.isin(n28)],'N28_unified_primary',SEED+1);rows+=joint(b[b.subj!='s18'],'leave_s18_out',SEED+2)
    for i,s in enumerate(allsubs):rows+=joint(b[b.subj!=s],f'LOPO_{s}',SEED+100+i)
    out=pd.DataFrame(rows);out.to_csv(OUT/'eeg_sample_sensitivity.csv',index=False,encoding='utf-8-sig')
    # Recover condition counts directly from derivative metadata.
    ctx=Context(ROOT,OUT,SEED,100,False,setup_dirs(ROOT,'ipm_stage_2_6/logs/_loader'));cmap=parse_eprime_condition_map(ROOT,ctx)
    subjects=[load_subject(p,ctx,cmap) for p in find_eeg_files(ROOT)];s18=next(s for s in subjects if s is not None and s.subj=='s18')
    q=s18.metadata[(~s18.metadata.is_control)&(~s18.metadata.attention_check)&s18.metadata.raw_cond_id.isin(range(2,18))]
    counts=q.groupby(['raw_cond_id','FSlim','Eye','Mouth','Skin']).size().rename('effective_trials').reset_index();counts.to_csv(OUT/'s18_condition_counts.csv',index=False,encoding='utf-8-sig')
    sig=out[(out.joint_familywise_p<.05)&out.metric.notna()]
    lopo=sig[sig.analysis.str.startswith('LOPO_')]
    report=f'''# EEG Sample Reconciliation\n\n## Frozen primary rule\n\nThe unified primary EEG sample is **N=28**, determined by the pre-existing data-quality rule: exclude s5 by the default EEG subject list and require at least 8 effective trials in every factorial cell after pooling the four identities. s18 fails this rule. N=29 is retained only as a sensitivity analysis; significance was not used to choose the primary sample.\n\n## Participant lists\n\n- N=29: {', '.join(allsubs)}.\n- N=28: {', '.join(n28)}.\n- Difference: s18 only (s5 is absent from both).\n\n## Why s18 entered the continuous model\n\nThe continuous G/A/I script inherited `DEFAULT_SUBJECTS`, which excludes s5 but includes s18, and did not reapply the factorial-cell minimum. Continuous regression can remain full-rank with incomplete factorial-cell coverage because G/A/I are trial-level continuous predictors; nevertheless, using a different quality rule for the same EEG dataset is not defensible. This was a pipeline omission, not a result-selected choice.\n\n## s18 coverage\n\n- Formal effective trials: {len(q)}.\n- Factorial cells present after pooling identities: {len(counts)}/16.\n- Minimum/median/maximum trials per cell: {counts.effective_trials.min()}/{counts.effective_trials.median():.1f}/{counts.effective_trials.max()}.\n- Cells below 8 trials: {(counts.effective_trials<8).sum()}.\n\nFull counts are in `s18_condition_counts.csv`.\n\n## Frozen analysis specification\n\nAll variants use the already-frozen G/A/I betas, posterior 25-channel ROI, 0–1000 ms in 20-ms steps, two-sided pointwise cluster-forming threshold p<.05, absolute-t cluster mass, participant-level sign flip, and 10,000 joint maximum-cluster permutations across G/A/I × time.\n\n## Sensitivity scope\n\nThe CSV contains N=29, N=28, an explicitly labeled leave-s18-out duplicate check, and all 29 leave-one-participant-out versions. Significant cluster rows across LOPO versions: {len(lopo)}. These analyses assess influence only and do not redefine the primary sample.\n\n## Timing of rules\n\nThe N=28 rule is documented in the prior Stage 2 exclusion-flow artifact before Stage 2.6. The N=29 continuous script used an earlier default loader rule but omitted the cell check. There is no evidence that either rule was chosen after inspecting the Stage 2.6 sensitivity results; however, the continuous analysis's inconsistent implementation means only the N=28 rule is retained prospectively here.\n'''
    (OUT/'eeg_sample_reconciliation.md').write_text(report,encoding='utf-8')
    (OUT/'eeg_sample_manifest.json').write_text(json.dumps({'primary_sample':'N28','n29':allsubs,'n28':n28,'s18_min_cell':int(counts.effective_trials.min()),'s18_cells_below_8':int((counts.effective_trials<8).sum()),'n_permutations':NPERM},indent=2),encoding='utf-8')
if __name__=='__main__':main()
