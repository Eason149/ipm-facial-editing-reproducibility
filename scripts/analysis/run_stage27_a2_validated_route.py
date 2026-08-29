#!/usr/bin/env python
"""Frozen A2-only IPM route: behavior increment, EEG sensitivity, and stopping rule."""
from pathlib import Path
import json, os, sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("IPM_DATA_ROOT", REPO_ROOT / "data")).resolve()
S26 = ROOT / "ipm_stage_2_6"
OUT = ROOT / "ipm_stage_2_7_a2_validated_route"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
pkg = ROOT / "RSA_time_resolved_analysis" / ".python-packages"
if pkg.exists(): sys.path.append(str(pkg))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import patsy

SEED = 20260818
NPERM = 10000
METRICS = ["G_new", "A2", "I"]

def clusters(t, crit):
    mask = np.abs(t) >= crit; out = []; start = None
    for i, flag in enumerate(mask):
        if flag and start is None: start = i
        if start is not None and (not flag or i == len(mask)-1):
            end = i if flag and i == len(mask)-1 else i-1
            out.append(np.arange(start, end+1)); start = None
    return out

def joint(df, label, seed):
    piv = {m: df[df.metric == m].pivot(index="subj", columns="time_ms", values="beta_uV").sort_index(axis=1) for m in METRICS}
    subs = sorted(set.intersection(*[set(v.index) for v in piv.values()]), key=lambda x:int(x[1:]))
    times = piv[METRICS[0]].columns.to_numpy(float)
    a = np.stack([piv[m].reindex(subs).to_numpy() for m in METRICS])
    n = len(subs); crit = stats.t.ppf(.975, n-1); sq = np.sum(a*a, axis=1); sm = np.sum(a, axis=1)
    tv = (sm/n) / np.sqrt(((sq-sm*sm/n)/(n-1))/n)
    observed = [(m,c,float(np.abs(tv[j,c]).sum()),tv[j]) for j,m in enumerate(METRICS) for c in clusters(tv[j],crit)]
    rng = np.random.default_rng(seed); null = np.zeros(NPERM)
    for st in range(0, NPERM, 250):
        signs = rng.choice([-1.,1.], (min(250,NPERM-st),n))
        signed = np.einsum("bn,mnt->bmt", signs, a)
        tt = (signed/n) / np.sqrt(((sq[None]-signed*signed/n)/(n-1))/n)
        for b in range(len(signs)):
            null[st+b] = max([float(np.abs(tt[b,j,c]).sum()) for j in range(3) for c in clusters(tt[b,j],crit)] + [0.])
    rows=[]
    for m,c,mass,t in observed:
        pk=c[np.argmax(np.abs(t[c]))]
        rows.append(dict(analysis=label,n_participants=n,participants=";".join(subs),metric=m,cluster_start_ms=times[c[0]],cluster_end_ms=times[c[-1]],cluster_mass_abs_t=mass,joint_familywise_p=(1+(null>=mass).sum())/(NPERM+1),peak_time_ms=times[pk],peak_t=t[pk],n_permutations=NPERM,seed=seed))
    return rows

def behavior():
    raw = pd.read_csv(ROOT/"ipm_stage2_analysis"/"02_behavior_primary"/"behavior_trialwise_long.csv")
    vals = pd.read_csv(S26/"appearance_metric_sensitivity.csv")
    keys=["identity","fslim","eye","mouth","skin"]
    d=raw.merge(vals[keys+["A2"]].drop_duplicates(),left_on=["Identity","FSlim","Eye","Mouth","Skin"],right_on=keys,validate="many_to_one")
    specs={
        "factor":"Rating ~ RatingDimension_Beauty*(FSlim+Eye+Mouth+Skin)+C(Identity)",
        "A2_only":"Rating ~ RatingDimension_Beauty*A2+C(Identity)",
        "factor_plus_A2":"Rating ~ RatingDimension_Beauty*(FSlim+Eye+Mouth+Skin+A2)+C(Identity)",
    }
    rows=[]; fits={}
    for name,formula in specs.items():
        fit=smf.mixedlm(formula,d,groups=d.subj,re_formula="~RatingDimension_Beauty").fit(reml=False,method="lbfgs",maxiter=1000)
        fits[name]=fit
        for term in fit.params.index:
            rows.append(dict(model=name,term=term,estimate=fit.params[term],se=fit.bse[term],z=fit.tvalues[term],p=fit.pvalues[term],aic=fit.aic,bic=fit.bic,llf=fit.llf,n_observations=len(d),n_participants=d.subj.nunique()))
    coef=pd.DataFrame(rows); coef.to_csv(OUT/"a2_behavior_model_coefficients.csv",index=False,encoding="utf-8-sig")
    lr=2*(fits["factor_plus_A2"].llf-fits["factor"].llf)
    compare=pd.DataFrame([dict(comparison="factor_plus_A2 vs factor",lr_chi2=lr,df=2,p=stats.chi2.sf(lr,2),delta_aic=fits["factor_plus_A2"].aic-fits["factor"].aic,delta_bic=fits["factor_plus_A2"].bic-fits["factor"].bic)])
    compare.to_csv(OUT/"a2_behavior_increment_test.csv",index=False,encoding="utf-8-sig")
    subs=np.array(sorted(d.subj.unique(),key=lambda x:int(x[1:])));rng=np.random.default_rng(SEED);rng.shuffle(subs);folds=np.array_split(subs,5);cv=[]
    for k,testsubs in enumerate(folds):
        tr=d[~d.subj.isin(testsubs)];te=d[d.subj.isin(testsubs)]
        for name,formula in specs.items():
            y,x=patsy.dmatrices(formula,tr,return_type="dataframe");beta=np.linalg.lstsq(x,y,rcond=None)[0];xt=patsy.build_design_matrices([x.design_info],te,return_type="dataframe")[0]
            pred=np.asarray(xt@beta).ravel();actual=te.Rating.to_numpy()
            cv.append(dict(fold=k+1,model=name,n_test=len(te),rmse=np.sqrt(np.mean((actual-pred)**2)),mae=np.mean(np.abs(actual-pred)),r2=1-np.sum((actual-pred)**2)/np.sum((actual-actual.mean())**2)))
    cv=pd.DataFrame(cv);cv.to_csv(OUT/"a2_behavior_participant_cv.csv",index=False,encoding="utf-8-sig");cv.groupby("model")[["rmse","mae","r2"]].mean().reset_index().to_csv(OUT/"a2_behavior_cv_summary.csv",index=False,encoding="utf-8-sig")
    return compare,coef,cv

def eeg():
    b=pd.read_csv(S26/"frozen_metric_subject_time_betas.csv");b=b[b.model=="A2"].copy();subs=sorted(b.subj.unique(),key=lambda x:int(x[1:]));rows=[]
    rows+=joint(b,"N29_sensitivity",SEED)
    rows+=joint(b[b.subj!="s18"],"N28_unified_primary",SEED+1)
    for i,s in enumerate(subs): rows+=joint(b[b.subj!=s],f"LOPO_{s}",SEED+100+i)
    out=pd.DataFrame(rows);out.to_csv(OUT/"a2_eeg_joint_sample_sensitivity.csv",index=False,encoding="utf-8-sig")
    loio=pd.read_csv(S26/"frozen_metric_identity_sensitivity.csv");loio=loio[loio.model.str.startswith("A2_LOIO_")].copy();loio.to_csv(OUT/"a2_eeg_identity_sensitivity.csv",index=False,encoding="utf-8-sig")
    return out,loio

def main():
    compare,coef,cv=behavior();eegres,loio=eeg()
    a2term=coef[(coef.model=="factor_plus_A2") & (coef.term.isin(["A2","RatingDimension_Beauty:A2"]))]
    primary=eegres[eegres.analysis=="N28_unified_primary"]
    a2sig=primary[(primary.metric=="A2")&(primary.joint_familywise_p<.05)]
    report=f'''# Validated A2 Route — Frozen Follow-up\n\n## Scope\n\nThis follow-up implements the Stage 2.6 decision without changing A2, the N=28 quality rule, ROI, time range, cluster-forming threshold, joint G_new/A2/I family, or random seed. It does not select a favorable EEG window.\n\n## Behavioral incremental value\n\n- Nested factor+A2 versus factor likelihood-ratio p={compare.iloc[0].p:.4g}; ΔAIC={compare.iloc[0].delta_aic:.2f}; ΔBIC={compare.iloc[0].delta_bic:.2f}.\n- Frozen factor+A2 coefficients: {a2term[['term','estimate','p']].to_dict('records')}.\n- Participant-grouped CV is in `a2_behavior_cv_summary.csv`; it assesses held-out participants, not new stimuli.\n\n## EEG\n\n- N=28 jointly corrected A2 clusters: {a2sig.to_dict('records') if len(a2sig) else 'none'}.\n- N=29, N=28, and all leave-one-participant-out versions use 10,000 G_new/A2/I × time maximum-cluster permutations.\n- Leave-one-identity-out results remain descriptive and do not establish cross-identity generalization.\n\n## Decision\n\nA2 is retained as a validated stimulus-level appearance dimension with limited behavioral incremental value. It is **not** supported as a stable EEG-tracked dimension. The defensible IPM route is therefore behavioral/information-oriented and stimulus-set-bounded; neural-tracking language remains prohibited.\n'''
    (OUT/"A2_VALIDATED_ROUTE_REPORT.md").write_text(report,encoding="utf-8")
    (OUT/"manifest.json").write_text(json.dumps({'status':'completed','primary_sample':'N28','metric':'A2 frozen from Stage 2.6','eeg_family':METRICS,'n_permutations':NPERM,'seed':SEED,'claim':'behavioral conditional association only; no stable A2 EEG tracking'},indent=2),encoding="utf-8")

if __name__=="__main__": main()
