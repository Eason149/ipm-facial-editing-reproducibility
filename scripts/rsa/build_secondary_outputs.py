from pathlib import Path
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO_ROOT=Path(__file__).resolve().parents[2];SOURCE=Path(os.environ.get('IPM_DATA_ROOT',REPO_ROOT/'data')).resolve();R=SOURCE/'RSA_multimodal_geometry';F=R/'figures'
loo=pd.read_excel(R/'RSA_11_Identity_Sensitivity.xlsx',sheet_name='LeaveOneIdentityOut');spec=pd.read_excel(R/'RSA_Specification_Results.xlsx');neg=pd.read_excel(R/'RSA_Negative_Controls.xlsx');vp=pd.read_excel(R/'RSA_09_Variance_Partitioning.xlsx',sheet_name='Summary')
with pd.ExcelWriter(R/'RSA_Secondary_Results.xlsx',engine='openpyxl') as w:loo.to_excel(w,index=False,sheet_name='Identity_LOO');spec.to_excel(w,index=False,sheet_name='Specifications');neg.to_excel(w,index=False,sheet_name='Negative_controls');vp.to_excel(w,index=False,sheet_name='Variance_partitioning')
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'});colors=['#0072B2','#D55E00','#009E73','#CC79A7']
fig,ax=plt.subplots(figsize=(8,4))
for i,(name,g) in enumerate(vp.groupby('Component')):ax.plot(g.TimeMs,g.DeltaR2,label=name,color=colors[i%4],lw=1.8)
ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Time (ms)',ylabel='Unique ΔR²');ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(F/'Figure_4_Commonality_Timecourse.png',dpi=300);fig.savefig(F/'Figure_4_Commonality_Timecourse.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));x=range(len(spec));ax.scatter(x,spec.MeanBeta,c=[colors[['Gaze','Beauty','Naturalness','BeautyMinusNaturalness'].index(c)] for c in spec.Contrast]);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Mean RSA coefficient');ax.set_xticks(list(x));ax.set_xticklabels(['20' if '20-' in s else '40' for s in spec.Specification],rotation=0);fig.tight_layout();fig.savefig(F/'Figure_6_RSA_Specification_Curve.png',dpi=300);fig.savefig(F/'Figure_6_RSA_Specification_Curve.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));ax.barh(neg.Control,neg.MeanBeta,color=['#D55E00' if not v else '#0072B2' for v in neg.Pass]);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Mean control coefficient');fig.tight_layout();fig.savefig(F/'Figure_7_Negative_Controls.png',dpi=300);fig.savefig(F/'Figure_7_Negative_Controls.pdf');plt.close(fig)

# Separate requested primary panels without implying significance.
pr=pd.read_excel(R/'RSA_Primary_Results.xlsx',sheet_name='Coefficients')
for fname,contrasts in [('Figure_5_EEG_Gaze_RSA',['Gaze']),('Figure_6_EEG_Rating_RSA',['Beauty','Naturalness']),('Figure_7_Beauty_Minus_Naturalness',['BeautyMinusNaturalness'])]:
    fig,ax=plt.subplots(figsize=(7,4))
    for i,c in enumerate(contrasts):
        g=pr[pr.Contrast.eq(c)].groupby('TimeMs').Beta.agg(['mean','sem']).reset_index();ax.plot(g.TimeMs,g['mean'],label=c,color=colors[i+1],lw=2);ax.fill_between(g.TimeMs,g['mean']-g['sem'],g['mean']+g['sem'],color=colors[i+1],alpha=.18)
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Time (ms)',ylabel='Standardized RSA coefficient');ax.legend(frameon=False);fig.tight_layout();fig.savefig(F/(fname+'.png'),dpi=300);fig.savefig(F/(fname+'.pdf'));plt.close(fig)

import numpy as np
z=np.load(R/'outputs/RSA_08_EEG_RDMs.npz',allow_pickle=True);e=z['EEG'];avg=np.nanmean(e,axis=(0,1));strength=np.nanmean(np.abs(avg[:,np.triu_indices(16,1)[0],np.triu_indices(16,1)[1]]),axis=1)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(z['times'],strength,color=colors[0],lw=2);ax.set(xlabel='Time (ms)',ylabel='Mean absolute crossnobis distance');fig.tight_layout();fig.savefig(F/'Figure_3_EEG_RDM_Timecourse.png',dpi=300);fig.savefig(F/'Figure_3_EEG_RDM_Timecourse.pdf');plt.close(fig)
