"""Generate descriptive QC and per-recording figures without inferential stars."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':200,'font.family':'sans-serif'})

def save(figure,out,stem):
    figure.tight_layout()
    figure.savefig(out/(stem+'.png'))
    figure.savefig(out/(stem+'.pdf'))
    plt.close(figure)

def time_edges(times):
    times=np.asarray(times,dtype=float)/60
    if len(times)<2:
        return np.array([times[0]-.5,times[0]+.5])
    edges=np.r_[(3*times[0]-times[1])/2,(times[:-1]+times[1:])/2,
                (3*times[-1]-times[-2])/2]
    return edges

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):
        parser.error('Figure directory must be empty; select a new --out.')
    names=json.loads((args.results/'params.json').read_text(encoding='utf-8'))['names']
    cf=pd.read_csv(args.results/'colocalisation.csv')
    tracks=pd.read_csv(args.results/'tracks.csv')
    displacement=pd.read_csv(args.results/'displacement.csv')
    summary=cf.groupby(['file','structure','structure_class'],as_index=False).agg(
        mean_matched_fraction=('matched_fraction','mean'),mean_spatial_null=('null_mean','mean'),
        n_frames=('frame','nunique'),length_um=('length_um','first'))
    summary.to_csv(args.out/'summary_per_structure.csv',index=False)
    with np.load(args.results/'masks.npz',allow_pickle=False) as masks:
        for key in [k for k in masks.files if k.endswith('|tmean_A')]:
            stem=key.split('|')[0]
            a,b=masks[stem+'|tmean_A'],masks[stem+'|tmean_B']
            figure,axes=plt.subplots(1,3,figsize=(9,3))
            axes[0].imshow(a,cmap='gray'); axes[0].set_title(names[0])
            axes[1].imshow(b,cmap='gray'); axes[1].set_title(names[1])
            axes[2].imshow(a,cmap='gray')
            for mk in [k for k in masks.files if k.startswith(stem+'|') and k.endswith('|mask')]:
                axes[2].contour(masks[mk],levels=[.5],linewidths=.8)
            axes[2].set_title('All retained analysis regions')
            for ax in axes: ax.axis('off')
            save(figure,args.out,'QC_'+stem)
    with np.load(args.results/'kymographs.npz',allow_pickle=False) as archive:
        bases=sorted({k.rsplit('|',1)[0] for k in archive.files if k.endswith('|s_um')})
        for number,base in enumerate(bases):
            s=archive[base+'|s_um']; edges=time_edges(archive[base+'|times_s'])
            if len(s) < 2:
                continue
            spatial_edges=np.r_[s[0]-(s[1]-s[0])/2,(s[:-1]+s[1:])/2,
                                s[-1]+(s[-1]-s[-2])/2]
            figure,axes=plt.subplots(1,2,figsize=(8,3.5))
            for ax,name in zip(axes,names):
                image=archive[base+'|'+name]
                ax.pcolormesh(spatial_edges,edges,image,shading='flat',cmap='magma')
                ax.invert_yaxis(); ax.set_title(name); ax.set_xlabel('Distance along path (um)')
                ax.set_ylabel('Actual time (min)')
            figure.suptitle(base+'; contrast independent per channel',fontsize=9)
            save(figure,args.out,f'kymograph_{number:03d}')
    for number,((file,structure),group) in enumerate(cf.groupby(['file','structure'])):
        group=group.sort_values('frame')
        figure,ax=plt.subplots(figsize=(6,3.5))
        ax.plot(group.time_min,group.matched_fraction,'o-',color='#0072B2',label='Observed pairs / min(counts)')
        ax.plot(group.time_min,group.null_mean,'--',color='#666666',label='Uniform-mask null mean')
        ax.fill_between(group.time_min,group.null_p2_5,group.null_p97_5,color='#999999',alpha=.25,
                        label='Null simulation quantiles')
        ax.set_ylim(0,1.02); ax.set_xlabel('Time (min)'); ax.set_ylabel('Matched fraction')
        ax.set_title(file+' / '+structure); ax.legend(fontsize=7)
        save(figure,args.out,f'colocalisation_{number:03d}')
    if not tracks.empty:
        for number,((file,structure),group) in enumerate(tracks.groupby(['file','structure'])):
            figure,ax=plt.subplots(figsize=(6,3.5))
            for label,subset in group.groupby('channel'):
                values=np.sort(subset.persistence_min.to_numpy())
                ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label)
            ax.set_xlabel('Observed first-to-last detection span (min)')
            ax.set_ylabel('Empirical cumulative fraction'); ax.set_title(file+' / '+structure)
            ax.legend(); figure.suptitle('Descriptive spans; boundary tracks are censored',fontsize=9)
            save(figure,args.out,f'persistence_{number:03d}')
    if not displacement.empty:
        for number,((file,structure),group) in enumerate(displacement.groupby(['file','structure'])):
            figure,ax=plt.subplots(figsize=(5,4))
            ax.scatter(group.disp_A_um,group.disp_B_um,s=10,alpha=.5)
            ax.set_xlabel(names[0]+' profile shift (um)'); ax.set_ylabel(names[1]+' profile shift (um)')
            ax.set_title(file+' / '+structure)
            figure.suptitle('Profile shift does not prove cargo transport',fontsize=9)
            save(figure,args.out,f'displacement_{number:03d}')
    print('Wrote descriptive figures; recordings/frames are not independent biological replicates.')

if __name__=='__main__':
    main()
