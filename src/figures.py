import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utils import load_config

MODELS=['rnn','gru','lstm','transformer']; LABELS=['RNN','GRU','LSTM','Transformer']
COLORS=['#355C8C','#B07B22','#708238','#AD567B']

def figures(cfg):
    base=Path(cfg['output']); out=base/'figures'; out.mkdir(exist_ok=True)
    results=base/'results'; df=pd.read_csv(results/'metrics.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    scope=f"PHOENIX14T | {cfg['experiment']} | {df.seed.nunique()} seeds"
    def save(fig,name,note=None):
        notes={'loss_curves':'Train / dev; each line is one seed. Training includes dropout; validation disables it.',
               'validation_bleu4':'Development set; greedy decoding; each line is one seed.',
               'parameters':'Exact trainable parameter counts; identical across the three seeds.',
               'training_wall_seconds':'Concurrent shared-host runs; mean ± 1 sample SD. Not a controlled speed benchmark.',
               'length_bleu4':'Test groups; mean ± 1 sample SD across seeds; cutoffs fixed from training.'}
        caption=note or notes.get(name,'Test n=642; greedy decoding; bars show mean ± 1 sample SD across seeds.')
        fig.text(.01,.038,scope,fontsize=8,color='#555555')
        fig.text(.01,.008,caption,fontsize=7.5,color='#555555'); fig.tight_layout(rect=(0,.08,1,1))
        for ext in ('png','pdf','svg'): fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    for metric,title in [('bleu4','BLEU-4 (higher is better)'),('chrf','chrF (higher is better)'),('ter','TER (lower is better)'),('parameters','Trainable parameters'),('training_wall_seconds','Training + validation wall time (s; shared CPU host)')]:
        fig,ax=plt.subplots(figsize=(7,4)); stats=df.groupby('model')[metric].agg(['mean','std']).reindex(MODELS)
        bars=ax.bar(LABELS,stats['mean'],color=COLORS,width=.6)
        if df.seed.nunique()>1: ax.errorbar(LABELS,stats['mean'],yerr=stats['std'],fmt='none',ecolor='#333333',capsize=4)
        for bar,mean,sd in zip(bars,stats['mean'],stats['std'].fillna(0)):
            ax.text(bar.get_x()+bar.get_width()/2,mean+sd+max(stats['mean'])*.015,f'{mean:,.0f}' if metric=='parameters' else f'{mean:.2f}',ha='center',va='bottom')
        ax.set_ylim(0,max(stats['mean']+stats['std'].fillna(0))*1.2 if stats['mean'].max()>0 else 1); ax.set_title(title); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        save(fig,metric)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    fig2,ax2=plt.subplots(figsize=(7,4))
    for i,kind in enumerate(MODELS):
        for j,path in enumerate(sorted(base.glob(kind+'_seed*/history.csv'))):
            h=pd.read_csv(path); ax=axes.flat[i]
            ax.plot(h.epoch,h.train_loss,color=COLORS[i],label='Train' if j==0 else None)
            ax.plot(h.epoch,h.dev_loss,color=COLORS[i],linestyle='--',label='Dev' if j==0 else None)
            ax2.plot(h.epoch,h.dev_bleu4,color=COLORS[i],marker=['o','s','^','D'][i],markersize=3,label=LABELS[i] if j==0 else None,alpha=.85)
        ax.set_title(LABELS[i]); ax.set_xlabel('Epoch'); ax.set_ylabel('Token cross entropy'); ax.legend(); ax.grid(alpha=.2)
    save(fig,'loss_curves'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dev BLEU-4'); ax2.legend(ncol=2); ax2.grid(alpha=.2); save(fig2,'validation_bleu4')
    ld=pd.read_csv(results/'length_analysis.csv'); fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(3); width=.19
    for i,kind in enumerate(MODELS):
        d=ld[ld.model==kind].groupby('group').bleu4.agg(['mean','std']).reindex(['short','medium','long'])
        ax.bar(x+(i-1.5)*width,d['mean'],width,color=COLORS[i],label=LABELS[i])
        if df.seed.nunique()>1: ax.errorbar(x+(i-1.5)*width,d['mean'],yerr=d['std'],fmt='none',ecolor='#333',capsize=2)
    ns=ld[ld.model=='rnn'].groupby('group').n.first(); q1,q2=ld.iloc[0][['q1','q2']]
    ax.set_xticks(x,[f'Short ≤{q1}\nn={ns["short"]}',f'Medium {q1+1}–{q2}\nn={ns["medium"]}',f'Long >{q2}\nn={ns["long"]}'])
    ax.set_ylabel('Test BLEU-4'); ax.set_title('Source-length groups fixed from training tertiles'); ax.legend(ncol=4); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); save(fig,'length_bleu4')
    print('Saved figures:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config/default.yaml'); a=p.parse_args(); figures(load_config(a.config))
