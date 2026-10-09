import argparse,json
from pathlib import Path
from collections import Counter
import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein
from .inference import restore
from .data.dataset import loader
from .data.vocabulary import tokenize,EOS
from .metrics import score
from .utils import load_config,setup,dump,fingerprint

@torch.no_grad()
def evaluate(cfg,models,split='test',seeds=None):
    device=setup(cfg); base=Path(cfg['output']); results=base/'results'; results.mkdir(parents=True,exist_ok=True)
    prep=json.loads((Path(cfg['processed'])/'preprocessing.json').read_text()); q1,q2=prep['length_bins_train_tertiles']
    metrics_rows=[]; length_rows=[]; prediction_tables={}; signatures={}; sensitivity=[]; counts=[]
    for seed in seeds or [cfg['seed']]:
        for kind in models:
            folder=base/f'{kind}_seed{seed}'; m,src,tgt,c=restore(folder/'best.pt',device)
            if c['train_dev_fingerprint']!=fingerprint(cfg['processed']): raise ValueError('Checkpoint/data mismatch')
            pred=[]
            for x,_,rows in loader(cfg['processed'],split,cfg):
                generated=m.generate(x.to(device),c['config']['max_decode_len']).tolist()
                for r,ids in zip(rows,generated):
                    hyp=tgt.decode(ids); ref=r['gloss']; n=len(tokenize(r['text'])); rt,ht=ref.split(),hyp.split()
                    ops=Counter(op.tag for op in Levenshtein.editops(rt,ht))
                    same_bag=Counter(rt)==Counter(ht) and rt!=ht
                    pred.append({**r,'prediction':hyp,'source_length':n,'length_group':'short' if n<=q1 else 'medium' if n<=q2 else 'long',
                         'ended_eos':EOS in ids,'hit_max_length':EOS not in ids,'source_oov_tokens':sum(t not in src.stoi for t in tokenize(r['text'])),
                         'deletions':ops['delete'],'insertions':ops['insert'],'substitutions':ops['replace'],
                         'edit_distance':Levenshtein.distance(rt,ht),'order_only_candidate':same_bag,
                         'repetition_candidate':any(ht[i]==ht[i-1] for i in range(1,len(ht))),
                         'short_output_candidate':len(ht)<0.5*len(rt),
                         'unknown_output':'<UNK>' in ht})
            df=pd.DataFrame(pred); key=f'{kind}_seed{seed}'; prediction_tables[key]=df
            (results/'predictions').mkdir(exist_ok=True); df.to_csv(results/'predictions'/f'{key}_{split}.csv',index=False)
            scores,sigs=score(df.prediction.tolist(),df.gloss.tolist()); signatures[key]=sigs
            summary=json.loads((folder/'summary.json').read_text()); metrics_rows.append({**summary,**scores,'split':split,'n':len(df),'decode':'greedy','eos_rate':float(df.ended_eos.mean())})
            for group in ('short','medium','long'):
                d=df[df.length_group==group]
                met=score(d.prediction.tolist(),d.gloss.tolist())[0] if len(d) else {}
                length_rows.append({'model':kind,'seed':seed,'split':split,'group':group,'n':len(d),'q1':q1,'q2':q2,**met})
            unseen=df[~df.seen_train_pair] if 'seen_train_pair' in df else df
            if len(unseen): sensitivity.append({'model':kind,'seed':seed,'subset':'unseen_train_pairs','n':len(unseen),**score(unseen.prediction.tolist(),unseen.gloss.tolist())[0]})
            counts.append({'model':kind,'seed':seed,**{k:int(df[k].sum()) for k in ['deletions','insertions','substitutions','order_only_candidate','repetition_candidate','short_output_candidate','unknown_output','hit_max_length']}})
            del m
            if device.type=='cuda': torch.cuda.empty_cache()
    # Filenames include split; metrics.csv is the requested test comparison only.
    suffix='' if split=='test' else '_'+split
    pd.DataFrame(metrics_rows).to_csv(results/f'metrics{suffix}.csv',index=False)
    pd.DataFrame(length_rows).to_csv(results/f'length_analysis{suffix}.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(results/f'unseen_pair_sensitivity{suffix}.csv',index=False)
    pd.DataFrame(counts).to_csv(results/f'error_counts{suffix}.csv',index=False)
    dump(results/f'metric_signatures{suffix}.json',signatures)
    # Representative selection happens AFTER frozen evaluation, never for tuning.
    first=next(iter(prediction_tables.values())); merged=first[['id','text','gloss','source_length','length_group']].copy()
    errors=[]
    for key,df in prediction_tables.items():
        merged[key]=df.prediction.values; errors.append(df.edit_distance.to_numpy())
    merged['mean_edit_distance']=sum(errors)/len(errors)
    cols=list(prediction_tables); merged['model_disagreement']=merged[cols].nunique(axis=1)
    selected=[]
    for label,subset,sort_key,ascending in [('easy',merged,'mean_edit_distance',True),('medium',merged[merged.length_group=='medium'],'mean_edit_distance',True),('long',merged[merged.length_group=='long'],'source_length',False),('disagreement',merged,'model_disagreement',False),('failure',merged,'mean_edit_distance',False)]:
        if len(subset):
            r=subset.sort_values(sort_key,ascending=ascending).iloc[0].to_dict(); r['selection_reason']=label; r['human_verified_category']=''; selected.append(r)
    pd.DataFrame(selected).to_csv(results/f'error_analysis{suffix}.csv',index=False)
    all_metrics=pd.DataFrame(metrics_rows)
    if all_metrics.seed.nunique()>1:
        all_metrics.groupby('model')[['bleu1','bleu2','bleu3','bleu4','chrf','ter']].agg(['mean','std','count']).to_csv(results/f'seed_summary{suffix}.csv')
    print(all_metrics[['model','seed','bleu4','chrf','ter','n']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config/default.yaml'); p.add_argument('--model',default='all',choices=['all','rnn','gru','lstm','transformer']); p.add_argument('--split',default='test',choices=['dev','test']); p.add_argument('--seeds',nargs='+',type=int)
    a=p.parse_args(); evaluate(load_config(a.config),['rnn','gru','lstm','transformer'] if a.model=='all' else [a.model],a.split,a.seeds)
