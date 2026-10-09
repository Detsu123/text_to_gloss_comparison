"""Strict official CSV import. Test content is never used to fit preprocessing."""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
from .vocabulary import Vocabulary, normalize, tokenize

EXPECTED={'train':7096,'dev':519,'test':642}

def prepare(raw='data/raw', output='data/processed/phoenix14t', synthetic=False):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    splits={}; manifest={}
    for split in EXPECTED:
        name=f'PHOENIX-2014-T.{split}.corpus.csv'
        matches=list(Path(raw).rglob(name))
        if len(matches)!=1:
            raise FileNotFoundError(f'Need exactly one {raw}/**/{name}; found {len(matches)}. See data/README.md.')
        p=matches[0]
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f,delimiter='|')
            if not {'name','translation','orth'} <= set(reader.fieldnames or []):
                raise ValueError(f'{p}: expected name|...|orth|translation; got {reader.fieldnames}')
            rows=[{'id':normalize(r['name'] or ''), 'text':normalize(r['translation'] or ''),
                   'gloss':normalize(r['orth'] or ''), 'split':split} for r in reader]
        if not synthetic and len(rows)!=EXPECTED[split]:
            raise ValueError(f'{split}: expected {EXPECTED[split]} rows, got {len(rows)}')
        if any(not r[k] for r in rows for k in ('id','text','gloss')):
            raise ValueError(f'{split}: missing required value; repair upstream, never silently drop')
        if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate IDs in '+split)
        splits[split]=rows
        manifest[split]={'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'columns':reader.fieldnames}
    # IDs may be identical text on different broadcasts, but may not overlap splits.
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if {r['id'] for r in splits[a]} & {r['id'] for r in splits[b]}:
            raise ValueError(f'Overlapping IDs: {a}/{b}')
    src=Vocabulary.build(r['text'] for r in splits['train'])
    tgt=Vocabulary.build(r['gloss'] for r in splits['train'])
    stats={'dataset':'SYNTHETIC_PIPELINE_TEST' if synthetic else 'PHOENIX14T',
           'vocab_fit_split':'train','vocab_including_special':{'source':len(src),'target':len(tgt)},
           'vocab_excluding_special':{'source':len(src)-4,'target':len(tgt)-4},'splits':{},'cross_split_overlap':{}}
    train_pairs={(r['text'],r['gloss']) for r in splits['train']}
    train_texts={r['text'] for r in splits['train']}
    train_glosses={r['gloss'] for r in splits['train']}
    for split, rows in splits.items():
        st={'samples':len(rows),'missing_values':0,'duplicate_pairs':len(rows)-len({(r['text'],r['gloss']) for r in rows})}
        for key,v in [('text',src),('gloss',tgt)]:
            tokens=[tokenize(r[key]) for r in rows]; lens=[len(t) for t in tokens]
            counts=Counter(t for seq in tokens for t in seq); unknown={t:n for t,n in counts.items() if t not in v.stoi}
            st[key]={'mean_length':float(np.mean(lens)), 'max_length':max(lens), 'min_length':min(lens),
                     'tokens':sum(lens),'vocabulary':len(counts),'oov_types':len(unknown),
                     'oov_tokens':sum(unknown.values()),'oov_rate':sum(unknown.values())/sum(lens)}
        stats['splits'][split]=st
        if split!='train':
            stats['cross_split_overlap'][f'train_{split}']={
                'exact_pair_rows':sum((r['text'],r['gloss']) in train_pairs for r in rows),
                'source_rows':sum(r['text'] in train_texts for r in rows),
                'target_rows':sum(r['gloss'] in train_glosses for r in rows)}
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in rows:
                if split!='train': r['seen_train_pair']=(r['text'],r['gloss']) in train_pairs
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
    lens=[len(tokenize(r['text'])) for r in splits['train']]
    bounds=np.quantile(lens,[1/3,2/3],method='nearest').astype(int).tolist()
    prep={'normalization':'NFC + whitespace', 'tokenizer':'case-sensitive whitespace',
          'length_bins_train_tertiles':bounds,'length_bin_rule':'short <= q1; medium q1 < n <= q2; long > q2',
          'split_policy':'official unchanged; duplicates audited; unseen-pair sensitivity reported',
          'dataset':stats['dataset'],'raw_manifest':manifest}
    for name,obj in [('src_vocab',src.tokens),('tgt_vocab',tgt.tokens),('statistics',stats),('preprocessing',prep)]:
        (out/f'{name}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'dataset':stats['dataset'],'samples':{s:len(r) for s,r in splits.items()},'vocab':stats['vocab_excluding_special'],'length_bins':bounds},indent=2))
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--raw',default='data/raw'); p.add_argument('--output',default='data/processed/phoenix14t'); p.add_argument('--synthetic',action='store_true')
    a=p.parse_args(); prepare(a.raw,a.output,a.synthetic)
