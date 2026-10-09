import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .vocabulary import Vocabulary, PAD

class GlossDataset(Dataset):
    def __init__(self, rows, src, tgt): self.rows,self.src,self.tgt=rows,src,tgt
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return torch.tensor(self.src.encode(r['text'])),torch.tensor(self.tgt.encode(r['gloss'],target=True)),r

def collate(batch):
    x,y,rows=zip(*batch)
    return pad_sequence(x,batch_first=True,padding_value=PAD),pad_sequence(y,batch_first=True,padding_value=PAD),list(rows)

def load_vocab(folder):
    p=Path(folder)
    return tuple(Vocabulary(json.loads((p/f'{side}_vocab.json').read_text(encoding='utf-8'))) for side in ('src','tgt'))

def load_rows(folder,split):
    return [json.loads(line) for line in (Path(folder)/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]

def loader(folder,split,cfg,shuffle=False):
    src,tgt=load_vocab(folder)
    return DataLoader(GlossDataset(load_rows(folder,split),src,tgt),batch_size=cfg['batch_size'],shuffle=shuffle,
                      collate_fn=collate,num_workers=cfg['num_workers'],generator=torch.Generator().manual_seed(cfg['seed']))
