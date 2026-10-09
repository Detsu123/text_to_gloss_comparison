"""Measure training + greedy-decode CUDA peak for one batch; not a full-run peak."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from src.utils import load_config,setup
from src.models.seq2seq import build_model
from src.data.dataset import load_vocab,loader

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config/default.yaml'); a=p.parse_args()
    cfg=load_config(a.config); cfg['device']='cuda'
    if not torch.cuda.is_available(): raise SystemExit('No CUDA GPU available; no VRAM measurement made.')
    device=setup(cfg); src,tgt=load_vocab(cfg['processed']); records=[]
    x,y,_=next(iter(loader(cfg['processed'],'train',cfg))); x,y=x.to(device),y.to(device)
    for kind in ['rnn','gru','lstm','transformer']:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m=build_model(kind,len(src),len(tgt),cfg).to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate']); scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'])
        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
            z=m(x,y[:,:-1]); loss=F.cross_entropy(z.reshape(-1,len(tgt)),y[:,1:].reshape(-1),ignore_index=0)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); m.eval(); m.generate(x,cfg['max_decode_len']); torch.cuda.synchronize()
        records.append({'model':kind,'batch':len(x),'source_length':x.size(1),'target_length':y.size(1),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
        del m,opt,z,loss,scaler
    Path('results').mkdir(exist_ok=True); Path('results/gpu_memory_probe.json').write_text(json.dumps(records,indent=2)); print(json.dumps(records,indent=2))
if __name__=='__main__': main()
