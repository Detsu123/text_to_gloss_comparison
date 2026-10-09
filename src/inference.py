import argparse
import torch
from .models.seq2seq import build_model
from .data.vocabulary import Vocabulary
from .utils import load_config,setup

def restore(path,device):
    c=torch.load(path,map_location=device,weights_only=True)
    src,tgt=Vocabulary(c['src_vocab']),Vocabulary(c['tgt_vocab'])
    m=build_model(c['config']['model'],len(src),len(tgt),c['config']).to(device)
    m.load_state_dict(c['state_dict']); m.eval()
    return m,src,tgt,c

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='transformer'); p.add_argument('--config',default='config/default.yaml'); p.add_argument('--checkpoint'); p.add_argument('--text',required=True)
    a=p.parse_args(); cfg=load_config(a.config); device=setup(cfg)
    path=a.checkpoint or f"{cfg['output']}/{a.model}_seed{cfg['seed']}/best.pt"
    m,src,tgt,c=restore(path,device); x=torch.tensor([src.encode(a.text)],device=device)
    print(tgt.decode(m.generate(x,c['config']['max_decode_len'])[0].tolist()))
