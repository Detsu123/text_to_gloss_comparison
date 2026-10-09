import os,json,random,hashlib,platform
from pathlib import Path
import numpy as np
import torch,yaml

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text()); return cfg

def setup(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(cfg['threads']); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True,warn_only=True)
    device=cfg['device']
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def fingerprint(folder):
    h=hashlib.sha256()
    # Test content deliberately excluded from the training artifact identity.
    for name in ('train.jsonl','dev.jsonl','src_vocab.json','tgt_vocab.json'):
        h.update((Path(folder)/name).read_bytes())
    return h.hexdigest()

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def environment(device):
    return {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
            'platform':platform.platform(),'device':str(device),'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'cpu':platform.processor(),'cpu_count':os.cpu_count()}
