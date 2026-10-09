"""Download only PHOENIX14T CSV annotations from a pinned public mirror."""
import argparse,urllib.request,json,hashlib
from pathlib import Path
COMMIT='ebb29ba996f748f5775b0a95a515d49b729e2ddb'
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='data/raw'); args=parser.parse_args()
    p=Path(args.output); p.mkdir(parents=True,exist_ok=True); records=[]
    for split in ('train','dev','test'):
        name=f'PHOENIX-2014-T.{split}.corpus.csv'
        url=f'https://huggingface.co/datasets/aipieces/RWTH-PHOENIX-Weather-2014T/resolve/{COMMIT}/annotations/manual/{name}'
        payload=urllib.request.urlopen(url,timeout=60).read()
        if not payload.startswith(b'name|video|start|end|speaker|orth|translation'): raise ValueError('Unexpected annotation header')
        (p/name).write_bytes(payload); records.append({'file':name,'url':url,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)})
    (p/'download_provenance.json').write_text(json.dumps({'source':'PHOENIX14T public mirror','commit':COMMIT,'files':records},indent=2))
    print('Downloaded three annotation files. Follow dataset attribution and noncommercial terms in data/README.md.')
if __name__=='__main__': main()
