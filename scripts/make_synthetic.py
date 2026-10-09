"""Mechanical synthetic sequences only; NOT German/DGS linguistic evidence."""
import csv,itertools,random
from pathlib import Path
p=Path('data/synthetic_raw'); p.mkdir(parents=True,exist_ok=True)
pairs=list(itertools.product(range(4),repeat=3)); random.Random(42).shuffle(pairs)
for split,seqs in [('train',pairs[:48]),('dev',pairs[48:56]),('test',pairs[56:])]:
    with (p/f'PHOENIX-2014-T.{split}.corpus.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['name','orth','translation'],delimiter='|'); w.writeheader()
        for i,seq in enumerate(seqs): w.writerow({'name':f'SYNTHETIC_{split}_{i}','translation':' '.join(f'x{a}' for a in seq),'orth':' '.join(f'Y{a}' for a in reversed(seq))})
print('SYNTHETIC mechanical reversal data generated; not valid DGS.')
