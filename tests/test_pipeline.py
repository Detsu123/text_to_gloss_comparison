import io,json
import pytest
import torch
from torch.nn import functional as F
from src.models.seq2seq import build_model
from src.data.vocabulary import Vocabulary,normalize,PAD,SOS,EOS
from src.metrics import score

torch.set_num_threads(2)
CFG=dict(embedding_size=32,hidden_size=32,recurrent_layers=1,transformer_layers=1,n_heads=4,feedforward_dim=64,dropout=0.)

@pytest.mark.parametrize('kind',['rnn','gru','lstm','transformer'])
def test_shapes_backward_padding_causality_checkpoint(kind):
    torch.manual_seed(42); m=build_model(kind,15,12,CFG)
    x=torch.tensor([[4,5,3,0],[6,7,8,3]]); y=torch.tensor([[2,4,5],[2,6,7]])
    z=m(x,y); assert z.shape==(2,3,12)
    loss=F.cross_entropy(z.reshape(-1,12),torch.tensor([4,5,3,6,7,3])); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval(); a=m(x,y); xp=F.pad(x,(0,3)); b=m(xp,y)
    assert torch.allclose(a,b,atol=1e-5)
    yy=y.clone(); yy[:,2]=9
    assert torch.allclose(a[:,:2],m(x,yy)[:,:2],atol=1e-5)
    buffer=io.BytesIO(); torch.save(m.state_dict(),buffer); buffer.seek(0)
    restored=build_model(kind,15,12,CFG); restored.load_state_dict(torch.load(buffer,weights_only=True)); restored.eval()
    assert torch.equal(m.generate(x,10),restored.generate(x,10))
    # Force EOS and ensure decoding halts immediately, never pads or SOS.
    with torch.no_grad(): m.output.weight.zero_(); m.output.bias.fill_(-20); m.output.bias[EOS]=20
    assert m.generate(x,10).shape==(2,1)

@pytest.mark.parametrize('kind',['rnn','gru','lstm','transformer'])
def test_tiny_overfit(kind):
    torch.manual_seed(42); m=build_model(kind,12,12,CFG)
    x=torch.tensor([[4,5,3],[6,7,3]]); y=torch.tensor([[2,5,4,3],[2,7,6,3]])
    opt=torch.optim.Adam(m.parameters(),lr=.02); losses=[]
    for _ in range(140):
        opt.zero_grad(); logits=m(x,y[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,12),y[:,1:].reshape(-1)); loss.backward(); opt.step(); losses.append(loss.item())
    m.eval(); assert losses[-1]<.05 and losses[-1]<losses[0]/20
    assert torch.equal(m.generate(x,5),y[:,1:])

def test_metrics():
    refs=['A B C D','X Y Z W']; metrics,_=score(refs,refs)
    assert metrics['bleu4']==pytest.approx(100); assert metrics['chrf']==pytest.approx(100)
    assert metrics['ter']==0 and metrics['token_error_rate']==0
    bad,_=score(['',''],refs); assert bad['bleu4']==0 and bad['token_error_rate']==100
    # Cumulative BLEU-2 is NOT isolated bigram precision.
    from sacrebleu.metrics import BLEU
    h=['A B X D']; r=['A B C D']; actual,_=score(h,r)
    direct=BLEU(tokenize='none',max_ngram_order=2).corpus_score(h,[r])
    assert actual['bleu2']==pytest.approx(direct.score)

def test_vocab():
    v=Vocabulary.build(['Äpfel heute','heute sonnig'])
    assert v.encode('UNSEEN')[0]==1
    assert normalize(' A\u0308pfel\t heute ')== 'Äpfel heute'
    assert v.decode(v.encode('Äpfel heute',True))=='Äpfel heute'

def test_scheduled_sampling():
    for kind in ('rnn','gru','lstm'):
        m=build_model(kind,12,12,CFG); x=torch.tensor([[4,5,3]]); y=torch.tensor([[2,5,4]])
        for ratio in (0.,.5):
            z=m(x,y,ratio); assert z.shape==(1,3,12); z.sum().backward()

def test_preprocessing_train_only_and_overlap(tmp_path):
    import csv
    from src.data.preprocess import prepare
    raw=tmp_path/'raw'; raw.mkdir()
    samples={'train':[('t1','a b','A B'),('t2','a c','A C')],
             'dev':[('d1','a unseen','A NEW')], 'test':[('e1','a b','A B')]}
    for split,rows in samples.items():
        with (raw/f'PHOENIX-2014-T.{split}.corpus.csv').open('w',newline='') as f:
            w=csv.writer(f,delimiter='|'); w.writerow(['name','translation','orth']); w.writerows(rows)
    out=tmp_path/'processed'; stats=prepare(raw,out,True)
    assert 'unseen' not in json.loads((out/'src_vocab.json').read_text())
    assert 'NEW' not in json.loads((out/'tgt_vocab.json').read_text())
    assert stats['cross_split_overlap']['train_test']['exact_pair_rows']==1
    assert stats['splits']['dev']['text']['oov_tokens']==1
    with pytest.raises(ValueError,match='expected 7096'): prepare(raw,tmp_path/'invalid',False)
    (raw/'PHOENIX-2014-T.test.corpus.csv').write_text('name|translation|orth\nt1|a b|A B\n')
    with pytest.raises(ValueError,match='Overlapping IDs'): prepare(raw,tmp_path/'overlap',True)

@pytest.mark.parametrize('kind',['rnn','gru','lstm','transformer'])
def test_end_to_end_real_cli_components(tmp_path,kind):
    import csv
    import pandas as pd
    from src.data.preprocess import prepare
    from src.train import train
    from src.evaluate import evaluate
    from src.utils import load_config
    raw=tmp_path/'raw'; raw.mkdir()
    for split,n in [('train',7),('dev',3),('test',3)]:
        with (raw/f'PHOENIX-2014-T.{split}.corpus.csv').open('w',newline='') as f:
            w=csv.writer(f,delimiter='|'); w.writerow(['name','translation','orth'])
            for i in range(n):
                tokens=[f'x{(i+j)%4}' for j in range(2+i%3)]
                w.writerow([f'{split}_{i}',' '.join(tokens),' '.join(t.upper() for t in reversed(tokens))])
    processed=tmp_path/'data'; prepare(raw,processed,True)
    cfg=load_config('config/cpu_pilot.yaml'); cfg.update(CFG)
    cfg.update(processed=str(processed),output=str(tmp_path/'run'),max_epochs=1,patience=1,batch_size=2,grad_accumulation=2,max_decode_len=8,device='cpu')
    summary=train(kind,cfg); assert summary['epochs_run']==1
    evaluate(cfg,[kind])
    result=pd.read_csv(tmp_path/'run/results/metrics.csv')
    assert result.iloc[0]['n']==3 and result.iloc[0]['dataset']=='SYNTHETIC_PIPELINE_TEST'
    assert (tmp_path/f'run/{kind}_seed42/best.pt').exists()
    with pytest.raises(FileExistsError): train(kind,cfg)
