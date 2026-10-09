import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from src.data.vocabulary import PAD,SOS,EOS

class Attention(nn.Module):
    """Luong general attention; mask padded encoder positions."""
    def __init__(self,hidden): super().__init__(); self.key=nn.Linear(hidden,hidden,bias=False)
    def forward(self,query,memory,pad_mask):
        scores=torch.bmm(query,self.key(memory).transpose(1,2))
        scores=scores.masked_fill(pad_mask[:,None,:],float('-inf'))
        weights=scores.softmax(-1)
        return torch.bmm(weights,memory)

class RecurrentSeq2Seq(nn.Module):
    def __init__(self,kind,n_src,n_tgt,cfg):
        super().__init__(); e,h=cfg['embedding_size'],cfg['hidden_size']; layers=cfg['recurrent_layers']; d=cfg['dropout']
        self.source=nn.Embedding(n_src,e,padding_idx=PAD); self.target=nn.Embedding(n_tgt,e,padding_idx=PAD)
        cls={'rnn':nn.RNN,'gru':nn.GRU,'lstm':nn.LSTM}[kind]
        self.encoder=cls(e,h,layers,batch_first=True,dropout=d if layers>1 else 0)
        self.decoder=cls(e,h,layers,batch_first=True,dropout=d if layers>1 else 0)
        self.attention=Attention(h); self.drop=nn.Dropout(d); self.fuse=nn.Linear(2*h,h); self.output=nn.Linear(h,n_tgt)
    def encode(self,x):
        mask=x.eq(PAD); lens=(~mask).sum(1).cpu()
        packed=pack_padded_sequence(self.drop(self.source(x)),lens,batch_first=True,enforce_sorted=False)
        encoded,state=self.encoder(packed)
        memory,_=pad_packed_sequence(encoded,batch_first=True,total_length=x.size(1))
        return memory,state,mask
    def decode(self,y,state,memory,mask):
        q,state=self.decoder(self.drop(self.target(y)),state)
        ctx=self.attention(q,memory,mask)
        logits=self.output(self.drop(torch.tanh(self.fuse(torch.cat([q,ctx],-1)))))
        return logits,state
    def forward(self,x,y,teacher_forcing=1.0):
        memory,state,mask=self.encode(x)
        if teacher_forcing==1: return self.decode(y,state,memory,mask)[0]
        outputs=[]; token=y[:,0:1]
        for t in range(y.size(1)):
            logits,state=self.decode(token,state,memory,mask); outputs.append(logits)
            if t+1<y.size(1):
                choose=torch.rand(x.size(0),1,device=x.device)<teacher_forcing
                token=torch.where(choose,y[:,t+1:t+2],logits.argmax(-1))
        return torch.cat(outputs,1)
    @torch.no_grad()
    def generate(self,x,max_len):
        memory,state,mask=self.encode(x); token=torch.full((len(x),1),SOS,device=x.device,dtype=torch.long)
        done=torch.zeros(len(x),device=x.device,dtype=torch.bool); generated=[]
        for _ in range(max_len):
            logits,state=self.decode(token,state,memory,mask)
            logits[:,:,PAD]=float('-inf'); logits[:,:,SOS]=float('-inf')
            token=logits.argmax(-1); token=torch.where(done[:,None],torch.full_like(token,EOS),token)
            generated.append(token); done|=token[:,0].eq(EOS)
            if done.all(): break
        return torch.cat(generated,1)

class Position(nn.Module):
    def __init__(self,d,dropout):
        super().__init__(); self.d=d; self.drop=nn.Dropout(dropout)
    def forward(self,x):
        n=x.size(1); pos=torch.arange(n,device=x.device).float()[:,None]
        scale=torch.exp(torch.arange(0,self.d,2,device=x.device).float()*(-math.log(10000)/self.d))
        pe=torch.zeros(n,self.d,device=x.device); pe[:,0::2]=torch.sin(pos*scale); pe[:,1::2]=torch.cos(pos*scale)
        return self.drop(x*math.sqrt(self.d)+pe.to(x.dtype))

class TransformerSeq2Seq(nn.Module):
    def __init__(self,n_src,n_tgt,cfg):
        super().__init__(); d=cfg['embedding_size']
        self.source=nn.Embedding(n_src,d,padding_idx=PAD); self.target=nn.Embedding(n_tgt,d,padding_idx=PAD)
        self.position=Position(d,cfg['dropout'])
        self.net=nn.Transformer(d_model=d,nhead=cfg['n_heads'],num_encoder_layers=cfg['transformer_layers'],
             num_decoder_layers=cfg['transformer_layers'],dim_feedforward=cfg['feedforward_dim'],dropout=cfg['dropout'],batch_first=True)
        self.output=nn.Linear(d,n_tgt)
    def decode(self,y,memory,src_mask):
        causal=torch.ones(y.size(1),y.size(1),device=y.device,dtype=torch.bool).triu(1)
        z=self.net.decoder(self.position(self.target(y)),memory,tgt_mask=causal,
             tgt_key_padding_mask=y.eq(PAD),memory_key_padding_mask=src_mask)
        return self.output(z)
    def forward(self,x,y,teacher_forcing=1.0):
        memory=self.net.encoder(self.position(self.source(x)),src_key_padding_mask=x.eq(PAD))
        return self.decode(y,memory,x.eq(PAD))
    @torch.no_grad()
    def generate(self,x,max_len):
        mask=x.eq(PAD); memory=self.net.encoder(self.position(self.source(x)),src_key_padding_mask=mask)
        y=torch.full((len(x),1),SOS,device=x.device,dtype=torch.long); done=torch.zeros(len(x),device=x.device,dtype=torch.bool)
        for _ in range(max_len):
            logits=self.decode(y,memory,mask)[:,-1]; logits[:,PAD]=float('-inf'); logits[:,SOS]=float('-inf')
            next_id=logits.argmax(-1); next_id=torch.where(done,torch.full_like(next_id,EOS),next_id)
            y=torch.cat([y,next_id[:,None]],1); done|=next_id.eq(EOS)
            if done.all(): break
        return y[:,1:]

def build_model(kind,n_src,n_tgt,cfg):
    if kind=='transformer': return TransformerSeq2Seq(n_src,n_tgt,cfg)
    return RecurrentSeq2Seq(kind,n_src,n_tgt,cfg)
