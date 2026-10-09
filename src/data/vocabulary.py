import unicodedata
from collections import Counter

SPECIAL = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
PAD, UNK, SOS, EOS = range(4)

def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', str(text)).split())

def tokenize(text):
    # Dataset text is already whitespace tokenized. Preserve case, punctuation,
    # umlauts, numbers and gloss compound markers; no linguistic rewriting.
    return normalize(text).split()

class Vocabulary:
    def __init__(self, tokens):
        self.tokens = tokens
        self.stoi = {t:i for i,t in enumerate(tokens)}
        if tokens[:4] != SPECIAL or len(self.stoi) != len(tokens):
            raise ValueError('Invalid vocabulary')
    @classmethod
    def build(cls, sentences):
        counts = Counter(t for s in sentences for t in tokenize(s))
        if any(t in counts for t in SPECIAL):
            raise ValueError('Raw data contains reserved special tokens')
        return cls(SPECIAL + sorted(counts, key=lambda t:(-counts[t], t)))
    def encode(self, text, target=False):
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)]
        return ([SOS] if target else []) + ids + [EOS]
    def decode(self, ids):
        out=[]
        for i in ids:
            i=int(i)
            if i==EOS: break
            if i not in (PAD,SOS): out.append(self.tokens[i])
        return ' '.join(out)
    def __len__(self): return len(self.tokens)
