from sacrebleu.metrics import BLEU,CHRF,TER
from rapidfuzz.distance import Levenshtein

def score(hypotheses,references):
    if len(hypotheses)!=len(references) or not references: raise ValueError('Aligned nonempty corpus required')
    scores={}; signatures={}
    for n in range(1,5):
        metric=BLEU(max_ngram_order=n,tokenize='none',lowercase=False,smooth_method='exp',effective_order=False)
        scores[f'bleu{n}']=metric.corpus_score(hypotheses,[references]).score
        signatures[f'bleu{n}']=str(metric.get_signature())+f'|max_order:{n}'
    for name,metric in [('chrf',CHRF(char_order=6,word_order=0,beta=2)),
                         ('ter',TER(normalized=False,no_punct=False,asian_support=False,case_sensitive=True))]:
        scores[name]=metric.corpus_score(hypotheses,[references]).score; signatures[name]=str(metric.get_signature())
    scores['exact_match']=100*sum(h==r for h,r in zip(hypotheses,references))/len(references)
    total=sum(len(r.split()) for r in references)
    scores['token_error_rate']=100*sum(Levenshtein.distance(r.split(),h.split()) for h,r in zip(hypotheses,references))/total
    return scores,signatures
