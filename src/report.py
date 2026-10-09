"""Generate document-ready factual results from completed, frozen test evaluation."""
import argparse,json
from pathlib import Path
import pandas as pd
from .utils import load_config

def report(cfg):
    root=Path(cfg['output']); result=root/'results'; notes=root/'paper_notes'; notes.mkdir(exist_ok=True)
    df=pd.read_csv(result/'metrics.csv'); seeds=sorted(df.seed.unique().tolist()); models=['rnn','gru','lstm','transformer']
    metrics=['bleu1','bleu2','bleu3','bleu4','chrf','ter','exact_match','token_error_rate']
    def cell(x): return f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if len(x)>1 else f'{x.iloc[0]:.2f}'
    rows=[]
    for model in models:
        d=df[df.model==model]; rows.append({'Model':model.upper() if model!='transformer' else 'Transformer',
           **{m:cell(d[m]) for m in metrics},'Parameters':int(d.parameters.iloc[0]),
           'Wall seconds':cell(d.training_wall_seconds)})
    table=pd.DataFrame(rows); table.to_csv(result/'comparison_mean_std.csv',index=False)
    table.to_latex(result/'comparison_mean_std.tex',index=False,escape=True)
    rank=df.groupby('model').bleu4.mean().sort_values(ascending=False); winner=rank.index[0]; second=rank.index[1]
    length=pd.read_csv(result/'length_analysis.csv'); lengths=[]
    for model in models:
        d=length[length.model==model]; lengths.append({'Model':model,**{g:cell(d[d.group==g].bleu4) for g in ('short','medium','long')}})
    lengths=pd.DataFrame(lengths); lengths.to_csv(result/'length_mean_std.csv',index=False)
    counts=length.groupby('group').n.first().to_dict()
    unseen=pd.read_csv(result/'unseen_pair_sensitivity.csv'); us=unseen.groupby('model').bleu4.mean().sort_values(ascending=False)
    caps=int((df.best_epoch==cfg['max_epochs']).sum())
    is_pilot='pilot' in cfg['experiment']
    scope_note='These are **pilot results**, not results for the unexecuted 256-width RTX configuration.' if is_pilot else 'These results apply only to the completed configuration and devices recorded in the run metadata.'
    device_note=', '.join(sorted(df.device.unique()))
    future_note='The 256-width, 40-epoch configuration remains a proposed next experiment.' if is_pilot else 'Additional optimization and linguistic evaluation remain future work.'
    text=f'''# Results and Discussion — {cfg['experiment']}

## Results: experimental scope

All results below are measured on real PHOENIX14T text→DGS-gloss pairs. The official training, development and test partitions contain 7,096, 519 and 642 samples. The executed configuration uses width {cfg['embedding_size']}, up to {cfg['max_epochs']} epochs, batch {cfg['batch_size']} and device(s) {device_note} (precision settings are saved in each run config); seeds are {seeds}. Each architecture is trained independently for each seed. Development corpus BLEU-4 selects the checkpoint, and all training is complete before final test evaluation. {scope_note}

## Overall comparison

Values are mean ± sample standard deviation across {len(seeds)} seeds. Higher BLEU/chrF/exact match and lower TER/token error rate are preferred. All quality scores use a 0–100 presentation scale; error rates can exceed 100. Parameter counts are exact trainable-parameter totals. Wall time includes training, validation and checkpoint work on a shared host; concurrent seed execution prevents a controlled speed ranking.

{table.to_markdown(index=False)}

**{winner.upper()} has the highest mean test BLEU-4 in this experiment**, {rank.iloc[0]:.2f}, compared with {rank.iloc[1]:.2f} for {second.upper()} (difference {rank.iloc[0]-rank.iloc[1]:.2f} BLEU points). This is a descriptive comparison; no statistical significance test has been performed. The standard deviations measure variation from training seeds, not uncertainty from drawing a new test corpus. GPU measurements, when available, are in metrics.csv; CPU-only runs contain null GPU-memory fields.

### Checkpoint selection and duration by run

{df[['model','seed','best_epoch','epochs_run','best_dev_bleu4','training_wall_seconds','parameters']].round(3).to_markdown(index=False)}

The best checkpoint occurs at the final allowed epoch for {caps} of {len(df)} runs. Consequently, convergence or optimal attainable accuracy has not been established; the finite epoch cap may favor architectures that learn faster under this optimizer.

## Source-length analysis

The thresholds were fixed from training tertiles: short ≤11 source tokens, medium 12–16, long ≥17. Test counts are short {counts['short']}, medium {counts['medium']}, long {counts['long']}. Scores below are corpus BLEU-4 within each group, then summarized across seeds.

{lengths.to_markdown(index=False)}

The group results are descriptive. Different groups contain different sentences, vocabulary, targets and numbers of references. They do not isolate a causal effect of recurrent memory, and the lower group sample sizes make rankings less stable. No model was tuned to these test groups.

## Duplicate-pair sensitivity

Fourteen test rows have an exact normalized source/target pair present in training. The primary benchmark retains the original 642-row split. The secondary analysis evaluates the remaining 628 rows. Mean BLEU-4 by model on that subset is:

{us.rename('BLEU-4 (mean)').round(3).to_frame().to_markdown()}

The highest mean BLEU-4 on the overlap-excluded subset belongs to {us.index[0].upper()}. This sensitivity analysis handles exact text/gloss pairs only; it does not eliminate near-duplicate forecasts or shared domain vocabulary. Vocabularies, normalization and length thresholds were fitted without test data.

## Error-analysis scope

Complete predictions, edit-operation counts and representative cross-model examples are supplied in `predictions/`, `error_counts.csv` and `error_analysis.csv`. Candidate deletions/insertions/substitutions come from token edit alignment; repetition, ordering, very-short-output and missing-EOS flags are additional diagnostics. These labels are not native-sign-language judgments. See `paper_notes/qualitative_observations.md` in the project root for inspection of concrete outputs. A single reference cannot determine whether every alternative gloss sequence is linguistically invalid.

## Discussion

The observed ordering supports a conditional model comparison under a common preprocessing and optimization protocol. It does not establish that one architecture is universally superior. The recurrent cells share the same encoder/decoder/attention implementation, so their measured differences are less exposed to unrelated preprocessing or decoder changes. The Transformer differs in depth and parameter count; matched width is not matched model capacity. A common learning rate, dropout and epoch cap also need not be equally favorable to each architecture.

{'The hypothesis that Transformer necessarily wins is not supported by this pilot.' if winner!='transformer' else 'Transformer ranks first in this experiment, but the result does not establish a universal advantage.'} Claims that Transformers specifically handle long sentences better require the length-group evidence, repeated converged runs and better-controlled length comparisons. GPU efficiency cannot be inferred from CPU wall time.

The train-only vocabulary and preserved target references allow unknown-token behavior to be assessed without inflating scores by replacing reference OOVs. Preserving the original split keeps the main experiment comparable at the partition level, while the overlap-excluded sensitivity exposes one source of optimistic evaluation. Corpus/tokenizer and metric signatures must accompany any cross-paper comparison, especially because published gloss→German scores address the opposite translation direction.

## Limitations and next experiments

This is a {cfg['max_epochs']}-epoch-budget, {len(seeds)}-seed weather-domain experiment with one gloss reference per sentence and no DGS expert evaluation. There is no parameter-matched ablation, architecture-specific learning-rate search, pretrained model, dictionary baseline or beam-search comparison. A pinned public annotation mirror was used; its bytes were not independently compared with the full official archive. {future_note} CUDA/AMP validation is supported only for runs that actually used those settings.

Next: (1) run the full RTX configuration across the same three seeds; (2) use development data for a small predeclared optimization comparison; (3) compare parameter-matched variants; (4) report paired bootstrap intervals; (5) ask qualified DGS users to inspect an anonymized sample; (6) evaluate a common beam size as a separate experiment. A future Mongolian extension requires an independently annotated, consented Mongolian text/gloss corpus and expert validation, not relabeling DGS tokens.

## Conclusion

The pipeline has produced inspectable real-data model outputs and a reproducible pilot comparison. {winner.upper()} achieves the highest mean test BLEU-4 within the measured conditions. The appropriate conference claim is an empirical pilot with explicit resource and annotation limitations; algorithmic novelty, general signed-language fluency and deployment readiness are not demonstrated.
'''
    (notes/'results_and_discussion.md').write_text(text,encoding='utf-8')
    print(notes/'results_and_discussion.md')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config/default.yaml'); a=p.parse_args(); report(load_config(a.config))
