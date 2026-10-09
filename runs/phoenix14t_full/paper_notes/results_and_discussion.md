# Results and Discussion — phoenix14t_full

## Results: experimental scope

All results below are measured on real PHOENIX14T text→DGS-gloss pairs. The official training, development and test partitions contain 7,096, 519 and 642 samples. The executed configuration uses width 256, up to 50 epochs, batch 16 and device(s) cuda (precision settings are saved in each run config); seeds are [42]. Each architecture is trained independently for each seed. Development corpus BLEU-4 selects the checkpoint, and all training is complete before final test evaluation. These results apply only to the completed configuration and devices recorded in the run metadata.

## Overall comparison

Values are mean ± sample standard deviation across 1 seeds. Higher BLEU/chrF/exact match and lower TER/token error rate are preferred. All quality scores use a 0–100 presentation scale; error rates can exceed 100. Parameter counts are exact trainable-parameter totals. Wall time includes training, validation and checkpoint work on a shared host; concurrent seed execution prevents a controlled speed ranking.

| Model       |   bleu1 |   bleu2 |   bleu3 |   bleu4 |   chrf |   ter |   exact_match |   token_error_rate |   Parameters |   Wall seconds |
|:------------|--------:|--------:|--------:|--------:|-------:|------:|--------------:|-------------------:|-------------:|---------------:|
| RNN         |   40.01 |   24.4  |   15.85 |   11.26 |  38.22 | 72.96 |          2.8  |              73.8  |      1758785 |         230.9  |
| GRU         |   52.9  |   33.45 |   22.42 |   15.74 |  46.4  | 66.72 |          3.43 |              68.2  |      2285121 |         159.91 |
| LSTM        |   55.19 |   36.31 |   24.89 |   18.02 |  48.68 | 62.62 |          3.12 |              64.35 |      2548289 |         137.62 |
| Transformer |   51.05 |   32.61 |   22.08 |   15.97 |  46.47 | 62.55 |          4.36 |              65.67 |      3935553 |         580.29 |

**LSTM has the highest mean test BLEU-4 in this experiment**, 18.02, compared with 15.97 for TRANSFORMER (difference 2.06 BLEU points). This is a descriptive comparison; no statistical significance test has been performed. The standard deviations measure variation from training seeds, not uncertainty from drawing a new test corpus. GPU measurements, when available, are in metrics.csv; CPU-only runs contain null GPU-memory fields.

### Checkpoint selection and duration by run

| model       |   seed |   best_epoch |   epochs_run |   best_dev_bleu4 |   training_wall_seconds |   parameters |
|:------------|-------:|-------------:|-------------:|-----------------:|------------------------:|-------------:|
| rnn         |     42 |           40 |           40 |           12.532 |                 230.901 |      1758785 |
| gru         |     42 |           17 |           25 |           18.877 |                 159.907 |      2285121 |
| lstm        |     42 |           13 |           21 |           20.58  |                 137.623 |      2548289 |
| transformer |     42 |           39 |           39 |           17.301 |                 580.289 |      3935553 |

The best checkpoint occurs at the final allowed epoch for 0 of 4 runs. Consequently, convergence or optimal attainable accuracy has not been established; the finite epoch cap may favor architectures that learn faster under this optimizer.

## Source-length analysis

The thresholds were fixed from training tertiles: short ≤11 source tokens, medium 12–16, long ≥17. Test counts are short 328, medium 200, long 114. Scores below are corpus BLEU-4 within each group, then summarized across seeds.

| Model       |   short |   medium |   long |
|:------------|--------:|---------:|-------:|
| rnn         |   21.58 |     4.24 |   4.43 |
| gru         |   24.63 |    13.38 |   7.35 |
| lstm        |   26.27 |    16.25 |  10.44 |
| transformer |   27.62 |    11.25 |   6.57 |

The group results are descriptive. Different groups contain different sentences, vocabulary, targets and numbers of references. They do not isolate a causal effect of recurrent memory, and the lower group sample sizes make rankings less stable. No model was tuned to these test groups.

## Duplicate-pair sensitivity

Fourteen test rows have an exact normalized source/target pair present in training. The primary benchmark retains the original 642-row split. The secondary analysis evaluates the remaining 628 rows. Mean BLEU-4 by model on that subset is:

| model       |   BLEU-4 (mean) |
|:------------|----------------:|
| lstm        |          17.735 |
| transformer |          15.625 |
| gru         |          15.206 |
| rnn         |          10.745 |

The highest mean BLEU-4 on the overlap-excluded subset belongs to LSTM. This sensitivity analysis handles exact text/gloss pairs only; it does not eliminate near-duplicate forecasts or shared domain vocabulary. Vocabularies, normalization and length thresholds were fitted without test data.

## Error-analysis scope

Complete predictions, edit-operation counts and representative cross-model examples are supplied in `predictions/`, `error_counts.csv` and `error_analysis.csv`. Candidate deletions/insertions/substitutions come from token edit alignment; repetition, ordering, very-short-output and missing-EOS flags are additional diagnostics. These labels are not native-sign-language judgments. See `paper_notes/qualitative_observations.md` in the project root for inspection of concrete outputs. A single reference cannot determine whether every alternative gloss sequence is linguistically invalid.

## Discussion

The observed ordering supports a conditional model comparison under a common preprocessing and optimization protocol. It does not establish that one architecture is universally superior. The recurrent cells share the same encoder/decoder/attention implementation, so their measured differences are less exposed to unrelated preprocessing or decoder changes. The Transformer differs in depth and parameter count; matched width is not matched model capacity. A common learning rate, dropout and epoch cap also need not be equally favorable to each architecture.

The hypothesis that Transformer necessarily wins is not supported by this pilot. Claims that Transformers specifically handle long sentences better require the length-group evidence, repeated converged runs and better-controlled length comparisons. GPU efficiency cannot be inferred from CPU wall time.

The train-only vocabulary and preserved target references allow unknown-token behavior to be assessed without inflating scores by replacing reference OOVs. Preserving the original split keeps the main experiment comparable at the partition level, while the overlap-excluded sensitivity exposes one source of optimistic evaluation. Corpus/tokenizer and metric signatures must accompany any cross-paper comparison, especially because published gloss→German scores address the opposite translation direction.

## Limitations and next experiments

This is a 50-epoch-budget, 1-seed weather-domain experiment with one gloss reference per sentence and no DGS expert evaluation. There is no parameter-matched ablation, architecture-specific learning-rate search, pretrained model, dictionary baseline or beam-search comparison. A pinned public annotation mirror was used; its bytes were not independently compared with the full official archive. Additional optimization and linguistic evaluation remain future work. CUDA/AMP validation is supported only for runs that actually used those settings.

Next: (1) run the full RTX configuration across the same three seeds; (2) use development data for a small predeclared optimization comparison; (3) compare parameter-matched variants; (4) report paired bootstrap intervals; (5) ask qualified DGS users to inspect an anonymized sample; (6) evaluate a common beam size as a separate experiment. A future Mongolian extension requires an independently annotated, consented Mongolian text/gloss corpus and expert validation, not relabeling DGS tokens.

## Conclusion

The pipeline has produced inspectable real-data model outputs and a reproducible pilot comparison. LSTM achieves the highest mean test BLEU-4 within the measured conditions. The appropriate conference claim is an empirical pilot with explicit resource and annotation limitations; algorithmic novelty, general signed-language fluency and deployment readiness are not demonstrated.
