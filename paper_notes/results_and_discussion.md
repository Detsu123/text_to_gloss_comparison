# Results and Discussion — phoenix14t_cpu_pilot

## Results: experimental scope

All results below are measured on real PHOENIX14T text→DGS-gloss pairs. The official training, development and test partitions contain 7,096, 519 and 642 samples. The executed configuration uses width 128, up to 10 epochs, batch 32 and device(s) cpu (precision settings are saved in each run config); seeds are [42, 43, 44]. Each architecture is trained independently for each seed. Development corpus BLEU-4 selects the checkpoint, and all training is complete before final test evaluation. These are **pilot results**, not results for the unexecuted 256-width RTX configuration.

## Overall comparison

Values are mean ± sample standard deviation across 3 seeds. Higher BLEU/chrF/exact match and lower TER/token error rate are preferred. All quality scores use a 0–100 presentation scale; error rates can exceed 100. Parameter counts are exact trainable-parameter totals. Wall time includes training, validation and checkpoint work on a shared host; concurrent seed execution prevents a controlled speed ranking.

| Model       | bleu1        | bleu2        | bleu3        | bleu4        | chrf         | ter          | exact_match   | token_error_rate   |   Parameters | Wall seconds   |
|:------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:--------------|:-------------------|-------------:|:---------------|
| RNN         | 34.28 ± 0.61 | 19.59 ± 0.76 | 11.70 ± 0.55 | 7.66 ± 0.35  | 34.15 ± 0.64 | 73.79 ± 0.90 | 1.61 ± 0.32   | 74.55 ± 0.86       |       765249 | 62.89 ± 5.04   |
| GRU         | 49.45 ± 1.07 | 31.30 ± 0.71 | 20.59 ± 0.60 | 14.28 ± 0.42 | 44.98 ± 0.62 | 63.38 ± 0.48 | 3.27 ± 0.41   | 64.42 ± 0.58       |       897345 | 88.08 ± 5.86   |
| LSTM        | 47.96 ± 0.70 | 29.69 ± 0.74 | 18.82 ± 0.90 | 12.32 ± 0.89 | 43.64 ± 0.69 | 64.29 ± 0.83 | 2.70 ± 0.24   | 65.43 ± 0.83       |       963393 | 85.51 ± 5.27   |
| Transformer | 38.26 ± 0.28 | 21.70 ± 0.07 | 12.76 ± 0.12 | 8.30 ± 0.33  | 36.11 ± 1.27 | 77.57 ± 6.77 | 1.61 ± 0.09   | 78.92 ± 6.72       |      1312961 | 228.40 ± 17.37 |

**GRU has the highest mean test BLEU-4 in this experiment**, 14.28, compared with 12.32 for LSTM (difference 1.96 BLEU points). This is a descriptive comparison; no statistical significance test has been performed. The standard deviations measure variation from training seeds, not uncertainty from drawing a new test corpus. GPU measurements, when available, are in metrics.csv; CPU-only runs contain null GPU-memory fields.

### Checkpoint selection and duration by run

| model       |   seed |   best_epoch |   epochs_run |   best_dev_bleu4 |   training_wall_seconds |   parameters |
|:------------|-------:|-------------:|-------------:|-----------------:|------------------------:|-------------:|
| rnn         |     42 |            9 |           10 |            8.412 |                  57.143 |       765249 |
| gru         |     42 |           10 |           10 |           15.74  |                  81.316 |       897345 |
| lstm        |     42 |           10 |           10 |           12.987 |                  79.456 |       963393 |
| transformer |     42 |           10 |           10 |            9.032 |                 237.985 |      1312961 |
| rnn         |     43 |            9 |           10 |            8.228 |                  64.936 |       765249 |
| gru         |     43 |           10 |           10 |           14.776 |                  91.709 |       897345 |
| lstm        |     43 |           10 |           10 |           14.764 |                  89.012 |       963393 |
| transformer |     43 |            9 |           10 |            9.012 |                 208.345 |      1312961 |
| rnn         |     44 |            9 |           10 |            8.954 |                  66.587 |       765249 |
| gru         |     44 |           10 |           10 |           16.269 |                  91.201 |       897345 |
| lstm        |     44 |           10 |           10 |           13.015 |                  88.076 |       963393 |
| transformer |     44 |           10 |           10 |            9.073 |                 238.857 |      1312961 |

The best checkpoint occurs at the final allowed epoch for 8 of 12 runs. Consequently, convergence or optimal attainable accuracy has not been established; the finite epoch cap may favor architectures that learn faster under this optimizer.

## Source-length analysis

The thresholds were fixed from training tertiles: short ≤11 source tokens, medium 12–16, long ≥17. Test counts are short 328, medium 200, long 114. Scores below are corpus BLEU-4 within each group, then summarized across seeds.

| Model       | short        | medium       | long        |
|:------------|:-------------|:-------------|:------------|
| rnn         | 13.08 ± 0.26 | 4.36 ± 0.94  | 4.15 ± 1.02 |
| gru         | 21.07 ± 1.58 | 11.98 ± 1.35 | 9.18 ± 1.01 |
| lstm        | 18.25 ± 1.65 | 10.22 ± 0.91 | 7.87 ± 0.59 |
| transformer | 14.12 ± 1.13 | 4.96 ± 1.15  | 4.08 ± 1.23 |

Across the three seeds, GRU decreases from 21.07 BLEU-4 in the short group to 9.18 in the long group; Transformer decreases from 14.12 to 4.08. Thus, this pilot does not support a Transformer advantage on the long group. All four models score lower on long than short inputs. The group results are descriptive. Different groups contain different sentences, vocabulary, targets and numbers of references. They do not isolate a causal effect of recurrent memory, and the lower group sample sizes make rankings less stable. No model was tuned to these test groups.

## Duplicate-pair sensitivity

Fourteen test rows have an exact normalized source/target pair present in training. The primary benchmark retains the original 642-row split. The secondary analysis evaluates the remaining 628 rows. Mean BLEU-4 by model on that subset is:

| model       |   BLEU-4 (mean) |
|:------------|----------------:|
| gru         |          13.833 |
| lstm        |          11.794 |
| transformer |           7.931 |
| rnn         |           7.145 |

The highest mean BLEU-4 on the overlap-excluded subset belongs to GRU. This sensitivity analysis handles exact text/gloss pairs only; it does not eliminate near-duplicate forecasts or shared domain vocabulary. Vocabularies, normalization and length thresholds were fitted without test data.

## Error-analysis scope

Complete predictions, edit-operation counts and representative cross-model examples are supplied in `predictions/`, `error_counts.csv` and `error_analysis.csv`. Candidate deletions/insertions/substitutions come from token edit alignment; repetition, ordering, very-short-output and missing-EOS flags are additional diagnostics. Transformer seed 42 reaches the 80-token cap without EOS in 6/642 examples, while the other 11 runs do not; one inspected output loops over a two-token phrase. These labels are not native-sign-language judgments. See `paper_notes/qualitative_observations.md` in the project root for inspection of concrete outputs. A single reference cannot determine whether every alternative gloss sequence is linguistically invalid.

## Discussion

The observed ordering supports a conditional model comparison under a common preprocessing and optimization protocol. It does not establish that one architecture is universally superior. The recurrent cells share the same encoder/decoder/attention implementation, so their measured differences are less exposed to unrelated preprocessing or decoder changes. The Transformer differs in depth and parameter count; matched width is not matched model capacity. A common learning rate, dropout and epoch cap also need not be equally favorable to each architecture.

The hypothesis that Transformer necessarily wins is not supported by this pilot. Claims that Transformers specifically handle long sentences better require the length-group evidence, repeated converged runs and better-controlled length comparisons. GPU efficiency cannot be inferred from CPU wall time.

The train-only vocabulary and preserved target references allow unknown-token behavior to be assessed without inflating scores by replacing reference OOVs. Preserving the original split keeps the main experiment comparable at the partition level, while the overlap-excluded sensitivity exposes one source of optimistic evaluation. Corpus/tokenizer and metric signatures must accompany any cross-paper comparison, especially because published gloss→German scores address the opposite translation direction.

## Limitations and next experiments

This is a 10-epoch-budget, 3-seed weather-domain experiment with one gloss reference per sentence and no DGS expert evaluation. There is no parameter-matched ablation, architecture-specific learning-rate search, pretrained model, dictionary baseline or beam-search comparison. A pinned public annotation mirror was used; its bytes were not independently compared with the full official archive. The 256-width, 40-epoch configuration remains a proposed next experiment. CUDA/AMP validation is supported only for runs that actually used those settings.

Next: (1) run the full RTX configuration across the same three seeds; (2) use development data for a small predeclared optimization comparison; (3) compare parameter-matched variants; (4) report paired bootstrap intervals; (5) ask qualified DGS users to inspect an anonymized sample; (6) evaluate a common beam size as a separate experiment. A future Mongolian extension requires an independently annotated, consented Mongolian text/gloss corpus and expert validation, not relabeling DGS tokens.

## Conclusion

The pipeline has produced inspectable real-data model outputs and a reproducible pilot comparison. GRU achieves the highest mean test BLEU-4 within the measured conditions. The appropriate conference claim is an empirical pilot with explicit resource and annotation limitations; algorithmic novelty, general signed-language fluency and deployment readiness are not demonstrated.
