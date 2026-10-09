# Results and Discussion — phoenix14t_stronger

## Results: experimental scope

All results below are measured on real PHOENIX14T text→DGS-gloss pairs. The official training, development and test partitions contain 7,096, 519 and 642 samples. The executed configuration uses width 384, up to 80 epochs, batch 8 and device(s) cuda (precision settings are saved in each run config); seeds are [42, 43, 44]. Each architecture is trained independently for each seed. Development corpus BLEU-4 selects the checkpoint, and all training is complete before final test evaluation. These results apply only to the completed configuration and devices recorded in the run metadata.

## Overall comparison

Values are mean ± sample standard deviation across 3 seeds. Higher BLEU/chrF/exact match and lower TER/token error rate are preferred. All quality scores use a 0–100 presentation scale; error rates can exceed 100. Parameter counts are exact trainable-parameter totals. Wall time includes training, validation and checkpoint work on a shared host; concurrent seed execution prevents a controlled speed ranking.

| Model       | bleu1        | bleu2        | bleu3        | bleu4        | chrf         | ter          | exact_match   | token_error_rate   |   Parameters | Wall seconds     |
|:------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:--------------|:-------------------|-------------:|:-----------------|
| RNN         | 32.31 ± 4.74 | 19.10 ± 2.97 | 12.26 ± 2.46 | 8.88 ± 2.02  | 33.12 ± 3.29 | 77.45 ± 2.42 | 1.66 ± 0.50   | 78.28 ± 2.29       |      3573057 | 1782.29 ± 421.25 |
| GRU         | 52.62 ± 1.24 | 33.42 ± 1.28 | 22.67 ± 0.97 | 16.22 ± 0.79 | 46.44 ± 1.32 | 64.70 ± 2.07 | 3.01 ± 0.39   | 66.12 ± 2.01       |      5938497 | 1054.81 ± 154.72 |
| LSTM        | 56.32 ± 0.52 | 38.14 ± 0.32 | 27.19 ± 0.40 | 20.09 ± 0.45 | 50.43 ± 0.42 | 58.54 ± 0.83 | 4.57 ± 0.09   | 59.90 ± 0.84       |      7121217 | 677.69 ± 56.11   |
| Transformer | 51.78 ± 0.52 | 32.34 ± 0.04 | 21.50 ± 0.32 | 15.31 ± 0.50 | 45.96 ± 0.26 | 65.50 ± 2.19 | 3.84 ± 0.32   | 68.63 ± 2.41       |     12009921 | 740.34 ± 28.72   |

**LSTM has the highest mean test BLEU-4 in this experiment**, 20.09, compared with 16.22 for GRU (difference 3.87 BLEU points). This is a descriptive comparison; no statistical significance test has been performed. The standard deviations measure variation from training seeds, not uncertainty from drawing a new test corpus. GPU measurements, when available, are in metrics.csv; CPU-only runs contain null GPU-memory fields.

### Checkpoint selection and duration by run

| model       |   seed |   best_epoch |   epochs_run |   best_dev_bleu4 |   training_wall_seconds |   parameters |
|:------------|-------:|-------------:|-------------:|-----------------:|------------------------:|-------------:|
| rnn         |     42 |           75 |           80 |           11.771 |                2181.13  |      3573057 |
| gru         |     42 |           18 |           30 |           16.818 |                 877.123 |      5938497 |
| lstm        |     42 |           11 |           23 |           20.535 |                 731.355 |      7121217 |
| transformer |     42 |           38 |           38 |           17.589 |                 732.473 |     12009921 |
| rnn         |     43 |           39 |           51 |            7.407 |                1341.74  |      3573057 |
| gru         |     43 |           26 |           38 |           18.656 |                1127.65  |      5938497 |
| lstm        |     43 |           10 |           22 |           21.678 |                 682.3   |      7121217 |
| transformer |     43 |           38 |           39 |           17.034 |                 716.364 |     12009921 |
| rnn         |     44 |           65 |           77 |           10.571 |                1824.01  |      3573057 |
| gru         |     44 |           29 |           41 |           18.229 |                1159.67  |      5938497 |
| lstm        |     44 |            8 |           20 |           21.118 |                 619.418 |      7121217 |
| transformer |     44 |           35 |           42 |           17.795 |                 772.172 |     12009921 |

The best checkpoint occurs at the final allowed epoch for 0 of 12 runs. Consequently, convergence or optimal attainable accuracy has not been established; the finite epoch cap may favor architectures that learn faster under this optimizer.

## Source-length analysis

The thresholds were fixed from training tertiles: short ≤11 source tokens, medium 12–16, long ≥17. Test counts are short 328, medium 200, long 114. Scores below are corpus BLEU-4 within each group, then summarized across seeds.

| Model       | short        | medium       | long         |
|:------------|:-------------|:-------------|:-------------|
| rnn         | 16.62 ± 3.78 | 4.22 ± 1.23  | 3.45 ± 0.91  |
| gru         | 24.90 ± 0.54 | 13.12 ± 1.00 | 9.29 ± 1.38  |
| lstm        | 28.79 ± 0.53 | 18.38 ± 1.06 | 12.36 ± 1.00 |
| transformer | 25.26 ± 1.75 | 12.19 ± 0.48 | 4.75 ± 1.17  |

The group results are descriptive. Different groups contain different sentences, vocabulary, targets and numbers of references. They do not isolate a causal effect of recurrent memory, and the lower group sample sizes make rankings less stable. No model was tuned to these test groups.

## Duplicate-pair sensitivity

Fourteen test rows have an exact normalized source/target pair present in training. The primary benchmark retains the original 642-row split. The secondary analysis evaluates the remaining 628 rows. Mean BLEU-4 by model on that subset is:

| model       |   BLEU-4 (mean) |
|:------------|----------------:|
| lstm        |          19.71  |
| gru         |          15.728 |
| transformer |          15.052 |
| rnn         |           8.302 |

The highest mean BLEU-4 on the overlap-excluded subset belongs to LSTM. This sensitivity analysis handles exact text/gloss pairs only; it does not eliminate near-duplicate forecasts or shared domain vocabulary. Vocabularies, normalization and length thresholds were fitted without test data.

## Error-analysis scope

Complete predictions, edit-operation counts and representative cross-model examples are supplied in `predictions/`, `error_counts.csv` and `error_analysis.csv`. Candidate deletions/insertions/substitutions come from token edit alignment; repetition, ordering, very-short-output and missing-EOS flags are additional diagnostics. These labels are not native-sign-language judgments. See `paper_notes/qualitative_observations.md` in the project root for inspection of concrete outputs. A single reference cannot determine whether every alternative gloss sequence is linguistically invalid.

## Discussion

The observed ordering supports a conditional model comparison under a common preprocessing and optimization protocol. It does not establish that one architecture is universally superior. The recurrent cells share the same encoder/decoder/attention implementation, so their measured differences are less exposed to unrelated preprocessing or decoder changes. The Transformer differs in depth and parameter count; matched width is not matched model capacity. A common learning rate, dropout and epoch cap also need not be equally favorable to each architecture.

The hypothesis that Transformer necessarily wins is not supported by this pilot. Claims that Transformers specifically handle long sentences better require the length-group evidence, repeated converged runs and better-controlled length comparisons. GPU efficiency cannot be inferred from CPU wall time.

The train-only vocabulary and preserved target references allow unknown-token behavior to be assessed without inflating scores by replacing reference OOVs. Preserving the original split keeps the main experiment comparable at the partition level, while the overlap-excluded sensitivity exposes one source of optimistic evaluation. Corpus/tokenizer and metric signatures must accompany any cross-paper comparison, especially because published gloss→German scores address the opposite translation direction.

## Limitations and next experiments

This is a 80-epoch-budget, 3-seed weather-domain experiment with one gloss reference per sentence and no DGS expert evaluation. There is no parameter-matched ablation, architecture-specific learning-rate search, pretrained model, dictionary baseline or beam-search comparison. A pinned public annotation mirror was used; its bytes were not independently compared with the full official archive. Additional optimization and linguistic evaluation remain future work. CUDA/AMP validation is supported only for runs that actually used those settings.

Next: (1) run the full RTX configuration across the same three seeds; (2) use development data for a small predeclared optimization comparison; (3) compare parameter-matched variants; (4) report paired bootstrap intervals; (5) ask qualified DGS users to inspect an anonymized sample; (6) evaluate a common beam size as a separate experiment. A future Mongolian extension requires an independently annotated, consented Mongolian text/gloss corpus and expert validation, not relabeling DGS tokens.

## Conclusion

The pipeline has produced inspectable real-data model outputs and a reproducible pilot comparison. LSTM achieves the highest mean test BLEU-4 within the measured conditions. The appropriate conference claim is an empirical pilot with explicit resource and annotation limitations; algorithmic novelty, general signed-language fluency and deployment readiness are not demonstrated.
