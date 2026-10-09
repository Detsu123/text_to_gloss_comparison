# Verification evidence

`verification.txt` is the actual pytest output. Synthetic integer sequences are used
only for unit-level tiny-overfit checks; all reported pilot scores use real PHOENIX14T.

- Shape, forward loss, gradients, causal behavior and pad masks: CPU tests for all four models.
- Tiny overfit: each model learns two synthetic reversal examples; exact greedy reconstruction required.
- Save/load: state-dictionary roundtrip must preserve greedy output; real checkpoint restore exercised separately.
- Metrics: identity, empty hypotheses and a cumulative BLEU cross-check against SacreBLEU.
- Preprocessing: training-only vocabulary, OOV count, duplicate-pair audit, strict split counts and ID overlap rejection.
- Full training: all four architectures on official-sized real train/dev splits, three fixed seeds.
- Inference/evaluation: real best-checkpoint loading, 642 test predictions per model per seed.
- Graphics: all eight figure families inspected as exported PNGs; score labels reconciled to CSV means; error-bar labels moved above the bars and uncertainty captions added. PDF/SVG exports were generated from the same figures, not separately rendered for inspection.

CUDA, AMP and the 256-width full training configuration have not been executed here.
The GPU profiling command is included for the target machine. There is no claim that
CPU testing proves bit-identical GPU results or a measured 6 GB peak.
