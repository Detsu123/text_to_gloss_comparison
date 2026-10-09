# A Controlled Comparison of Recurrent and Transformer Models for German Text-to-Gloss Translation

## 1. Introduction

Text-to-gloss translation predicts an ordered sequence of sign annotation labels from written language. It is an intermediate representation task rather than a complete signed-language communication system: gloss does not encode all spatial or nonmanual information. PHOENIX14T pairs German weather-broadcast transcriptions with DGS gloss annotation [1]. Reversing these aligned fields supports a text-to-gloss experiment without processing video. This study compares four lightweight encoder–decoder architectures through a shared implementation and evaluation protocol.

## 2. Research problem

The practical problem is to establish reproducible baselines under a laptop-scale model budget while separating architectural effects from changes in preprocessing, data splits, decoding and metric implementation. Applying familiar architectures is not an algorithmic novelty claim. The contribution is a controlled empirical comparison, an explicit duplicate-pair audit, model complexity measurement and source-length stratification. This is a pilot study until longer runs and stronger linguistic validation are completed.

## 3. Research questions

1. How do vanilla RNN, GRU and LSTM encoders/decoders with the same attention compare with a Transformer?
2. What accuracy–parameter trade-offs arise under matched hidden dimensions and a common epoch budget?
3. Does performance change across train-defined source-length groups?
4. Which observable token-level errors characterize the outputs?

There is no assumed Transformer winner. A group score alone cannot establish a causal long-range modeling advantage because vocabulary, content, target length and sample count vary with input length.

## 4. Related work and scope

Camgöz et al. [1] introduced PHOENIX14T for sign-video/gloss-to-spoken-language translation; those direction-specific scores are not text-to-gloss baselines for this work. Egea Gómez, McGill and Saggion [2] explicitly studied Transformer-based text-to-sign-gloss translation with syntactic features. Saunders et al. [3] studied sign production including a gloss intermediate. These precedents exclude any claim that the text-to-gloss task or use of Transformers is new. Luong et al. [4] motivates general attention for recurrent translation models; Vaswani et al. [5] provides the Transformer architecture. Post [6] motivates explicit, reproducible BLEU configurations. This is a focused source review, not a claim of exhaustive state-of-the-art coverage through 2026.

## 5. Dataset

The experiment uses the ordinary PHOENIX14T annotation CSVs with official train/dev/test row counts of 7,096/519/642. Only `translation` and `orth` form the learning pair; `name` is the sample identifier. The public pinned metadata mirror, original archive README, licensing caveat and checksums are documented in `data/README.md`. The full video archive was not downloaded. The observed train vocabulary is 2,887 German types and 1,085 gloss types. Distinguish these observed values from published counts under different annotation conventions. Same-pair overlap is audited without silently changing the benchmark split.

## 6. Preprocessing

NFC Unicode and whitespace normalization are applied equally to all architectures. Word-level whitespace tokenization preserves case, umlauts, punctuation, digits and compound gloss markers. No stopword removal, stemming, lowercasing, linguistic gloss cleanup or pretrained tokenizer is used. Source sequences append EOS; target sequences use SOS and EOS. PAD, UNK, SOS and EOS occupy fixed indices 0–3. Vocabularies include every training type and no held-out types. Test references remain in their original normalized token strings, so target OOVs are not incorrectly replaced by UNK during scoring. No sequences are truncated during preprocessing or training.

## 7. Architectures

All models learn separate source and target embeddings and an untied output projection. The three recurrent systems use a unidirectional encoder and decoder with the same cell type, hidden dimension and layer count. Encoder states initialize the decoder (both hidden and cell states for LSTM). Packed sequences avoid encoding padded suffixes. Luong general attention attends from every decoder state to masked encoder states; their concatenation passes through a tanh projection and output layer. At full teacher forcing the decoder runs vectorized across target positions; partial teacher forcing uses a stepwise schedule.

The Transformer uses sinusoidal positions, encoder self-attention and a causal decoder with cross-attention. Boolean causal, source-padding, target-padding and memory-padding masks are applied. It uses two encoder and two decoder layers in both supplied research configurations. This is a matched-width comparison, not a parameter-matched comparison: depth, parameter counts, inductive biases and optimization suitability differ.

## 8. Experimental setup

The executed CPU pilot configuration uses 128-dimensional embeddings/hidden states, one recurrent layer, two Transformer encoder and decoder layers, four heads, feed-forward dimension 256, dropout 0.2 and batch size 32. AdamW uses learning rate 0.001 and weight decay zero. All models use teacher forcing 1.0, token-mean cross entropy ignoring PAD and gradient clipping at 1.0. The budget is ten epochs; validation BLEU-4 selects the earliest best checkpoint. Patience 10 cannot terminate before the ten-epoch cap. The final checkpoint is selected without inspecting test scores. Seeds 42, 43 and 44, actual durations, versions and device are recorded in each run. Independent seed runs can overlap on this shared 8-CPU-quota host (two Torch threads per run); their wall-clock values are descriptive, not controlled speed benchmarks. Greedy decoding suppresses PAD/SOS and ends at EOS or 80 generated tokens.

A separate, unexecuted RTX 4050 configuration specifies width 256, feed-forward 512, batch 16, 40 epochs and patience 8. CUDA mixed precision is enabled only on CUDA. Results of the CPU pilot do not establish the accuracy or memory usage of that larger configuration. Gradient accumulation correctly normalizes by actual non-PAD token count, including incomplete final windows. CPU-only mode and configurable data-loader workers are supported.

## 9. Evaluation metrics

BLEU-1 through BLEU-4 are **cumulative corpus BLEU scores** through the named maximum order, not separate n-gram precisions. SacreBLEU 2.5.1 uses case-sensitive, already-tokenized gloss strings (`tokenize=none`), exponential smoothing and no effective-order reduction. chrF uses character order 6, no word n-grams, beta 2. TER is case-sensitive with normalization and punctuation removal disabled. BLEU/chrF increase with quality; TER decreases. Exact match is a sentence percentage. Token error rate is summed Levenshtein distance divided by total reference gloss tokens, expressed as a percentage; unlike TER it does not include shift operations. Metric signatures are retained. Scores are lexical proxies, not expert measures of DGS grammaticality or comprehensibility.

## 10. Results

See generated `results_and_discussion.md` for measured values, scope, sample sizes and model ranking. No unmeasured default-config or GPU results are reported.

## 11. Error analysis

All predictions are retained with original normalized references. Token edit operations provide candidate deletions, insertions and substitutions. Equal multisets with different order flag possible ordering errors; consecutive repeats, very short outputs, UNK output and missing EOS are separately flagged. These automatic flags are descriptive proxies, not verified linguistic labels. A valid repeated sign must not automatically be judged an error. Representative examples are chosen after final evaluation: low-edit, medium-length, long, disagreement and high-edit cases. Expert human annotation remains necessary for semantic adequacy, lexical acceptability, premature EOS and hallucination claims.

## 12. Discussion

The measured discussion is supplied separately. Interpret model rankings as conditional on this corpus, annotation variant, initialization and ten-epoch optimization budget. A common learning rate and epoch cap control some factors but need not suit every architecture equally. CPU wall-clock time includes validation/checkpoint work and shared-host scheduling, so it is not an RTX latency benchmark. Input-length group metrics describe associations, not proof of memory mechanisms.

## 13. Limitations

Weather-domain restriction, a single reference per sentence, annotation/version differences, recurrent text pairs across split boundaries, no parameter matching, short training budget, no architecture-specific tuning and lack of DGS expert judgments limit external validity. The public mirror was pinned and audited but not byte-compared against the full official archive. No GPU was available, so CUDA AMP behavior and peak laptop VRAM are not measured. Gloss outputs are not sign videos and should not be presented as a finished accessibility product. Multiple seeds characterize training variation only; they are not test-sampling confidence intervals or a significance test.

## 14. Conclusion

The implementation enables a reproducible controlled comparison and preserves the evidence needed to inspect it. Conclusions about the winning architecture and sentence-length behavior must be stated from the measured result section, with the pilot scope visible. Do not promote the pilot into an unrestricted claim of architectural superiority.

## 15. Future work and Mongolian extension

First run the supplied full configuration on RTX 4050 across seeds 42, 43 and 44, then use development data for a small predeclared learning-rate comparison and evaluate a newly frozen protocol. Add parameter-matched variants and paired bootstrap confidence intervals. Beam search may be evaluated separately with the same beam size and length penalty for every model; it is intentionally omitted here. A handcrafted dictionary/grammar baseline is excluded because no validated DGS mapping or grammar specification was supplied or established. Simple capitalization/deletion would misrepresent sign language grammar.

For Mongolian, obtain a consented parallel Mongolian-text/Монгол дохионы хэл gloss corpus with annotation guidelines co-designed with Deaf signers and qualified linguists. Define segmentation, gloss conventions, spatial/nonmanual limitations and acceptable translation variants. Split by speaker/session/topic before modeling to avoid near-duplicate leakage. Replace tokenizer and train-only vocabularies, retain the model/training framework, and repeat leakage/OOV checks. DGS gloss labels cannot be relabeled as Mongolian signs. Evaluate with native experts and use multiple references where feasible. No Mongolian model has been implemented or evaluated in this delivery.
