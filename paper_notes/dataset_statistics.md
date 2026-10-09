# Observed dataset statistics

| split   |   samples |   text_vocabulary |   gloss_vocabulary |   text_mean_length |   gloss_mean_length |   text_max_length |   gloss_max_length |
|:--------|----------:|------------------:|-------------------:|-------------------:|--------------------:|------------------:|-------------------:|
| train   |      7096 |              2887 |               1085 |             13.963 |               7.786 |                52 |                 30 |
| dev     |       519 |               951 |                393 |             13.141 |               7.222 |                32 |                 18 |
| test    |       642 |              1001 |                411 |             12.174 |               6.642 |                30 |                 18 |

Vocabulary values in this table are distinct types **within each split**. Model vocabularies use training only.

Training vocabulary: 2,887 German tokens and 1,085 gloss tokens; four special tokens are added to each.

Within-train duplicate pairs: 117. Exact train–dev pair overlap: 10 rows. Exact train–test pair overlap: 14 rows.

Main test: 642 rows. Unseen-train-pair sensitivity subset: 628 rows. No missing required fields or cross-split sample IDs were observed.

Counts, lengths and OOV rates were computed from the pinned ordinary annotation CSV with case-sensitive whitespace tokenization. See raw and preprocessing manifests for SHA-256 identities.