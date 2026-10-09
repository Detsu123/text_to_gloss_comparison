# PHOENIX14T annotation setup and provenance

Task: German written text (`translation`) → German Sign Language/DGS gloss (`orth`).
This is the reverse of the commonly reported gloss→German benchmark. Do not compare our BLEU numbers directly with those benchmark scores.

## Files used

Place these three files directly in `data/raw/` (or one nested official archive directory):

- `PHOENIX-2014-T.train.corpus.csv` (7,096 rows)
- `PHOENIX-2014-T.dev.corpus.csv` (519 rows)
- `PHOENIX-2014-T.test.corpus.csv` (642 rows)

Use the ordinary `train.corpus.csv`, not `train-complex-annotation.corpus.csv`.
The pipe-separated header is `name|video|start|end|speaker|orth|translation`.
The importer reads only `name`, `translation`, `orth`; no images or videos.
The internal format is UTF-8 JSONL with `id`, `text`, `gloss`, `split`.

The delivery includes the downloaded annotations, processed files and SHA-256 manifest.
To fetch the same metadata again, run:

```bash
python scripts/download_annotations.py
python -m src.data.preprocess
```

Downloaded source: public PHOENIX14T mirror by `aipieces`, pinned commit
`ebb29ba996f748f5775b0a95a515d49b729e2ddb`.
The exact URLs and checksums are in `data/raw/download_provenance.json`.
These are real PHOENIX14T annotations, not synthetic examples or a substituted dataset.
The mirror's bytes were NOT compared byte-for-byte with the full official archive.
Schema, official split sizes and source vocabulary were verified; this residual provenance limitation is explicit.

Official dataset: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
Official archive: https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz

The official 39 GB archive is publicly linked without an observed login wall.
Its annotation files were not found within a bounded 128 MiB streaming attempt.
The official release README was separately recovered from the start of this archive
and is retained as `data/raw/RELEASE_README.txt`. The small website README link was 404.
No access form was submitted, no credentials were used, and no gated archive was bypassed.

## Attribution and terms

Dataset authors: Necati Cihan Camgöz, Simon Hadfield, Oscar Koller, Hermann Ney,
Richard Bowden. *Neural Sign Language Translation*. CVPR 2018, pp. 7784–7793.
https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html

RWTH's general database portal states CC BY-NC-SA 4.0 and requests a registration
form for its download workflow; the PHOENIX14T-specific page instead directly links
the archive and contains a CC BY-NC-SA 3.0 footer. The archive README does not resolve
the version ambiguity. We do not treat public download as an unrestricted license.
Preserve attribution, noncommercial use and share-alike restrictions for annotations
and their normalized derivatives; confirm the applicable version with RWTH before
redistribution outside this research context.

- https://www-i6.informatik.rwth-aachen.de/web/Software/Databases/Signlanguage/?db=rwth-phoenix-weather
- https://creativecommons.org/licenses/by-nc-sa/4.0/
- https://creativecommons.org/licenses/by-nc-sa/3.0/

Raw CSV content is unchanged. Derived JSONL applies NFC and whitespace normalization,
selects/renames columns, and adds overlap indicators. Vocabularies and length thresholds
are fitted on training data only. No linguistic rewrite or deduplication is applied.

## Statistics and annotation conventions

The actual ordinary annotation import yields 2,887 source types and 1,085 target
(or 2,891 / 1,089 including four special tokens). Published papers sometimes report
1,066 gloss types and different gloss token totals. These are source-reported numbers,
not substituted for our measured values. Tamer & Saraçlar (2020) also report 1,085
training glosses. Keep the raw annotation variant and tokenizer explicit in comparisons.

Official split IDs are disjoint; identical text/gloss pairs can nevertheless recur.
The main benchmark preserves all official rows. A secondary test subset excludes
pairs already present in training; it is not used for fitting or model selection.
Statistics are in `data/processed/phoenix14t/statistics.json`.

## Synthetic fallback (only for software checks)

```bash
python scripts/make_synthetic.py
python -m src.data.preprocess --raw data/synthetic_raw --output data/processed/synthetic --synthetic
```

This creates artificial symbol reversal pairs, NOT authentic German or DGS grammar.
To train on it, copy a config and change `processed`, `output`, `experiment` accordingly.
All four tiny-overfit unit tests also use clearly synthetic integer sequences.
