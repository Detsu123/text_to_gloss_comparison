# Inspection of real held-out predictions

Scope: normalized PHOENIX14T test outputs after the training protocol was frozen.
The observations below are token-level inspection by the assistant, **not DGS expert
validation**. Seed 42 is used consistently in the compact table; every seed's full
outputs are in `results/predictions/`. The multi-seed representative selection is
recorded in `results/error_analysis.csv`.

## Representative examples

| Case / source ID | Reference | RNN | GRU | LSTM | Transformer |
|---|---|---|---|---|---|
| Easy: `11November_2009_Wednesday_tagesschau-8042` | WIND SCHWACH MAESSIG WEHEN | WIND SCHWACH MAESSIG WEHEN | WIND SCHWACH MAESSIG WEHEN | WIND SCHWACH MAESSIG WEHEN | WIND SCHWACH MAESSIG WEHEN |
| Medium: `27November_2009_Friday_tagesschau-7340` | MONTAG WOLKE MAL REGEN | MONTAG WOLKE TEIL REGEN | MONTAG MEISTENS WOLKE TEIL REGEN | MONTAG MEISTENS WOLKE TEIL REGEN | MONTAG WOLKE REGEN |
| Disagreement: `01November_2010_Monday_tagesschau-133` | MORGEN DANN HERBST MISCHUNG HOCH NEBEL WOLKE SONNE | MORGEN FRUEH KOMMEN SONNE WOLKE | MORGEN HOCH DRUCK KOMMEN WOLKE SONNE | MORGEN BLEIBEN WOLKE SONNE WOLKE WECHSELHAFT | MORGEN SONNE WOLKE |
| Failure: `10January_2011_Monday_tagesschau-6403` | IN-KOMMEND DANN WECHSELHAFT STARK MILD | WEST MEHR WARM | NAECHSTE WOCHE MEHR MILD | NAECHSTE WOCHE MEHR KUEHL MILD | NAECHSTE WOCHE repeated 40 times; reaches the 80-token cap without EOS |

The failure-cell notation is a compact description, not a modified prediction.
The complete 80-token output is retained in the prediction CSV.

German source sentences, in table order:

1. `der wind weht schwach bis mäßig aus süd bis südwest`
2. `am montag ist es meist stark bewölkt hier und da regnet es`
3. `auch morgen erwartet uns eine ruhige herbstmischung aus hochnebel wolken und sonne`
4. `im verlauf der woche geht es dann unbeständig und sehr mild weiter`

The first example demonstrates exact reference reconstruction rather than general
translation fluency. In the medium example, the Transformer omits the reference token
`MAL`; the recurrent predictions use `TEIL` instead and GRU/LSTM add `MEISTENS`.
These are observable edit differences; some alternative choices might be acceptable
under richer annotation, which a single-reference score cannot establish.

The disagreement case shows losses of the reference's `HERBST MISCHUNG` and varying
substitutions/insertions. GRU produces `HOCH DRUCK` where the reference contains
`HOCH NEBEL`. This is a candidate lexical error worth expert review, not proof of a
specific DGS grammatical violation.

## A long-source example

ID: `28March_2011_Monday_tagesschau-2285`.

Source: `im süden gibt es zu beginn der nacht noch wolken die hier und da auch noch ein paar tropfen fallen lassen sonst ist es meist klar oder nur locker bewölkt`

Reference: `SUED WOLKE KOMMEN BISSCHEN REGEN KOENNEN REGION KLAR LOCKER WOLKE`

- RNN: `SUED REGION HEUTE NACHT WOLKE HIMMEL IX BISSCHEN REGEN`
- GRU: `SUED NACHT WOLKE HIMMEL KOENNEN KLAR HIMMEL STERN KOENNEN SEHEN`
- LSTM: `SUED NACHT WOLKE HIMMEL HIMMEL STERN KOENNEN SEHEN REGION KLAR HIMMEL KLAR HIMMEL`
- Transformer: `HEUTE NACHT REGION KLAR HIMMEL STERN KOENNEN SEHEN`

All four differ substantially from the reference. The GRU output, for example, lacks
reference tokens `KOMMEN`, `BISSCHEN`, `REGEN`, `REGION` and `LOCKER`; it introduces
other weather-related tokens. LSTM repeats the two-token phrase `KLAR HIMMEL` at the
end. This is one concrete long-input failure, not a representative estimate by itself.
The length-group corpus analysis provides the quantitative counterpart.

## EOS and repetition behavior

Transformer seed 42 reaches the 80-token cap without EOS on 6 of 642 examples
(0.93%). None of the other 11 evaluated runs reaches the cap. Its TER is 85.37,
compared with 73.22 and 74.13 for Transformer seeds 43 and 44. Long repeated outputs
are a plausible contributor to this variation; no causal ablation was performed.

The failure above repeats a two-token cycle. The automatic `repetition_candidate`
field checks adjacent equal tokens, so it does not cover every multi-token loop.
This limitation is why representative outputs were actually inspected. Do not
present the automated repetition count as an exhaustive count of looping errors.

## What remains for human assessment

The `human_verified_category` column is intentionally blank. Qualified DGS evaluators
should assess lexical adequacy, ordering, omitted semantic content, acceptable
variation and comprehensibility. A very short output is only a truncation candidate:
its length alone does not establish premature EOS. No output UNK token was observed,
but that does not mean unknown source words were handled correctly; source OOV counts
and their predictions remain available for review.
