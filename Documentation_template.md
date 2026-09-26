# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Claude's Plan

**Team Members:** Rahul Bokade (leader), Hrishita Chavan, Riddhesh Budha, Umed Indulkar

**Submission Date:** 2026-09-27

## 1. Executive Summary

The pipeline links each Source 1 record to the Source 2/3 records of the same business in six steps:

1. Normalization, including a learned native-script-to-Latin dictionary and country-aware address rules.
2. Candidate generation, as the union of rare-key blocking and nearest neighbours from a fine-tuned multilingual
   bi-encoder (pair recall 0.997).
3. A pairwise XGBoost matcher over 67 string, IDF, legal-form, house-number and embedding features.
4. Fine-tuned multilingual cross-encoders (small, base and large), blended with the matcher for countries that
   have training labels.
5. Set-level decoding that maximizes expected F0.5 per Source 1 entity, with at most one Source 1 entity per
   target record.
6. For a country without training labels (France), rules derived from how the data generator builds true records
   and decoys, checked against the training labels.

The test data differs from the training data in two ways, and both are handled without test labels:

- It holds about twice as many look-alike decoy records per entity. A per-segment prior-shift correction
  recalibrates the test probabilities.
- It includes an unlabelled country (France) whose business names are generic. For any country absent from the
  labels, the pipeline falls back to the pairwise matcher and then applies the generator-pattern rules.

Holdout macro F0.5 (22,144 training entities, singletons included) rose from 0.924 to 0.992.

## 2. Data and Preparation

The training set has 2,206,821 Source 1 records and 10,320,219 Source 2/3 records (7,638,365 labelled links; every
target belongs to at most one Source 1 entity; 26% of targets belong to no Source 1 entity). The test set has
1,732,544 Source 1 and 9,969,589 Source 2/3 records, including France, which has no training labels.

Noise seen in the data: native-script names (Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati, Gurmukhi, Odia,
Malayalam; 9% of test targets), native-script state names, typos and character swaps, inserted or removed words,
legal-suffix changes, reordered tokens, trade names, missing addresses, house-number typos and leading zeros,
abbreviated street types and state or department names.

Normalization (`normalize.py`, `translit.py`):

- **Transliteration dictionary.** Learned from matched training pairs. Name tokens are aligned by position when the
  token counts agree; address tokens are aligned by co-occurrence (Dice score). The dictionary has 1,363 entries
  and covers 97.4% of native-script test tokens. Unknown tokens fall back to Unicode transliteration.
- **Address rules per country.** US and India street abbreviations; French abbreviations (R., AV., BD, ALL, IMP,
  RTE, ST. = Saint, and others).
- **Regions.** Each US state, Indian state and French region/department collapses to one canonical token, so
  "Texas" equals "TX" and "Gironde" equals "Nouvelle-Aquitaine".
- **Legal forms and suffixes.** Runs of single letters are merged ("S.A.R.L." becomes "sarl", "L.L.C." becomes
  "llc"). Legal forms and the suffixes the data adds to French names (Cie, Compagnie, & Fils, Frères, Associés,
  Groupe, Développement) are dropped from the canonical name, as "Inc" or "Pvt Ltd" are.
- **Derived fields.** Sorted canonical name/address strings, the house number, and anchor tokens.
- **Country** is an open string label. The pipeline has no US/India-only branch.

## 3. Candidate Generation

Two independent generators are unioned; the union is exactly `candidate_pairs.tsv`.

**Key blocking** (`blocking.py`): exact canonical name or address, name-token pairs, name/address and house/address
token combinations, and prefix keys. Keys with long posting lists are dropped. The 30 best targets per Source 1
record are kept, ranked by Jaro-Winkler similarity and key evidence.

**Bi-encoder retrieval** (`embed.py`): `intfloat/multilingual-e5-small` (MIT, 118M parameters) fine-tuned with a
symmetric in-batch contrastive loss on 1.5M matched pairs. It uses training entities in hash buckets >= 100 only,
so the matcher's training and holdout entities are unseen. Batches group records with the same name prefix or city
to create hard negatives. All records are encoded and an exact cosine top-40 search runs per country on GPU. A
reverse search (each target against Source 1, top 5) supplies competition features.

Pair recall on 458,727 training links: keys 0.934, encoder top 40 0.997, union 0.997, at about 63 candidates per
Source 1 record (108M test pairs instead of 17 trillion possible pairs).

## 4. Matching Model and Features

**Pair features (67):**

- String similarities: raw and canonical Jaro-Winkler, bigram and token Jaccard, edit similarity.
- IDF-weighted token overlap (IDF computed over all records of a split): cosine, coverage, and the rarest shared
  and missing token.
- Legal-form agreement and conflict (Inc, LLC, Pvt, Ltd, SARL, SAS and others).
- House numbers: first-number match, conflict, edit distance and prefix/suffix relation. Also address-wide
  evidence: whether each record's house number appears anywhere among the other record's address numbers, and the
  gap and digit-difference shape to the closest number. This handles noise prefixes such as
  "Door No #834 House No 1958".
- Missing fields and name ambiguity (how many Source 1 or target records share the name).
- Embedding features: cosine, rank, gap to the best candidate, the best Source 1 match of the target, and the rank
  of this Source 1 record among them.
- Sibling agreement: similarity to the other candidates of the same Source 1 record.

**Stage 1:** GPU XGBoost (depth 9, learning rate 0.05, early stopping) trained on training entities in buckets
0-99 except the holdout (buckets 50-59), with four folds. Every training pair gets an out-of-fold score; the test
gets the fold average.

**Cross-encoders** (`crossenc.py`, `kaggle/`): pair classifiers (mean-pooled encoder, linear head) fine-tuned on
training entities from buckets >= 100, so the blend's training and holdout entities are unseen. Backbones:
`intfloat/multilingual-e5-small` (three seeds) and `multilingual-e5-base` on a local GPU;
`intfloat/multilingual-e5-large`, `BAAI/bge-reranker-v2-m3` and `FacebookAI/xlm-roberta-large` on Kaggle
2x T4 notebooks (DDP, fp16, frozen word embeddings). Training pairs are either encoder neighbours plus all true
links, or "hard pairs": every stage-1 candidate with p >= 0.01 plus two neighbours and the true links, which is the
region where the blend uses cross-encoder scores. They rescore every pair with stage-1 p >= 0.01 (8.4M test pairs).

**Pairwise blend:** XGBoost (depth 6) over logit(stage-1 p), every cross-encoder logit, the 67 stage-1 features and
name-genericness inputs. It is fitted on out-of-fold training pairs of buckets 0-99; holdout logloss on rescored
pairs is 0.044 (stage 1 alone: 0.073).

**Why no entity-context stage:** a second XGBoost stage with entity context was tried. Its inputs were rank and
sums of scores per entity, sibling support, the strongest competing Source 1 claim per target, and house-number
clusters. It raised holdout F0.5 to 0.9877. On the test set, however, those features inflate look-alike decoy
records (see section 5), so the final scorer is purely pairwise.

**Countries without labels:** the blend and the cross-encoders learn how much a name agreement is worth from US and
India, where business names are distinctive. In a country absent from the labelled holdout (France in the test
set), names are generic, e.g. "Lille Ecole SARL" exists many times. On France the blend puts many same-address
look-alikes at high scores, and its calibrated posterior stays near 0.3 even at the top bins, so the pipeline uses
the stage-1 score for any country without labels. Country stays an open label: the rule depends only on which
countries the labelled data contains.

**Decoding:** each target record is kept only for its best-scoring Source 1 entity (ties broken by the raw score).
For each entity, the candidate prefix (ranked by probability) that maximizes expected F0.5 is predicted, or an
empty list when that scores higher.

**Generator-pattern rules for the unlabelled country** (`france_rule.py`, `gen_features.py`). Each rule comes from
a training-label statistic about how true records and decoys are built, and is applied only to the country without
labels:

- *Restore noise patterns at the Source 1 address:* acronyms that spell the Source 1 initials, web domains that
  carry a Source 1 word or the initials, DBA/"formerly" aliases, near-typos and the Source 1 name plus extra words.
  In training such same-address pairs are 94-99.9% true; stage 1 scores the short French acronyms low (median 0.35)
  because the same acronym recurs across many targets.
- *Drop house-number conflicts that are not typos:* a typo keeps the number up to leading zeros, one changed digit
  or one dropped digit; France's accepted conflicts are unrelated numbers, the decoy signature.
- *Drop common-word swaps:* a record whose name replaces one Source 1 word by another common vocabulary word (not a
  filler such as "services", not a typo or abbreviation). In training, same-address swaps to a common word are true
  45% of the time, against 96-99% for fillers and garbled typos. France's swaps use exactly the words of its decoys
  a few doors away (club, école, comité, amicale), and decoy placement in training implies about 23% true.

## 5. Validation and Error Analysis

Holdout: 22,144 training Source 1 entities (hash buckets 50-59), macro F0.5 with singletons, never used for training
the encoder, the cross-encoders or the matchers. `docs/experiments.md` lists every result.

| Version | Holdout F0.5 | Leaderboard |
|---|---|---|
| Key blocking + XGBoost (27 features) | 0.924 | 0.897 |
| + normalization, encoder kNN, IDF/legal/embedding features, two stages | 0.983 | 0.961 |
| + test prior-shift correction | 0.983 | 0.968 |
| Stage 1 + two small cross-encoders, per-country scorer, address-wide house features | 0.990 | 0.9848 |
| + third small and first large cross-encoder, one winner per target | 0.991 | 0.9853 |

**Distribution shift.** The confident pairs per entity (p >= 0.99) are the same on holdout and test. Mid-range pairs
are 2-5 times more frequent on test, almost entirely where the house numbers conflict. Inspection shows decoy
entities absent from Source 1: the same or a perturbed name on the same street a few doors away (for example
"Williams Templeton" at 2212 Leeward Shore Dr, with decoys at 2213 and 2217).

`calibrate.py` corrects this without test labels. For each country, house relation (equal, conflict, unknown) and
score bin, it keeps the holdout positives per entity (scaled by a factor fitted on the top bins) and counts the
excess test pairs as negatives. The corrected posterior replaces the raw score before decoding.

**Label-free diagnostics.** Every labelled country has the same number of true links per Source 1 entity (US 3.459,
India 3.465, 5.6% singletons), so the generator is country-agnostic in that respect. The pipeline predicts 3.39 links
per entity for US and India but 3.20 for France, which located the France errors: generic-word swaps and
acronyms at the Source 1 address. France's true records are also cleaner than training ones (1.85 identical-name,
same-address targets per entity against 1.14). House-number statistics from training guided the conflict rule: with
the same street and name, a last-digit change of 3-10 is only 13.5% true (a decoy a few doors away), while an inner
digit change or a dropped digit is 82-99% true (a typo).

**Rules checked on labelled data.** The same drop rules applied to the US/India holdout decisions of the
cross-encoder blend would remove only true links (precision 0.99-1.00), so they are not applied there: the
cross-encoders already separate typos from decoys when labels exist.

**Remaining errors.** On the holdout, most missed links are records without an address whose name is shared by
several Source 1 entities, which is ambiguous even to a human. On the test set, the largest remaining uncertainty is
France, where generic names leave the address as the only strong signal.

## 6. Reproduction and Fair Play

`README.md` lists every command from the raw TSVs to the validated outputs. The archive contains all source code,
pinned dependencies (`pyproject.toml`, `uv.lock`, `requirements.txt`), the transliteration dictionary, the fine-tuned
encoder and the XGBoost models, both output TSVs and this document. Raw challenge data is not included.

Only the supplied TSVs are used. No external business database, API, geocoder or data augmentation. The pretrained
weights are general multilingual text encoders and rerankers under the MIT or Apache-2.0 license
(`multilingual-e5-small/base/large`, `bge-reranker-v2-m3`, `xlm-roberta-large`; at most 568M parameters each, far
below 8B).
