# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Claude's Plan

**Team Members:** Rahul Bokade (leader), Hrishita Chavan, Riddhesh Budha, Umed Indulkar

**Submission Date:** 2026-09-26

## 1. Executive Summary

The pipeline links each Source 1 record to the Source 2/3 records of the same business in four steps:

1. Normalization, including a learned dictionary that maps native-script Indian tokens to Latin spellings.
2. Candidate generation, as the union of rare-key blocking and nearest neighbours from a fine-tuned multilingual
   bi-encoder.
3. A two-stage XGBoost matcher.
4. Set-level decoding that maximizes expected F0.5 per Source 1 entity, with at most one Source 1 entity per
   target record.

The test data contains about twice as many look-alike decoy records per entity as the training data. A label-free
prior-shift correction recalibrates the test probabilities before decoding.

Holdout macro F0.5 (22,144 training entities, singletons included) rose from 0.924 to 0.983. The public leaderboard
score rose from 0.897 to 0.968.

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

**Pair features (about 60):**

- String similarities: raw and canonical Jaro-Winkler, bigram and token Jaccard, edit similarity.
- IDF-weighted token overlap (IDF computed over all records of a split): cosine, coverage, and the rarest shared
  and missing token.
- Legal-form agreement and conflict (Inc, LLC, Pvt, Ltd, SARL, SAS and others).
- House numbers: exact match, conflict, edit distance, prefix/suffix relation, and address-number overlap.
- Missing fields and name ambiguity (how many Source 1 or target records share the name).
- Embedding features: cosine, rank, gap to the best candidate, the best Source 1 match of the target, and the rank
  of this Source 1 record among them.
- Sibling agreement: similarity to the other candidates of the same Source 1 record.

**Stage 1:** GPU XGBoost trained on training entities in buckets 0-99 except the holdout (buckets 50-59), with four
folds. Every training pair gets an out-of-fold score; the test gets the fold average.

**Stage 2** adds context from the stage-1 scores:

- Within the entity: rank, margin, count and sum of confident candidates, and sibling support weighted by score.
- Across entities: the strongest competing Source 1 claim on the same target.
- House-number clusters. Decoy businesses sit a few doors away on the same street and bring several records that
  share their own house number. A typo on a true record leaves its house number alone among the candidates.

Both stages use XGBoost 3.4.1 (Apache-2.0) with early stopping on the holdout.

**Decoding:** each target record is kept only for its best-scoring Source 1 entity. For each entity, the candidate
prefix (ranked by probability) that maximizes expected F0.5 is predicted, or an empty list when that scores higher.

## 5. Validation and Error Analysis

Holdout: 22,144 training Source 1 entities (hash buckets 50-59), macro F0.5 with singletons, never used for training
the encoder or the matchers. `docs/experiments.md` lists every result.

| Version | Holdout F0.5 | Leaderboard |
|---|---|---|
| Key blocking + XGBoost (27 features) | 0.924 | 0.897 |
| + normalization, encoder kNN, IDF/legal/embedding features, stage 2 | 0.983 | 0.961 |
| + test prior-shift correction | 0.983 | 0.968 |

**Distribution shift.** The confident pairs per entity (p >= 0.99) are the same on holdout and test. Pairs scored
0.3-0.98 are 2-3 times more frequent on test. Inspection shows decoy entities that are absent from Source 1: the
same or a perturbed name on the same street a few doors away (for example "Williams Templeton" at 2212 Leeward Shore
Dr with decoys at 2213 and 2217). `calibrate.py` keeps the holdout positives per entity in each score bin, fitted on
the top bins per country, and counts the excess test pairs as negatives. The corrected posterior replaces the raw
score (a raw 0.9 becomes about 0.5 for US).

**Remaining errors.** Half of the missed links are targets without an address whose name is shared by several
Source 1 entities, which is ambiguous even to a human. False links are mostly decoy neighbours at nearby house
numbers.

## 6. Reproduction and Fair Play

`README.md` lists every command from the raw TSVs to the validated outputs. The archive contains all source code,
pinned dependencies (`pyproject.toml`, `uv.lock`, `requirements.txt`), the transliteration dictionary, the fine-tuned
encoder and the XGBoost models, both output TSVs and this document. Raw challenge data is not included.

Only the supplied TSVs are used. No external business database, API, geocoder or data augmentation. The pretrained
weights are general multilingual text encoders under the MIT license (118M parameters, far below 8B).
