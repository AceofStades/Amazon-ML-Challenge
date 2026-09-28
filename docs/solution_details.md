# Amazon ML Challenge 2026: solution details (team Claude's Plan)

Final leaderboard submission: 0.989389 public macro F0.5 (file 18). Full write-up: `Documentation_template.md`;
every command to reproduce it: `code/business_entity_resolution/README.md`.

## 1. Methodology used

A cascade that links each Source 1 record to its Source 2/3 records:

1. Normalization: a transliteration dictionary learned from matched training pairs (Devanagari, Bengali, Tamil and
   six other scripts to Latin, 97% of native-script test tokens covered), country-aware address abbreviations, one
   canonical token per state/region/department, merged dotted legal forms (S.A.R.L. -> sarl), house-number parsing.
2. Candidate generation (section 2).
3. A pairwise XGBoost matcher (stage 1) over 67 features, with out-of-fold scores for every training pair.
4. 29 fine-tuned multilingual cross-encoders rescore the uncertain pairs (0.01 <= stage-1 p < 0.995), and an XGBoost
   blend combines them with the stage-1 score, the features and full-competition target context.
5. Label-shift correction: the test set has about twice the look-alike decoys of the training data, so posteriors
   are recalibrated per country, house relation and score bin without test labels.
6. Decoding: each target record goes to at most one Source 1 entity, then for every entity the candidate prefix
   with the highest expected F0.5 is predicted (an empty list included, so singletons are handled explicitly).
   A composition model rescores ambiguous pairs with the entity's other matches.
7. France (no training labels): stage-1 scores with the same calibration, plus rules derived from training-label
   statistics about how the data generator builds true records and decoys (section 4).

Validation: 10-fold cross-validation over 220k training entities (macro F0.5 with singletons, 0.99327 in a
test-like view) and a fixed 22k-entity holdout; every experiment is logged in `docs/experiments.md`.

## 2. Candidate generation / blocking strategy

Union of two generators, run per country (country is an open label):

- Rare-key blocking (DuckDB): exact canonical name or address, name-token pairs, name x address and house x address
  token keys, prefix keys; keys with long posting lists are dropped; the 30 best targets per Source 1 record are
  kept by Jaro-Winkler and key evidence. Recall 0.934 alone.
- Bi-encoder retrieval: `intfloat/multilingual-e5-small` (MIT) fine-tuned with an in-batch contrastive loss on 1.5M
  matched pairs (hard negatives from same-prefix/same-city batches); exact cosine top-40 per Source 1 record on GPU,
  plus a reverse top-5 search that feeds competition features. Recall 0.997 alone.

Union: pair recall 0.997 on 458,727 training links, about 62 candidates per Source 1 record (108M test pairs out of
about 17 trillion possible). `candidate_pairs.tsv` is exactly this set: the stage-1 matcher and the final blend score
every pair in it. The cross-encoders run only on the 3.4M pairs stage 1 leaves uncertain.

## 3. Model architecture and feature engineering

- Stage 1: GPU XGBoost (depth 9, 4 folds) over 67 pair features: raw/canonical Jaro-Winkler, Jaccard and edit
  similarities; IDF-weighted token overlap (cosine, coverage, rarest shared/missing token); legal-form agreement;
  house-number match/conflict/edit distance and address-wide number evidence (handles "Door No #834 House No 1958");
  missing fields; name ambiguity counts; embedding cosine and ranks; reverse-neighbour competition; sibling agreement
  among an entity's candidates.
- Cross-encoders (all MIT/Apache-2.0, at most 568M parameters): multilingual-e5 small/base/large/large-instruct,
  bge-reranker-v2-m3, bge-m3, LaBSE, xlm-roberta-large. Mean-pooled encoder + linear head, trained on entities
  disjoint from the blend's, on neighbour pairs, "hard" stage-1 pairs or boundary pairs; large models trained on
  Kaggle 2x T4 (DDP, fp16), small ones locally.
- Blend: XGBoost (depth 6) over logit(stage-1 p), 29 cross-encoder logits, the stage-1 features, name-genericness
  and generator-aware name-change descriptors (exact, added/dropped words, typo, word swap by vocabulary class,
  acronym, domain, alias, made-up brand).
- Full-competition target context: the pair's stage-1 score renormalized over every candidate entity of the target,
  their number, the best competitor and raw-name agreement (CV F0.5 0.99167 -> 0.99261).
- Composition rescoring: ambiguous pairs rescored with the entity's confident matches (per-source counts,
  address-less records, noise kinds, fit to the training composition, competing claims).

## 4. Other relevant information

- Test shift: mid-score pairs are 2-5x more frequent on test, almost all "a few doors away" decoys; the per-segment
  prior-shift correction handles them without labels.
- France rules (each from a training statistic, applied only where labels are missing): restore acronyms, domains,
  aliases and near-typos at the Source 1 address; drop nearby house-number conflicts whose name adds words (3% true)
  and restore identical names moved more than 30 numbers (99.7% true); drop same-house swaps to common vocabulary
  words (about 23% true); add pairs the blend scores above 0.98 outside decoy classes; composition decides
  address-less owners; add same-house made-up brands the blend scores >= 0.95.
- Label-free diagnostics used for France: per-entity link budgets by category, a displacement test (true noise
  replaces an entity's record, decoys add one) and a Source 2/3 balance test, each calibrated on training truth.
- Remaining error: mostly address-less records whose name several entities share; the owner is not identifiable
  from the data.
- Fair play: only the supplied TSVs; no external data, APIs or geocoders; all models MIT or Apache-2.0 and far below
  8B parameters.
