# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Claude's Plan

**Team Members:** Rahul Bokade (leader), Hrishita Chavan, Riddhesh Budha, Umed Indulkar

**Submission Date:** 2026-09-26

## 1. Executive Summary

Our solution uses two bounded candidate-generation passes and two GPU-trained XGBoost pair classifiers. The second pass uses high-confidence links from the first classifier as anchors for finding additional Source 2/3 variants. A final classifier scores the exact union recorded in candidate_pairs.tsv. The deterministic holdout macro F0.5 is 0.924402 on 22,144 Source 1 entities. The pipeline uses only the supplied TSV data.

## 2. Data and Preparation

The training set has 2,206,821 Source 1 records and 10,320,219 Source 2/3 records. The test set has 1,732,544 Source 1 records and 9,969,589 Source 2/3 records. Exhaustive comparison is infeasible. Training records show transliteration, character substitutions, inserted or deleted words, legal-suffix changes, reordered addresses, extra house numbers, and missing fields.

Preparation streams the files with explicit tab delimiters. It lowercases and transliterates text, removes punctuation, expands common address abbreviations, and derives canonical sorted name/address strings, numeric house fields, and discriminative name/address token anchors. Country is normalized as an open string label and used in blocking keys. There is no US/India-only model branch or filter; test entities from France remain eligible.

## 3. Candidate Generation

The first blocker indexes targets by exact canonical name/address, pairs of name anchors, name/address token combinations, house/name and house/address combinations, rare single tokens, and prefix-based keys resilient to some misspellings. Keys with large target posting lists are suppressed, with caps ranging from 40 to 1,000 depending on specificity. DuckDB joins the keys and ranks pairs by raw name/address Jaro-Winkler similarity, house agreement, strongest key, and number of matching keys. It keeps at most 30 targets per Source 1 record.

The first classifier scores those pairs. Links with predicted probability at least 0.9 become expansion seeds. On the disjoint validation split, these seeds had 99.35% link precision. Each seed's rare indexed keys (target frequency at most 40) retrieve other target records. The second blocker ranks them using similarity to both the seed and original Source 1 record, excludes first-pass pairs, and keeps at most 30 additional candidates per Source 1 record. The final candidate set is the union of both passes; every pair in it is scored by the final classifier and every predicted match appears in it.

On a deterministic 6% training sample, the first 30-candidate pass retrieves 409,740 of 458,727 known links (0.893211 pair recall). Expansion retrieves 14,392 additional links, raising candidate recall to 0.924587. The combined set has 7,053,012 pairs across 132,455 Source 1 records, averaging 53.25 candidates per Source 1 record. This cuts roughly ten million possible comparisons per Source 1 entity to a few dozen.

## 4. Matching Model and Features

Both classifiers use XGBoost 3.4.1 (Apache-2.0 license), with GPU histogram training, depth 8, learning rate 0.07, subsampling, and validation-based early stopping. They use 27 numeric pair features: raw and canonical Jaro-Winkler, character-bigram Jaccard, token Jaccard, normalized edit similarity, name/address exact match, anchor overlap and anchor similarity, house match/conflict, address-number overlap, field length ratios, missingness, target source, and blocking rank/evidence.

The deterministic DuckDB hash split places roughly 5% of training Source 1 entities in model fitting and a disjoint roughly 1% in validation. It does not split individual candidate pairs from one Source 1 entity across fit and validation. Ground-truth matches include both Source 2 and Source 3 IDs; negatives come from the hard candidates produced by blocking. Threshold and candidate cap are tuned for macro F0.5, counting empty-truth singleton entities exactly as the challenge specifies. The first pass uses a 0.9 seed threshold for high precision. The final model selected a 0.65 decision threshold and up to 60 combined candidates.

## 5. Validation and Error Analysis

The final model achieves macro F0.5 of 0.924402 on 22,144 held-out Source 1 entities. The corresponding first-pass model achieved 0.919435 with at most 30 candidates. At combined candidate caps 40, 50, and 60, final-model validation macro F0.5 is 0.924000, 0.924264, and 0.924402. The small extra gain from cap 60 is retained because leaderboard F0.5 is the primary scored criterion; the candidate count remains small relative to exhaustive search.

Remaining false negatives occur when severe transliteration or abbreviated data removes both direct blocking keys and seed-neighbor keys, or when a recovered candidate's Source 1 similarity is too weak for the final model. Repeated business names and incomplete addresses can produce false positives. The 0.9 seed threshold favors precision so errors do not cascade through the expansion pass. France is unseen in the training labels and may cause validation-to-leaderboard shift.

## 6. Reproduction and Fair Play

The README gives exact uv/Python 3.12 commands from the original TSVs through normalization, both training passes, full-test inference, validation, and packaging. The archive contains code, pinned requirements, uv lockfile, both trained model artifacts, both required output TSVs, and this document. Raw challenge data is excluded from the archive and Git. No external business-identity database, API, geocoder, or augmentation service is used.
