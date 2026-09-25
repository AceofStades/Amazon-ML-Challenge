# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Claude's Plan

**Team Members:** Rahul Bokade (leader), Hrishita Chavan, Riddhesh Budha, Umed Indulkar

**Submission Date:** 2026-09-26

## 1. Executive Summary

Our pipeline uses bounded, multi-key blocking followed by an XGBoost pair classifier. It processes the supplied training and test records without external entity lookup. On a deterministic holdout of 22,144 Source 1 training entities, it obtains macro F0.5 of 0.884862 while sending at most 10 candidates per Source 1 entity to the classifier.

## 2. Methodology

### 2.1 Problem Analysis

The training set contains 2,206,821 Source 1 records and 10,320,219 Source 2/3 records. The test set contains 1,732,544 Source 1 records and 9,969,589 Source 2/3 records. Exhaustive comparison is infeasible. Labeled pairs show Latin, Devanagari, and Tamil name variants, word-order changes, character errors, legal-suffix changes, address reordering, abbreviation, and missing fields. The test set also contains France, so country is treated as an open string label.

### 2.2 Solution Strategy

We first normalize names and addresses with case folding, transliteration, punctuation cleanup, and common address-term expansion. We then build exact and token-based blocking keys, discard keys with large target posting lists, and rank candidate pairs with cheap name and address similarities. XGBoost scores the final bounded candidate set. A validation split chooses the classification threshold and candidate cap.

## 3. Candidate Generation

Blocking keys use canonical names, canonical addresses, name/address token pairs, house/address token pairs, house/name token pairs, and individual rare name or address tokens. Target posting-list limits vary by key type, from 60 for name-only tokens to 500 for exact canonical names. DuckDB joins key tables in parallel. Candidate ranking combines name and address Jaro-Winkler similarity, house-number agreement, strongest blocking key, and number of matching keys.

The model scores the top 10 candidates per Source 1 entity. The final test candidate set contains 17,067,625 pairs, or 9.85 candidates per Source 1 entity on average. This is about a millionfold reduction from exhaustive cross-source comparison. Pair-level blocking recall at 10 is 0.869125 on a deterministic 6% Source 1 training sample. At 30 candidates, recall is 0.891657. The `candidate_pairs.tsv` file contains exactly the candidates scored by the final model.

## 4. Matching Model

The classifier is XGBoost 3.4.1, distributed under Apache-2.0. It uses GPU histogram training, maximum depth 8, learning rate 0.07, and early stopping on held-out Source 1 entities. Best iteration was 495. The 18 pair features include name and address Jaro-Winkler similarities, canonical exact matches, token-anchor overlap, house-number match/conflict, length ratios, missingness, blocking evidence, candidate rank, and target source.

Training uses approximately 5% of Source 1 training entities. A disjoint approximately 1% Source 1 sample tunes the decision threshold and candidate cap against macro F0.5. Selected threshold is 0.575 and cap is 10. The split uses a stable DuckDB hash of the Source 1 entity ID. Every Source 1 entity in the holdout contributes to macro F0.5, including entities with no true matches.

## 5. Results and Error Analysis

On 22,144 held-out Source 1 entities, macro F0.5 is 0.884862. Link precision is 0.957443; link recall is 0.804489. The model correctly predicts no matches for 1,005 of 1,209 true singleton entities (0.831266 accuracy). With candidate caps of 5, 10, 15, 20, and 30, tuned macro F0.5 scores are 0.879026, 0.884862, 0.885867, 0.886162, and 0.886801 respectively. The cap of 10 gives a much smaller candidate set for a 0.001939 absolute F0.5 difference from cap 30.

False negatives mainly arise when transliteration or severe character errors remove shared blocking tokens, or when a shortened address loses discriminative tokens. Common business names without an address remain ambiguous. False positives tend to involve repeated names or nearby address strings that appear across distinct businesses. Candidate generation, not the classifier, sets the recall ceiling; further blocking research is the main improvement path.

Both final TSV files passed the supplied submission validator with full test ID checks. Each has exactly 1,732,544 Source 1 rows. `matching_results.tsv` has 125,440 empty match lists; `candidate_pairs.tsv` has 626 empty candidate lists. Every emitted match appears in the corresponding candidate list.

## 6. Reproduction and Fair Play

The archive contains code, pinned requirements, uv lockfile, trained model artifact, both outputs, and this document. The README gives exact commands from raw TSV files through normalization, blocking, training, inference, and validation. The raw supplied dataset is excluded from the archive and Git. The pipeline uses only the challenge data; it makes no business identity lookup or external data augmentation request.
