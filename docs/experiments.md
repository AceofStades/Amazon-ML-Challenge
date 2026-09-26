# Experiment log

Holdout: training Source 1 entities with hash bucket 50-59 (1%, about 22k entities), macro F0.5 including singletons.

| Date | Version | Change | Holdout F0.5 | Leaderboard |
|---|---|---|---|---|
| 2026-09-26 | v2 stage2 | Key blocking + seed expansion + XGBoost (27 string features) | 0.9244 | 0.896978 |
| 2026-09-26 | v2 stage2_deep | Deeper XGBoost | 0.9252 | not submitted |
| 2026-09-26 | v3 base | Transliteration + region normalization, fine-tuned multilingual-e5-small kNN (k=40) unioned with key blocking, IDF and embedding context features, 4-fold stage 1 + stage 2 context model, expected-F0.5 decoding | 0.9813 | not submitted |
| 2026-09-26 | v3 legal | + canonical legal-form agreement/conflict features (US, Indian and French forms) | 0.9832 | pending |

## Candidate recall (training sample, 458,727 positive pairs)

| Candidate source | Pair recall | Pairs |
|---|---|---|
| Key blocking (new normalization, top 30) | 0.9337 | 3.83M |
| Encoder kNN top 10 | 0.9915 | 1.32M |
| Encoder kNN top 40 | 0.9972 | 5.30M |
| Union | 0.9974 | 8.36M |

## Leaderboard gap

v2 test predictions per Source 1 entity: France 4.13 matches (4.0% empty), US 3.18 (6.1%), India 2.94 (9.1%).
France has no training labels; generic names and few cities suggest false merges drive the gap.

v2 claimed share of test target records: France 0.747, US 0.553, India 0.504. Training-like cluster sizes
(3.46 matches per entity) imply about 0.62 for France, so at least 170k France claims are false merges.
France Source 1 holds many same-name entities at different addresses ("Dunkerque Ecole SARL" x5+), so the
address decides. Rival records share the house number but differ in street and legal form.

## Remaining holdout errors (v3 base)

US 0.9814, India 0.9812. Loss: partial recall 0.0092, entity fully missed 0.0037, false merge 0.0045,
singleton false merge 0.0014. Half of the false-negative pairs have no target address, and 84% of those
share their name with another Source 1 entity, so they are ambiguous. No row-order or ID leakage exists
(correlation of file positions of matched records is 0.001).
