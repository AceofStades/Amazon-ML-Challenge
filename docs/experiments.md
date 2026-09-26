# Experiment log

Holdout: training Source 1 entities with hash bucket 50-59 (1%, about 22k entities), macro F0.5 including singletons.

| Date | Version | Change | Holdout F0.5 | Leaderboard |
|---|---|---|---|---|
| 2026-09-26 | v2 stage2 | Key blocking + seed expansion + XGBoost (27 string features) | 0.9244 | 0.896978 |
| 2026-09-26 | v2 stage2_deep | Deeper XGBoost | 0.9252 | not submitted |
| 2026-09-26 | v3 | Transliteration + region normalization, fine-tuned multilingual-e5-small kNN (k=40) unioned with key blocking, IDF and embedding context features, 4-fold stage 1 + stage 2 context model, expected-F0.5 decoding | 0.9813 | pending |

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
