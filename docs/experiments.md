# Experiment log

Holdout: training Source 1 entities with hash bucket 50-59 (1%, about 22k entities), macro F0.5 including singletons.

| Date | Version | Change | Holdout F0.5 | Leaderboard |
|---|---|---|---|---|
| 2026-09-26 | v2 stage2 | Key blocking + seed expansion + XGBoost (27 string features) | 0.9244 | 0.896978 |
| 2026-09-26 | v2 stage2_deep | Deeper XGBoost | 0.9252 | not submitted |
| 2026-09-26 | v3 base | Transliteration + region normalization, fine-tuned multilingual-e5-small kNN (k=40) unioned with key blocking, IDF and embedding context features, 4-fold stage 1 + stage 2 context model, expected-F0.5 decoding | 0.9813 | not submitted |
| 2026-09-26 | v3 legal | + canonical legal-form agreement/conflict features (US, Indian and French forms) | 0.9832 | 0.960932 |

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

## v3 legal test profile

Matches per entity: France 3.38 (5.3% empty), India 3.42 (5.7%), US 3.53 (5.7%).
Claimed share of targets: France 0.611, India 0.586, US 0.614 (v2 France: 0.747).
Validator PASS with ID checks: 1,732,544 rows, 97,787 empty; 108,147,874 candidate pairs.

## Test distribution shift (v3 legal)

Leaderboard 0.9609 against holdout 0.9832. Pairs per entity with p2 in [0.3, 0.98]:
US holdout 0.29, US test 0.60; India holdout 0.26, India test 0.40; France test 0.58.
The p2 > 0.98 bin is unchanged (US 3.17 holdout, 3.22 test), so the extra mass is look-alike records.

Inspection of mid-confidence test pairs: decoy entities with the same or a perturbed name
on the same street a few doors away ("Williams Templeton" 2212 vs "Williams Center" 2213,
"Williams Tempoettron Corp" 2217). Each decoy brings several records that share its own house number,
while a house-number typo on a true record stays alone. Test has about twice the decoy density of
training, so stage 2 gets explicit house-cluster features instead of relying on the training prior.

## Label-shift correction (calibrate.py, per-bin)

Holdout positives per entity per score bin are assumed to carry over to test (the top bins, p >= 0.99,
agree to within 1%: US a=1.011, India a=1.005, France a=0.957 against pooled shapes). Extra test pairs in a
bin are counted as negatives, so posterior(bin) = a * n1(bin) / m(bin), made monotone. US corrected
posteriors: raw 0.6 -> 0.22, raw 0.9 -> 0.54, raw 0.97 -> 0.76, raw >= 0.99 -> 1.0.

| Version | Matches | US mean k | India mean k | France mean k | Leaderboard |
|---|---|---|---|---|---|
| v3 legal | 5,985,926 | 3.53 | 3.42 | 3.38 | 0.960932 |
| v3 legal + per-bin correction | 5,624,024 | 3.23 | 3.34 | 2.99 | 0.968379 (rank ~403) |
