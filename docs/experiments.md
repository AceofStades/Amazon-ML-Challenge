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

## Cross-encoder (e5-small, 3.2M hard pairs from buckets >= 100)

Holdout pairs with stage-1 p >= 0.01 (101,103 pairs, 75% positive):

| Scorer | Logloss | AUC | Accuracy |
|---|---|---|---|
| v4 stage 1 | 0.0729 | 0.9943 | 0.9738 |
| cross-encoder | 0.0752 | 0.9949 | 0.9710 |
| average of both | 0.0625 | 0.9964 | 0.9806 |

Target-pair same-entity model: 15.2M labelled target pairs from buckets >= 100 (69% same entity).

## v4: full-train features, 4-fold stage 1, stage 2 with competition and house clusters

All 2.2M training entities have candidates and features (137M pairs), so stage-2 competition features and the
one-target rule see every rival Source 1 entity, as on test. Holdout (buckets 50-59):

| Scores | Decode | Holdout F0.5 |
|---|---|---|
| stage 1 | threshold 0.7, one-to-one | 0.98411 |
| stage 2 | threshold 0.7, free | 0.98666 |
| stage 2 | expected F0.5, one-to-one | 0.98707 |

Test profile: uncalibrated v4 predicts 3.80 matches per US entity (decoys); with the per-bin correction 3.20 (US),
3.30 (India), 3.08 (France). Test/holdout density ratio in the 0.74-0.92 bins is about 5 for US (v3: 2.5), so v4
separates holdout pairs better but test decoys still land mid-range.

| Submission | Holdout | Leaderboard |
|---|---|---|
| v4 + per-bin correction | 0.98707 | pending |

## Label-free test estimates (amzn.estimate)

Labels are sampled from calibrated posteriors and the submitted decisions are scored (3 samples). The
per-segment truth model splits pairs by house relation: 'equal' (the Source 1 house number appears among the
target's address numbers), 'conflict', 'unknown'. Checked against v3 calibrated (actual 0.968379): the uniform
truth model predicts 0.9753, the segmented one 0.9672, so the segmented model is used for decisions.

| Scores | Calibration | Estimated test F0.5 |
|---|---|---|
| v3 stage 2 | per-segment | 0.97175 |
| v4 stage 2 | per-segment | 0.97002 |
| v4 stage 1 | per-segment | 0.97806 |

Stage 2 context features (entity score sums, sibling support, house clusters, competition) amplify the test decoys:
the test/holdout density ratio for p2 in the 0.74-0.92 bins is about 5 for US, against 1.4-1.7 for p1.

House parsing artifact: noise prefixes such as 'Door No #834 House No 1958' made the first-number house field
disagree with Source 1 on true records, so in training 98% of 'conflicting house cluster' pairs were matches.
House relations now check the Source 1 house number against every number in the target address.

Submissions prepared: v3_segcal (5,646,694 matches), v4p1_segcal (5,676,074 matches).

## Pairwise blend: stage 1 + cross-encoder (amzn.blend)

Shallow XGBoost on (logit p1, cross-encoder logit) for pairs with p1 >= 0.01, fitted on training entities in
buckets < 100 outside the holdout (p1 out-of-fold, cross-encoder trained on buckets >= 100). Holdout logloss on
the scored pairs: 0.0504 (stage 1 alone 0.0729).

| Scores | Holdout F0.5 (in candidates) | Estimated test F0.5 (per-segment) |
|---|---|---|
| v4 stage 1 | 0.98392 | 0.97806 |
| v4 stage 2 | 0.98770 | 0.97002 |
| stage 1 + cross-encoder blend | 0.99040 | 0.98287 |

Submission blend_segcal: 5,718,944 matches; US 3.37, India 3.38, France 2.89 per entity.

## Per-country projections and the unlabelled-country fallback (amzn.hybrid)

Estimated test F0.5 per country (per-segment truth model):

| Scores | US | India | France |
|---|---|---|---|
| v4 stage 1 | 0.9727 | 0.9842 | 0.9725 |
| v4 stage 2 | 0.9777 | 0.9828 | 0.9099 |
| stage 1 + cross-encoder blend | 0.9931 | 0.9952 | 0.9180 |

France names are generic ("Lille Ecole SARL", "Roubaix Union" exist many times) and the cross-encoder and stage-2
context, both fitted on US/India where names are distinctive, over-trust name agreement there (e.g. "Roubaix Union
SARL | Rue du Curoir" for S1 at 27 Rue du Duc: p1 0.07, blend 0.88). Rule: countries present in the labelled holdout
use the blend; any other country uses stage 1; each group is calibrated against its own scorer's holdout.
Projected overall F0.5: 0.991. Submission hybrid: 5,758,810 matches.

## v6: stage 1 with address-wide pairwise house features

Seven pairwise features: Source 1 house number found among the target's address numbers (and the reverse), log gap
and digit-difference shape to the closest target number. Blend with name-genericness inputs (holdout logloss
0.0483 on cross-encoder pairs).

| Scores | US | India | France |
|---|---|---|---|
| v6 stage 1 | 0.9782 | 0.9856 | 0.9737 |
| v6 blend + name inputs | 0.9948 | 0.9958 | 0.9185 |

Hybrid v6 (blend for labelled countries, stage 1 otherwise): projected 0.9921; 5,766,877 matches.

## France noise vocabulary (v7 normalization)

Tokens most often present in a target name but absent from its Source 1 name:
- US/India true pairs: com, center, services, service, partners, lp, smt/shri/sri/mr/dr, dba/formerly/aka/fka/nee.
- France confident pairs: dotted legal-form letters (s, a, r, u, e from S.A.R.L., S.A.S., E.U.R.L.), fils, france,
  cie, groupe, associes, developpement, compagnie, frs.

The English suffixes appear in training, so the model has learned them; the French ones are unseen and push true
France records into mid scores. France's excess mid-score mass sits in the house-equal segment (p1 0.73-0.88:
0.069 per entity against 0.017 expected positives). v7 merges runs of single letters ('s a r l' -> 'sarl') and
treats cie, compagnie, fils, freres, frs, associes, groupe, developpement, pc, pllc, lp like legal forms.

## v7: renormalized names (dotted initials, French suffixes)

| Scores | US | India | France |
|---|---|---|---|
| v7 stage 1 | 0.9781 | 0.9859 | 0.9802 |
| v7 blend + name inputs | 0.9950 | 0.9955 | 0.9170 |

France stage-1 projection rose from 0.9737 (v6) to 0.9802. Hybrid v7: projected 0.9930; 5,815,669 matches
(France 3.21 per entity, up from 3.03).

Holdout loss breakdown for the v6 blend (F0.5 0.9900 including the recall ceiling): partial recall 0.0067
(1,971 entities), a false merge 0.0014, all links missed 0.0013, singleton false merge 0.0006.

## Second cross-encoder (seed 2, 600k entities, 4.8M pairs)

Holdout logloss on rescored pairs: first 0.0759, second 0.0729, logit mean 0.0724. Blend with both (plus name
inputs): logloss 0.0476 (single 0.0489), holdout F0.5 0.99028 (single 0.99007); projected US 0.9943,
India 0.9960 (single: 0.9945, 0.9961), i.e. a tie on test. Submission hybrid_v7ab: 5,812,412 matches.

## Full blend (stage-1 features + both cross-encoders, depth 6)

On pairs with stage-1 p >= 0.01 the blend also sees all stored stage-1 features (a 'stage 1.5' on the uncertain
subset). Holdout logloss 0.0466 (two-cross-encoder depth-3 blend: 0.0476); holdout F0.5 0.99046 (0.99028).

| Scores | US | India | France |
|---|---|---|---|
| full blend | 0.9948 | 0.9964 | 0.9221 |
| v7 stage 1 (France fallback) | | | 0.9802 |

A cross-encoder veto for France (min of stage 1 and blend) projects 0.9756, worse than stage 1 alone.
Hybrid v7full: projected 0.9934; 5,827,633 matches.

## Remaining holdout misses (full blend)

1,919 missed positive pairs, 1,489 of them targets without an address. The rest are mostly (a) generated trade names
at the exact address ("Solgildlum", "Zetagildveo", "Yumalum") and (b) nearby house numbers on true records (3688 vs
3690). Among different-name same-address pairs, the share of target name tokens that never occur in Source 1 names
separates matches (none: 1.3% true; all: 71% true), but the blend is already calibrated per slice (mean score equals
the match rate), and the reverse-neighbour rank splits the fully gibberish group into 94% vs 20% true. No further
holdout gain found from these signals. Depth-8 blend: logloss 0.0468, F0.5 0.9903 (depth 6 kept).

## Large cross-encoder (Kaggle, 2x T4)

intfloat/multilingual-e5-large (MIT, 560M) with frozen word embeddings, DDP fp16, 3.0M hard pairs from buckets >= 100:
6.0 h at 138 pairs/s. Validation on 30k held-out training pairs: logloss 0.0373, accuracy 0.9852. Scoring the 3.8M
pairs with stage-1 p in [0.01, 0.995) on Kaggle.
