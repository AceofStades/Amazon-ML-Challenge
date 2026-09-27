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

## Leaderboard: hybrid v7 full

Leaderboard 0.984806 (rank 71; top 0.990556) against a projection of 0.9934. US/India top-bin counts on test match
holdout (ratios 0.995-1.007) and cross-encoder promotions barely exceed holdout, so the gap most likely sits in
France (back-solved about 0.92 if US/India score as projected). France predictions per entity: 3.20 against 3.39 for
US/India, suggesting the pooled calibration over-corrects France.

## One-to-one tie bug (reported by a teammate)

Calibration maps many top scores to exactly 1.0, and the rule "zero every score below the target's maximum" kept
all tied Source 1 entities. hybrid v7 full assigned 1,420 targets to more than one entity (1,477 surplus pairs:
France 107, India 788, US 582). Fixed: one winner per target, ties broken by raw score, then Source 1 row.

## Large cross-encoder in the blend; France calibration variants

| Blend | Holdout logloss | Holdout F0.5 | US | India |
|---|---|---|---|---|
| full + two small cross-encoders | 0.0466 | 0.99046 | 0.9948 | 0.9964 |
| + third small cross-encoder | 0.0461 | 0.99052 | | |
| + large cross-encoder (uncertain pairs) | 0.0443 | 0.99094 | 0.9955 | 0.9973 |

France predictions per entity by calibration mode: segment 3.20, uniform 3.22, none (raw stage 1) 3.39. Pairs added
by 'none' are mostly acronyms, domains, typos and generic-word swaps at the same address (estimated 70-80% true).
Submissions: hybrid_L_frnone (5,875,982 matches), hybrid_L_frsegment (5,827,268).

## Calibration refined by name agreement (rejected)

Splitting each house-relation segment by name agreement (IDF cosine >= 0.95) leaves sparse cells: bins with few test
pairs get posterior 1.0 after clipping and the monotone pass, e.g. US conflicting-house different-name pairs reach
1.0 from mid scores, the opposite of the decoy pattern. Kept the house-relation segments.

## Additional cross-encoders in flight

Second large cross-encoder (seed-21 entities) on Kaggle; e5-base cross-encoder (MIT, 278M, seed-4 entities) on the
local GPU at 368 pairs/s.

## Leaderboard: large cross-encoder blend and France decisions

| Submission | US/India scorer | France decisions | Leaderboard |
|---|---|---|---|
| hybrid v7 full | stage 1 + 2 small cross-encoders | stage 1, segment-calibrated (tie bug) | 0.984806 |
| hybrid L, France raw | + third small + large cross-encoder | stage 1, uncalibrated | 0.983476 |
| hybrid L, France calibrated | + third small + large cross-encoder | stage 1, segment-calibrated | 0.985292 (rank 77) |

The large cross-encoder plus the tie fix is worth +0.0005. Raw France decisions (+52k pairs, 0.2 per entity) cost
0.0018 overall, i.e. about 1.2 points of France F0.5, so the extra pairs were only about half true.

## France recall rule (france_rule.py)

Keep the calibrated France decisions and add back a raw-only pair only at the Source 1 address (no house conflict)
when the target name is an acronym of the Source 1 initials, a web domain carrying a Source 1 word or initials,
a DBA/f.k.a. alias, a near-typo (compact Jaro-Winkler >= 0.92) or the Source 1 core name plus extra words.
Adds 18,399 pairs (acronym 6,653; domain 2,986; typo/containment 8,726; alias 34), which read as nearly all true.

## France audits (no labels)

Accepted France pairs per entity by house relation: equal 2.89, conflict 0.044, unknown 0.27 (US: 2.70 / 0.33 / 0.36).
Accepted different-name France pairs are overwhelmingly generator noise on true records (domains, DBA prefixes,
acronyms, typos). Same-name clusters are split correctly across same-name Source 1 entities by address, and
same-name records at other numbers stay unassigned. The optional --drop-conflicts rule would remove 9,321
accepted France pairs with conflicting house numbers; a sample mixes decoys with true records whose number was
randomized, so it is not part of the default output. Candidate recall for exact name+address pairs is 100% in
every country.

## Covariate-shift adaptation for France (shelved)

A domain classifier separates France test pairs from training pairs almost perfectly (top features: name duplicate
counts, neighbour-similarity profile, address similarity), so most training pairs get the minimum weight. The
weighted stage-1 refit reached holdout logloss 0.061 by round 1000 (worse than the unweighted model) and then
crashed twice with SIGSEGV inside XGBoost 3.4.1 CPU histogram building with sample weights. Shelved in favour of
the France recall rule, which needs no retraining.

## House conflicts: US versus France

Accepted US pairs with conflicting house numbers are mostly dropped-digit typos on true records (601 vs 60,
15911 vs 5911, 1203 vs 203). Accepted France conflicts are unrelated numbers (129 vs 445, 8 vs 165, 30 vs 289),
the decoy signature; treating one dropped digit as a typo changes only 110 of 9,321 France drops. Candidate
submission: calibrated France + recall rule + drop of non-typo France conflicts (5,836,456 matches).

## France error analysis without labels

- Every labelled country has the same match count: 3.46 true matches per Source 1 entity (US 3.459, India 3.465,
  5.6% singletons). The hybrid predicts 3.39 per entity for US and India but only 3.20 for France.
- France true records are cleaner than training ones: 1.85 identical-name, same-address targets per entity against
  1.14 in training, with fewer typos and added words.
- France rejects 0.30 pairs per entity with raw p1 >= 0.5, mostly same-address records whose name has one word
  swapped (0.078) or an unrelated name (0.089).
- Training labels by how the name changed, same house number and street, one word replaced in place:
  by a filler (center, services, service, partners) 98.7% true; by a rare or garbled word 95.9% true;
  by a common vocabulary word (20+ Source 1 names) only 45.3% true, and 5.1% at another house number.
- France gray swaps replace a generic type word by another (club, ecole, comite, amicale, sportive): the vocabulary
  of the France decoys a few doors away. Confident France swaps use services and france instead.
- Unrelated names at the same address: acronyms that spell the Source 1 initials are 99.9% true in training and
  12 times more frequent in France, where stage 1 gives them only 0.75; single made-up brand words are 80% true in
  training but 2.4 times over-represented in France, so most France ones are decoys.
- The model has no feature for the kind of word change, so gen_features.py adds them: exact/add/drop/typo/swap
  (in place or moved), the swapped-in word's class (filler, common, rare), log document frequencies of missing and
  extra words per country, acronym/domain/alias with initials consistency, and single-brand names.

## Generator-aware features, France swap and initials rules

- gen_features.py adds name-change descriptors to the blend (--extras gen). Holdout F0.5 (same decode, in
  candidates) 0.99169 against 0.99160 for blend_v7_fullL: the patterns are rare in training.
- A no-CE blend with gen features as the France scorer was rejected: the France segment calibration maps its
  0.96 scores for exact same-address names to about 0.6, dropping about 4k true pairs, plus 4.4k true
  "drop a type word, add a suffix" pairs ("Mes Sportive SARL" vs "Mes SARL Et Fils").
- France rule --drop-swaps: drop accepted France pairs whose name replaces one word by another common word
  (20+ France Source 1 names, not a filler such as services/france, not a typo or abbreviation of it).
  Evidence: in training, decoy swaps land on the Source 1 house 0.42 times as often as on another house of the
  same street; France has 22,445 same-house and 40,987 other-house common-word swaps, so only about 23% of the
  same-house ones are true (club 18%, ecole 21%, comite 28%). Drops 9.3k pairs.
- France rule --add-initials: add every scored candidate whose name is an acronym or web domain spelling the
  Source 1 initials or words, on the Source 1 street without a house conflict, target not yet assigned
  (99.9% true in training; France has 12 times more of them and stage 1 scores them at a median 0.35).
  Adds 6.6k pairs (5.8k acronyms).
- Holdout check of the same drops on US/India: the CE blend already accepts only true last-digit house changes
  (precision 1.000) and true swaps (0.991), so no rule is applied there. Stage-1 holdout decisions keep
  common-word swaps at 0.88 precision, above the F0.5 break-even of about 0.77, but France swaps are not
  comparable (vocabulary shared with France decoys).
- House-number differences in training, same street and same name with a conflict: last digit only,
  difference 1-2: 42.5% true; 3-10: 13.5% true; inner digit or dropped digit: 82-99% true (typos).
- Submission files: 6 = best + France patterns - conflicts - swaps + initials (5,833,759 matches);
  5 = without initials; 5b = without conflict drops.

## More cross-encoders

Six Kaggle GPU sessions (three accounts, two each) plus hades:
aceofstades L2 (e5-large, seed 21) and XLM-R large on hard pairs (seed 61); rekishu e5-large (seed 31) and
e5-large on hard pairs (seed 41); hrishitachavan bge-reranker-v2-m3 (Apache-2.0) on kNN pairs and on hard pairs
(seed 51); hades e5-base (kNN pairs, done) and e5-base on hard pairs (seed 71). Hard pairs are the stage-1
candidates with p1 >= 0.01 of 600k entities outside the blend buckets, plus two kNN neighbours and all true
matches (2.83M pairs, 73% positive): the region where the blend uses cross-encoder scores.

## e5-base cross-encoder and gen features in the blend

Holdout F0.5 (in candidates): blend_v7_fullL 0.99160; + e5-base (fullLB) 0.99174; + gen features (genLB) 0.99186.
File 7 = genLB for US/India + file-6 France rules (US -5.0k/+1.8k pairs, India -3.0k/+2.4k vs file 6).
cyclop5 Kaggle account: kernels ran with 0 GPUs and no internet (phone verification missing).

## Leaderboard: France rules and gen blend

| File | Change | Leaderboard |
|---|---|---|
| 6 | best + France patterns - conflicts - swaps + initials | 0.987207 (rank 47) |
| 7 | 6 + gen-feature blend with e5-base for US/India | 0.987379 (rank 44) |

The France rules added +0.0019 overall (about +0.013 France F0.5); the new blend +0.00017.
Top team 0.990788.

Checked and rejected on the holdout or by counts: topping up entities that have one or two accepted matches
(best rejected same-address candidate is 45-73% true), blend tuning (depth 8, eta 0.05, 5-model bagging:
0.99166-0.99173 vs 0.99186), a no-address calibration segment (curve unchanged), dropping France brands
(France accepts only 553 pure made-up brands). Accepted pairs per entity by house relation match the holdout
for US and India, so the extra test decoys are already kept out there.

## Diagnostic: France emptied (file 9)

File 7 with every France entity predicted empty scored 0.851 (three decimals). With France singletons (about 5.6-6.1%)
still counting as correct, US/India F0.5 is about 0.990-0.991 and file 7 France F0.5 about 0.966-0.971.
The holdout F0.5 used so far (0.9919) counts only links inside the candidates (candidate recall 0.9974), so the
comparable full holdout value is about 0.9909: US/India test performance matches the holdout.

France checks after file 7: same-name same-address targets are always candidates (recall proxy); missing
same-address targets are unrelated businesses sharing the address. France accepts below calibrated 0.99 (33k)
are mostly exact names and domains/acronyms at the Source 1 house, so no precision push. France pure brands at
the same address: 0.0895 per entity against 0.034-0.038 in US/India test and 0.0385 in training (77% true),
about 33% true in France and inseparable. A no-address calibration segment changes nothing material.

## Kaggle GPU quota plan (2026-09-27 02:20)

The weekly quota is 30 GPU-hours per account and a 2x T4 session counts twice. Two concurrent 7.5-hour runs would
use the whole week with nothing left for scoring, so every second run was deleted and re-pushed with a 3.75-hour
budget (kernels 21-27; rekishu and tiya train boundary specialists on stage-1 pairs with 0.02 <= p1 < 0.98).
Per account: first run about 15.6, deleted run about 1.5, two scorings about 3.4, second run about 8.1 GPU-hours.
The user account keeps only the XLM-R scoring (about 10 GPU-hours were left).

## Cross-validated blend comparison (10 folds over 220k entities)

The 22k-entity holdout cannot separate blends that differ by 0.0002, so scripts/blend_cv.py fits each blend on
nine groups of ten buckets and predicts the tenth, over all training entities in buckets 0-99.

| Blend | CV F0.5 | Mean fold logloss |
|---|---|---|
| fullL (file 6) | 0.99115 | 0.04465 |
| genLB (file 7) | 0.99121 | 0.04418 |
| auto2 = genLB + hard-pair e5-base + second large e5 | 0.99130 | 0.04383 |

auto2 is at least as good as genLB in 9 of 10 folds.

## France calibration without decoy-heavy classes (segment-clean)

Fitting the France posterior curves without one-word common swaps and single made-up brands raises the equal-house
curve (for example 0.29-0.64 becomes 0.48-0.86 in the upper-middle bins) and adds 3.9k France pairs. Only the 2.4k in
classes that are true noise in training (drop plus suffix, filler swap, exact, domain, typo, acronym) are kept
(scripts/merge_france.py); brands and multi-word swaps among the additions look like decoys.

## Holdout error audit and composition rescoring

Best blend (auto2) on the 22k holdout: 1,836 false negatives and 109 false positives. 78% of the false negatives
are targets without an address whose name is shared by several Source 1 entities (up to 283), which is ambiguous.
No leakage: row order and ids are random (median gap between matches of one entity is 1.09M rows).

Generator structure: records of an entity are spread over both sources (an entity with three records has none in
Source 2 only 5.9% of the time, none for six or more), 13% of entities have one record without an address and 1%
have two; at most one alias per entity (8.0% vs 0.00%), acronym (0.7% vs 0.01%), and rarely two brands or domains.
scripts/compose.py rescores ambiguous pairs (0.01 <= p < 0.99) with features from the entity's confident matches
(source counts, no-address count, kind counts, fit of the extra record under the training composition, competing
claims on the target). Cross-validated over 220k entities: logloss on ambiguous pairs 0.2254 -> 0.2186, macro F0.5
0.99130 -> 0.99150. Applied to US/India only: on France it removes true suffix/filler pairs.

Blend-scored France (with segment-clean calibration) drops 24.5k true "drop plus French suffix" pairs: the
cross-encoders do not know the French suffixes, stage 1 does through normalization.

## Fleet handoff verification (2026-09-27 04:05 IST)

The user confirms that file 10 is unsubmitted. Its leaderboard score is unknown,
and all five daily submissions remain available based on the handoff and this
confirmation. The expected score is still only an estimate.

Host process checks confirm that the twelve remaining Kaggle chains, the
automatic fleet watcher, and both local GPU chains are alive. Repeated chain
command lines are parent and child shells in the same session, not duplicate
watchers. Kaggle confirms that XLM-R 7, bge-m3 9, and LaBSE 14 have completed
training and scoring, with scores downloaded. The other twelve training kernels
remain running. Nine additional cross-encoder directories have both score files.

The automatic auto4 blend reports holdout F0.5 0.99188 and logloss 0.00395,
compared with 0.99186 and 0.00402 for genLB. These are the candidate-conditioned
holdout values, not full CV or leaderboard scores. The second local hard-pair
e5-base model is scoring test pairs; its boundary successor continues waiting.
The root disk has 64 GB free. No training or scoring kernels were stopped.

The automatic and final assembly scripts now share a file lock to prevent
overlapping heavy pipelines. Their validator runs preserve failure exit codes;
the previous pipe into `tail` could falsely report a successful assembly.
Isolated fixtures verify that both scripts wait for the lock and return the
validator's nonzero exit code. The noon runner fixture verifies that validation
failure does not stage a file and that valid output is staged as a hardlink.

## Submission timing policy (2026-09-27 04:15 IST)

The user prefers waiting for the Kaggle fleet before the first submission. File
10 remains unsubmitted and is renamed with a HOLD label. Its expected score is
unmeasured. All five daily slots remain available under the recorded submission
history; keep them for a completed-fleet baseline, controlled France changes, and
the final selection.

File 10 changes the cross-encoder blend, composition rescoring, and France rules
together. Its leaderboard delta cannot isolate composition. A leaderboard check
of composition requires otherwise identical fleet variants.

The scheduled noon build checks readiness from 11:00 IST and waits until all
fleet and remaining local score files are readable, with a 12:00 IST cutoff for
drafting with available results. No submission occurs automatically. The current
session runs directly on hades, as confirmed by the host name.

Seven small readiness tests cover complete files, missing train/test files,
partial parquet files, local models, comments, malformed configuration, and a
missing fleet definition. Runner fixtures verify that configuration and
validation failures do not stage a submission, that the cutoff drafts with
available scores, and that a push retry preserves the completed build and
hardlink. The live readiness check reports twelve Kaggle models and both local
models pending.

## Exact expected F0.5 decoder (2026-09-27)

The current decoder approximates the random true-count denominator with its
mean and also multiplies by the probability of a nonempty truth set. That extra
factor can change the optimal choice: for one candidate with probability 0.55,
predicting the link has expected F0.5 0.55, versus 0.45 for an empty prediction,
but the existing approximation predicts empty.

An optional `expected-exact` mode integrates the independent Bernoulli
true-count distribution using Gaussian quadrature. For F0.5 the integrand is a
polynomial, so the selected quadrature order is exact up to floating-point
error. Six small tests compare it with explicit enumeration of every outcome,
cover zero/one probabilities and empty input, and verify the existing decoder
dispatch. The existing approximate mode remains the default pending holdout
evaluation. Independence and probability calibration remain modelling assumptions.

The experiment command is:
`flock logs/blend.lock env OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/decode_compare.py`.
The comparison uses held-out source buckets 50-59, global stage-1 target
ownership over all training sources, all ground-truth links including links
missing from candidates, and singleton entities. Results are recorded in
`output/decode_exact_holdout.json` and `logs/decode_exact_holdout.log`.

## Deferred recovery of local cross-encoder scoring (2026-09-27)

The second hard e5-base model saved its checkpoint and completed 1,038,797
training scores. Its original batch-512 test scorer aborted with
`free(): invalid pointer` after logging 4,864,000 of 8,831,732 pairs. No test
parquet or intermediate checkpoints were saved. CUDA allocation OOM warnings
also occurred during train-pair scoring; these suggest memory pressure but do
not establish the native crash's root cause. The boundary training chain remains live.

A deferred recovery watcher waits for the captured boundary-chain process,
every cross-encoder GPU worker, and GPU memory usage below 1,024 MiB. It
preserves complete score files, recovers hard2 first, and then recovers missing
or corrupt boundary scores if its final checkpoint exists. The coordinator
scores 250,000-row chunks in fresh processes at batch size 128, validates row
order and finite, non-null logits, and records immutable model/pair SHA-256
identities. Completed chunks survive failures; incompatible manifests are
rejected; the complete output is published atomically.

Fourteen CPU fixture tests cover scoring, crash/resume, corrupted chunks,
identity drift, invalid logits and deferred-launch guards. Shell syntax and
Python compilation pass. Real GPU recovery remains unverified until the
boundary chain releases the GPU. No existing training or running shell changes.

## Exact decoder holdout result: rejected (2026-09-27)

On 22,144 held-out entities with global stage-1 target ownership and full
ground-truth denominators, the existing decoder scores 0.984830851671 and the
exact independent-Bernoulli decoder scores 0.984272355674: a decrease of
0.000558495997. Existing decisions have 73,930 true positives and 327 false
positives; exact decisions have 73,902 true positives and 342 false positives.
The mathematical calculation is correct under its assumptions, but those
assumptions do not improve these model decisions. Retain the existing decoder
for submissions. The optional exact mode remains available for reproducible
experiments; it is not enabled in any submission pipeline.

## Addressed France rejection audit (2026-09-27)

`scripts/france_audit_rejected.py` joins the current France decisions with
stage-1 scores, address features and generator-aware name classes. It records
unassigned candidates without changing a submission. Run with
`flock logs/blend.lock env POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python scripts/france_audit_rejected.py`.
The baseline is `output/sub_auto2_frsafe/matching_results.tsv`; the detailed
artifact is `output/france_audit_rejected.parquet` and the run log is
`logs/france_audit_rejected.log`.

There are 701,331 unassigned candidate pairs passing the address filter. Of
these, 543,720 have p1 below 0.01 and no cached generator classification.
Among classified candidates, 45,150 are brands, 43,323 common-word swaps,
24,230 exact core names, 19,572 multiple-word changes, and 4,635 dropped-word
names. Exact names have median p1 0.1434 and only 85 exact-address pairs;
nearby-number decoys make indiscriminate restoration unsafe. Review instead
found false house conflicts caused by number prefixes, address token order,
street-type artifacts, postal codes and apartment numbers. These narrow
parser hypotheses are evaluated separately. Python compilation passes.

## Address-equality parser restoration (2026-09-27)

`scripts/france_audit_restore.py` evaluates exact normalized address token sets
after repairing the literal French `ndeg` number prefix. A broader comparison
allows the target's numeric tokens to be a subset of the source's numbers
when nonnumeric address tokens agree. Both restrict names to exact, drop,
typo, filler or added-word classes; rare replacements must preserve a source
word and cannot be single made-up brands. Training labels are joined directly
to out-of-fold stage-1 candidate pairs.

Run `train` to build `output/france_audit_train_parser.parquet`, then `report`
to reproduce `logs/france_audit_restore_train_report.log`. Run `test` against
the rejection artifact with `--min-p 0.9` and optionally
`--matching-output output/france_parser_eq/matching_results.tsv` to draft a
comparison. Use the shared pipeline lock and two CPU threads for each command.

At p1 at least 0.9, exact repaired-address pairs are 99.93744% true across
233,376 labelled pairs; number-subset pairs are 99.93885% true across 258,402.
Conditioning on rejection by the original house-conflict rule gives 1,478
exact-address pairs, all true, and 4,877 number-subset pairs at 99.95899%.
These are training-country results, not measured France precision.

The exact-address France comparison adds 991 links to 932 owners, including
53 links to 32 currently empty owners. The broader number-subset alternative
has 2,165 additions. Existing target assignments are preserved. The exact
draft's additions all occur in the candidate file. Neither variant is staged
or submitted; its small plausible gain does not merit a separate submission
slot. Python compilation passes.

## Shortened-address recall audit (2026-09-27)

`scripts/france_audit_shortaddr.py` examines safe-name, unassigned France
candidates with equal first house numbers and address cosine below 0.8. Run
under the shared lock with two CPU threads. Results are saved in
`output/france_audit_shortaddr.parquet` and
`logs/france_audit_shortaddr.log`.

The pool contains 9,595 pairs. Only 182 have target address-token coverage at
least 0.9, and only 98 of those also have p1 at least 0.5. Asymmetric address
coverage therefore offers a small restoration pool, not the large missing
France recall hoped for. The audit instead finds 104 additional p1 at least
0.99 candidates affected by the literal `ndeg` prefix, including 93 exact
names and 33 with address cosine below 0.6. Some overlap the equality draft;
do not add the pool sizes together. No decisions changed. Compilation passes.

## Explicit French house-number prefix correction (2026-09-27)

The conflict pruner ignores digits glued to letters so street artifacts such as
`cite1` and `chem1` do not become house numbers. That also incorrectly ignores
`ndeg210`, the normalized spelling of an explicit French number prefix.
An address with house 210 and apartment 74 can therefore appear to conflict
with source house 210.

`scripts/france_audit_ndeg.py` measures this correction on the rejection and
short-address artifacts. It finds 286 newly nonconflicting pairs across 217
targets before score/name safeguards; 188 safe-name pairs have p1 at least
0.9 and 184 have p1 at least 0.99. All 184 high-score pairs were already
accepted before house pruning. Training contains no literal `ndeg` targets;
the analogous exact-address parser rejections at p1 at least 0.8 are 1,481
labelled pairs, all true. France restoration precision remains unmeasured.

`standalone_target_numbers` in `amzn.france_rule` now removes only the explicit
`ndeg` prefix before extracting numbers. The source-house parser and common
word swap rule retain their existing behavior. Four fixture tests cover
prefixes, apartment numbers, street artifacts, zero padding, nulls, actual
house conflicts, country scope and the common-swap guard. They pass with
`env POLARS_MAX_THREADS=2 PYTHONPATH=src .venv/bin/python -m unittest src.tests.test_france_house_numbers`.
The noon assembly still merges the saved France baseline; this correction
does not silently change that first submission.

## Leaderboard: composition, France decoy structure and the blend union (2026-09-27)

| File | Change | Leaderboard |
|---|---|---|
| 10 | auto2 blend + composition rescoring (US/India) + France safe-class additions | 0.987648 |
| 11 | 10 + France house-conflict rule + France blend union (6-model blend, p2 > 0.98) | 0.988413 (rank 50) |
| 12 | 11 + 22-model blend (US/India) + France pairs with 22-model p2 > 0.9 including same-house swaps | 0.987802 |

France decoy structure (training truth, same street, similar names, house numbers differ):
- Decoys "a few doors away" change the name: targets that add words within 30 numbers are 2.7-2.9% true. In France
  the added words are legal forms, SNC, "Groupe" and "Developpement" (v7 normalization hides the last two, so these
  decoys looked like exact names). Added-word decoys almost never sit at the Source 1 house (SNC: 23 same-house pairs
  against 9,876 nearby).
- True records whose house number was moved keep the name: identical raw names (accents, case, dotted letters folded)
  are 99.7% true more than 50 numbers away and 91% within 50. France decoys lie within 30 numbers, so France restores
  identical names beyond 30 (scripts/france_conflicts.py: drops 1,118 added-word decoys, restores 2,960).
- Blend union (scripts/france_union.py): out-of-fold blend scores above 0.98 are 99.9-100% true in every class, and the
  France scorer (stage 1) had rejected many of them. File 11 added 8,853 such pairs; the +0.00077 gain of file 11
  means nearly all added pairs were true, so France errors were mostly missed links.
- Same-house common-word swaps are not safe even when the blend is sure: file 12 added ~7k of them (plus pairs from the
  22-model blend) and lost 0.0006, i.e. the additions were ~25% true, matching the earlier ~23% estimate for France
  swaps. Several new cross-encoders score France pairs systematically higher than US/India (bge25 +1.9 logits, hard
  e5-base +1.6, reference large model +0.7; scripts/ce_sanity.py), so the 22-model blend is not used for France.

Per-entity link budget (scripts/link_budget.py): US/India submissions match the training truth category by category
(3.39 accepted vs 3.46 true per entity); the gap is address-less targets. France accepts 3.28 per entity after file 11.

Address-less targets (scripts/noaddr_*.py): 97.7% are true links in training and their per-entity count is the same
in test, but test leaves 111k unassigned (US 40k, India 48k, France 23.5k). The owner is ambiguous among same-name
entities: raw token-set agreement with the owner's Source 1 name (60% vs 29% for other candidates), shared noise
tokens with the owner's other records (16.6% vs 7.8%; the generator adds noise per record) and composition are weak.
An owner model (renormalized candidate scores, raw-name agreement, entity composition; scripts/noaddr_owner.py) is
96% precise at probability >= 0.9 on held-out ambiguous training targets but covers only ~12% of them; it adds 3,083
US/India assignments.

22-model blend: 10-fold CV F0.5 0.99167 vs 0.99129 for the 6-model blend, better in all ten folds.

## Full-competition target context (2026-09-27 midday)

The cross-validated blend and composition only see the 220k cross-validated entities, so competition for a target is
understated there. Stage-1 scores exist for every candidate entity of every target (all 2.2M training entities,
out-of-fold), so a target's full competition can be summarized the same way on training and test:
q1 = the pair's stage-1 score over the sum for all candidate entities of the target (p1 >= 0.01), m1 = their number,
the best competitor's score, raw name token-set agreement, and how many candidates share it
(`scripts/ctx_features.py`, `compose.extra_features`).

| Change (US/India, 10-fold CV over 220k entities) | CV F0.5 | Test-like view* |
|---|---|---|
| 22-model blend | 0.99167 | 0.99269 |
| + composition (previous features) | 0.99190 | 0.99283 |
| + composition with target context | 0.99260 | 0.99318 |
| context blend (23 models, context as pairwise inputs) | 0.99261 | 0.99314 |
| context blend + composition with target context | 0.99270 | 0.99327 |

*Test-like view: targets whose true owner is a training entity outside the cross-validated set are removed (in test
the owner is present; in CV every claim on them is a false positive), which removes a CV-only artefact.

Train/test shift of the context features is small within strata (address-less or not, stage-1 band); test has
slightly higher q1 because its Source 1 set is smaller (1.73M vs 2.2M entities). Tried without gain: raw-name
agreement with the entity's confident records, a second composition pass, a wider or deeper composition model, a
second context set (rank, sharper renormalization, name-core frequency), blend depth 8, blend eta 0.05, and the
address-less owner model on top (only ~60% precise once composition has the context).
France stays on file 11's decisions: a 6-model context blend becomes sure of 4.6k common-word swaps and 2.1k made-up
brands in France, the two classes that are mostly decoys there.

Remaining US/India loss in the test-like view (0.0068): address-less records whose name several entities share
(0.0053), addressed records with made-up brand names or identical names a few doors away (0.0011), false links
(0.0004).

More checks (13:00-13:40): a 24th cross-encoder (third local e5-base) leaves the context-blend CV unchanged (0.99259
vs 0.99261), so the ensemble is saturated. Composition with context on a stage-1-only scorer (the kind France uses)
gains +0.00087 in the test-like US/India view (0.98560 -> 0.98648); not applied to France, where the decoy classes
differ from the US/India ones the model would learn.

## Leaderboard: context blend (file 13) and the final plan

File 13 (file 11's France + context blend with context composition for US/India) scored **0.989261** (+0.00085 over
file 11, in line with the test-like CV gain). The context composition applied to France's stage-1 scores
(trained on US/India stage-1 CV) would add 23.6k common-word swaps and 6.0k made-up brands (France decoy classes)
and drop 3.1k acronyms and 2.9k exact names, so it is not used for France. France entities predicted empty
(15,067) whose best unclaimed candidate scores p1 > 0.7 (2,059) are mostly paired with same-house swaps (1,364)
and nearby added-word records (336): France's decoys, so no top-up there.

Final file (`scripts/final_night.sh`): US/India from a rebuild with every landed cross-encoder (kept only if its
CV is not below the 23-model context blend), France = file 13's France plus pairs the six-model blend scores
0.9-0.98 on unclaimed targets (swaps, brands and nearby added-word conflicts excluded; 2,894) and the address-less
owner-model picks (819).

## Leaderboard: France frontier test (file 14) and closing checks (2026-09-27 evening)

File 14 = file 13 + France pairs the six-model blend scores 0.9-0.98 on unclaimed targets (2,894; no swaps, brands or
nearby added-word conflicts) + 819 address-less owner-model picks: **0.989208** (-0.00005 vs file 13). The additions
were about 64% true, below the ~75% break-even, so France is saturated beyond file 11's decisions (the pairs the blend
is sure of were ~100% true; the next band is a coin flip). Other closing checks, none usable:
- France same-name same-house same-street pairs: 99.8% accepted (520,061 of 520,992); low-address-similarity
  same-house pairs are genuinely different streets.
- France candidate lists (bi-encoder top 40) are saturated by generic names (0.49% of accepted France links at ranks
  36-40 vs 0.01% US); a deeper search adds ~1.3k clean same-house pairs, the rest are ambiguous address-less records
  and nearby decoys. Training candidate misses (0.25% of links) are also mostly address-less shared names.
- Same-house "drop a type word, add a filler" France pairs (cie, groupe, developpement, associes, fils, services,
  france) mirror the US filler swaps (98.7% true); common type words (club, amicale, ecole...) are the decoys.
- Entity-side context in the blend (share of the entity's stage-1 mass, strong-candidate count, rank): CV 0.99259
  vs 0.99261 (test-like 0.99313 vs 0.99314), no gain. Relaxed France calibration (odds x2) + rules: rejected.
- France accepts where both stage 1 and the blend are sceptical: 288 pairs; no removal pool.
Best leaderboard score: file 13, 0.989261.
