# Business entity resolution pipeline

Final Rank ~120

Team Claude's Plan's solution to the Business Entity Resolution Challenge (Amazon ML Challenge 2026): final
leaderboard score 0.989389 (macro F0.5 per Source 1 entity). `Documentation_template.md` is the methodology document
submitted to the organizers, and `docs/experiments.md` logs every experiment.

Links every Source 1 test record to zero or more Source 2/Source 3 records using only the supplied challenge TSVs.
No external identity data, geocoder or lookup service is used. Pretrained weights (all MIT or Apache-2.0, at most
568M parameters): `intfloat/multilingual-e5-small` for the bi-encoder and small cross-encoders;
`intfloat/multilingual-e5-base`, `intfloat/multilingual-e5-large`, `intfloat/multilingual-e5-large-instruct`,
`BAAI/bge-reranker-v2-m3`, `BAAI/bge-m3` and `FacebookAI/xlm-roberta-large` for the other cross-encoders; XGBoost
(Apache-2.0) for the matchers.

Requirements: Python 3.12 with uv, an NVIDIA GPU (16 GB is enough), about 32 GB RAM and 150 GB of disk.
The challenge data is not included in this repository. Place the TSVs under `dataset/train/` and `dataset/test/`
with their original names and run everything from this directory. `dataset/`, `cache/`, `models/`, `logs/` and
`output/` are ignored by Git.

## Reproduce

```bash
uv sync --python 3.12

# 1. Normalization: learn native-script -> Latin token dictionary, then normalize all records.
uv run python -m amzn.translit
uv run python -m amzn.prepare --workers 6

# 2. Key blocking (top 30 per Source 1 record) for every Source 1 record of both splits.
uv run python -m amzn.blocking --split train --import-records --index-targets --candidates --max-candidates 30 --partitions 16
uv run python -m amzn.blocking --split test  --import-records --index-targets --candidates --max-candidates 30 --partitions 16

# 3. Bi-encoder: fine-tune on training entities in hash buckets >= 100, encode, exact GPU kNN.
uv run python -m amzn.embed train
uv run python -m amzn.embed encode --split train
uv run python -m amzn.embed encode --split test
uv run python -m amzn.embed knn --split train
uv run python -m amzn.embed knn --split train --reverse --k 5
uv run python -m amzn.embed knn --split test
uv run python -m amzn.embed knn --split test --reverse --k 5

# 4. Candidates (key blocking union encoder top 40) and pair features.
uv run python -m amzn.v3 candidates --split train --base-table train_candidates --sample 1000
uv run python -m amzn.v3 features --split train --partitions 32
uv run python -m amzn.v3 candidates --split test
uv run python -m amzn.v3 features --split test --partitions 16

# 5. Stage-1 matcher (4 folds; out-of-fold scores for every training pair).
uv run python -m amzn.train_v4 stage1 p1_train p1_test --model-dir output/v7 --pair-house

# 6. Cross-encoder: train on encoder hard negatives (buckets >= 100), score pairs with stage-1 p >= 0.01.
uv run python -m amzn.crossenc train
uv run python -m amzn.crossenc select --split train --model-dir output/v7 --output cache/ce/train_pairs.parquet
uv run python -m amzn.crossenc select --split test --model-dir output/v7 --output cache/ce/test_pairs.parquet
uv run python -m amzn.crossenc score --split train --pairs cache/ce/train_pairs.parquet --output cache/ce/train_scores.parquet
uv run python -m amzn.crossenc score --split test --pairs cache/ce/test_pairs.parquet --output cache/ce/test_scores.parquet
# Second cross-encoder on other entities (seed 2, 600k entities), scored on the same pairs.
uv run python -m amzn.crossenc train --seed 2 --entities 600000 --model-dir models/crossencoder_b
uv run python -m amzn.crossenc score --split train --model-dir models/crossencoder_b --pairs cache/ce/train_scores.parquet --output cache/ce_b/train_scores.parquet
uv run python -m amzn.crossenc score --split test --model-dir models/crossencoder_b --pairs cache/ce/test_scores.parquet --output cache/ce_b/test_scores.parquet

# Third small cross-encoder (seed 3); same pairs.
./scripts/run_ce_c.sh

# Large cross-encoders on Kaggle 2x T4 notebooks: export texts and pair lists, train, score the pairs with
# stage-1 p in [0.01, 0.995), then place the scores at cache/<name>/{train,test}_scores.parquet.
uv run python -m amzn.kaggle_export
# Hard training pairs (stage-1 candidates with p1 >= 0.01, two neighbours, all true links), one file per seed.
uv run python -m amzn.kaggle_export --skip-texts --hard-from output/v7 --seed 41 --pairs-name ce_train_pairs_hard_seed41.parquet
# kaggle/ce_train_N trains one model (e5-large, e5-large-instruct, bge-reranker-v2-m3, bge-m3 or xlm-roberta-large)
# and kaggle/ce_score_N scores it; scripts/fleet_blend.sh lists the score directories it looks for.
# Before pushing, replace YOUR_KAGGLE_USERNAME in kaggle/*/kernel-metadata.json and kaggle/fleet.tsv with your
# Kaggle username and upload the exported files (kaggle/data/) as the datasets the kernel metadata lists.
# Local extra cross-encoders: multilingual-e5-base on neighbour pairs and on hard pairs.
./scripts/run_ce_base.sh
./scripts/run_ce_base_hard.sh

# 7. Generator-aware name features and full-competition target context (stage-1 score of every candidate entity
#    of each target, renormalized; raw name token-set agreement).
uv run python -m amzn.gen_features --split train
uv run python -m amzn.gen_features --split test
uv run python scripts/ctx_features.py train
uv run python scripts/ctx_features.py test

# 8. France (the country without labels). Six-model blend (small e5 x3, e5-base, e5-large x2), stage-1 scorer with
#    segment calibration and the generator-pattern rules, safe-class additions from the clean calibration, the
#    house-conflict rule and the confident cross-encoder pairs. Only the France rows of these files are used.
#    (Cross-encoder directories other than the six below must be moved aside while running fleet_blend.sh auto2.)
./scripts/fleet_blend.sh auto2        # b, base, base_hard, c, large, large2 -> output/sub_auto2
uv run python -m amzn.hybrid --blend-dir output/blend_auto2 --fallback-calibration segment-clean --output-dir output/hybrid_auto2_clean
uv run python -m amzn.france_rule --calibrated output/hybrid_auto2_clean/matching_results.tsv --raw output/hybrid_L_frnone/matching_results.tsv \
    --drop-conflicts --drop-swaps --add-initials --output output/sub_auto2_clean/matching_results.tsv
uv run python scripts/merge_france.py --usin output/sub_auto2/matching_results.tsv --france output/sub_auto2/matching_results.tsv \
    --france-extra output/sub_auto2_clean/matching_results.tsv --output output/sub_auto2_frsafe/matching_results.tsv
uv run python scripts/france_conflicts.py --base output/sub_auto2_frsafe/matching_results.tsv --output output/fr_conflicts/matching_results.tsv
uv run python scripts/france_union.py --base output/fr_conflicts/matching_results.tsv --scores output/blend_auto2/test_scored.parquet \
    --output output/fr_final/matching_results.tsv

# 9. US/India and the final file: 10-fold CV blend of every cross-encoder with the context features, full blend,
#    segment calibration and decoding, composition rescoring with the same context, France rows from step 8.
./scripts/final_assemble.sh final output/fr_final/matching_results.tsv   # output/final_final/*.tsv

# 10. France links to address-less records from the context composition model (stage-1-only US/India CV,
#     output/cv_p1only.parquet, from scripts/blend_cv.py on stage-1 scores); every other France decision is unchanged.
uv run python scripts/france_noaddr_compose.py prepare
uv run python scripts/compose.py test --cv-scores output/cv_p1only.parquet --test-calibrated cache/fr_p1_as_calibrated.parquet \
    --countries France --extra --output output/compose_fr_p1/matching_results.tsv
mkdir -p output/final_frna
uv run python scripts/france_noaddr_compose.py merge output/final_final/matching_results.tsv output/compose_fr_p1/matching_results.tsv \
    output/final_frna/matching_results.tsv

# 11. France same-house made-up brand records that the blend scores >= 0.95 on unclaimed targets (final file).
mkdir -p output/final_brand
uv run python scripts/france_brand_add.py output/final_frna/matching_results.tsv output/final_brand/matching_results.tsv 0.95
cp output/final_brand/matching_results.tsv output/final_final/candidate_pairs.tsv output/

uv run python src/utils_validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`output/matching_results.tsv` is the leaderboard upload. `output/candidate_pairs.tsv` lists exactly the pairs the
final matcher scored. `docs/experiments.md` records every holdout and leaderboard result.

## Method in brief

- **Normalization**: case folding, a learned dictionary that maps Devanagari, Bengali, Tamil, Telugu, Kannada,
  Gujarati, Gurmukhi, Odia and Malayalam tokens to their Latin spellings (97% of native test tokens covered),
  country-aware address abbreviations (US, India, France), and one canonical token per state, region or French
  department. Country is an open label; nothing is hard-coded to US or India.
- **Candidates**: rare-key blocking plus exact cosine kNN over a fine-tuned multilingual bi-encoder. Pair recall on
  training entities: keys 0.934, encoder top 40 0.997, union 0.997.
- **Features**: string similarities, IDF-weighted token overlap, legal-form agreement, house-number similarity,
  embedding similarity, reverse-neighbour competition, and sibling agreement among the candidates of a record.
- **Matcher**: stage-1 XGBoost over pairwise features (4 folds, out-of-fold scores for every training pair).
  A fine-tuned multilingual cross-encoder rescores pairs with stage-1 p >= 0.01, and a shallow XGBoost blends the two
  scores with name-genericness inputs. Entity-context features (a second stage) raise holdout F0.5 but amplify the
  test decoys, so the final scorer stays pairwise.
- **Countries without labels**: the blend and cross-encoder are fitted where names are distinctive (US, India). For
  any country absent from the labelled holdout the stage-1 score is used instead; France, whose names are generic
  ("Lille Ecole SARL" exists many times), falls in this group automatically.
- **Decoding**: each target goes to at most one Source 1 entity, then the candidate prefix with the highest expected
  F0.5 is chosen (an empty list included).
- **Generator-aware rules for countries without labels** (`amzn.france_rule`, `amzn.gen_features`,
  `scripts/france_conflicts.py`, `scripts/france_union.py`): restore acronyms, web domains, aliases and near-typos at
  the Source 1 address; drop one-word swaps to another common vocabulary word (true only about 23% of the time in
  France); drop nearby house-number conflicts whose name adds words (the decoy signature, 3% true in training) and
  restore identical-name records whose number moved more than 30 away (99.7% true beyond 50); add pairs the
  six-model cross-encoder blend scores above 0.98 on unclaimed targets (99.9-100% true in training), except the
  decoy-heavy classes. Each rule rests on a training-label statistic and is applied only where the model has no
  labels.
- **Full-competition target context** (`scripts/ctx_features.py`, `scripts/compose.py --extra`): the pair's
  stage-1 score renormalized over every candidate entity of the target, their number, the best competitor and raw
  name agreement, used by the blend and by the composition rescoring of ambiguous pairs (10-fold CV F0.5
  0.99167 -> 0.99270 for US/India).
- **Label shift**: the test set holds about twice as many look-alike decoy records per entity as the training data,
  concentrated at nearby house numbers. Per country, house relation (equal, conflict, unknown) and score bin, holdout
  positives per entity are kept and extra test pairs count as negatives; the corrected posterior replaces the raw
  score before decoding (`amzn.calibrate`). `amzn.estimate` samples labels from these posteriors to rank candidate
  submissions without test labels.

## License

The code is released under the [MIT License](LICENSE). The challenge data belongs to the organizers and is not
covered by this license.
