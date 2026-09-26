# Business entity resolution pipeline

Links every Source 1 test record to zero or more Source 2/Source 3 records using only the supplied challenge TSVs.
No external identity data, geocoder or lookup service is used. Pretrained weights: `intfloat/multilingual-e5-small`
(MIT) for the bi-encoder and cross-encoder; XGBoost (Apache-2.0) for the matchers.

Requirements: Python 3.12 with uv, an NVIDIA GPU (16 GB is enough), about 32 GB RAM and 150 GB of disk.
Place the TSVs under `dataset/train/` and `dataset/test/` with their original names and run everything from this
directory. `dataset/`, `cache/`, `models/`, `logs/` and `output/` are ignored by Git.

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

# 5. Two-stage matcher, holdout evaluation, calibrated test decoding.
uv run python -m amzn.train_v4 stage1 p1_train p1_test stage2 p2_train p2_test evaluate --model-dir output/v4
uv run python -m amzn.train_v4 write --model-dir output/v4 --output-dir output --calibrate

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
- **Matcher**: stage-1 XGBoost (4 folds, out-of-fold scores for every training pair). Stage-2 XGBoost adds the
  score context of each Source 1 entity, per-target competition across Source 1 entities, and house-number cluster
  evidence.
- **Decoding**: each target goes to at most one Source 1 entity, then the candidate prefix with the highest expected
  F0.5 is chosen (an empty list included).
- **Label shift**: the test set holds about twice as many look-alike decoy records per entity as the training data.
  Per score bin, holdout positives per entity are kept and extra test pairs count as negatives. The corrected
  posterior replaces the raw score before decoding (`amzn.calibrate`).
