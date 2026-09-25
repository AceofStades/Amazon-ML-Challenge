# Business entity resolution pipeline

This pipeline links every Source 1 test record to zero or more Source 2 and Source 3 records, using only the supplied challenge TSVs. No external identity data or lookup service is used.

Use Python 3.12 with uv. Training and prediction use an NVIDIA GPU through XGBoost; normalization and blocking use CPU cores and DuckDB. Place the supplied TSV files under dataset/train/ and dataset/test/ with their original names. Run all commands from this directory. The dataset/, cache/, and output/ directories are ignored by Git.

~~~bash
uv sync --python 3.12
uv run python -m amzn.prepare --workers 6

# First-pass training blocker and GPU matcher.
uv run python -m amzn.blocking --split train --import-records --index-targets --candidates --sample-per-thousand 60 --max-candidates 30
uv run python -m amzn.evaluate_blocking
uv run python -m amzn.train --output-dir output/stage1 --rounds 850

# Expand around high-confidence first-pass matches, then train the final matcher.
uv run python -m amzn.seeds --model-dir output/stage1 --feature-table train_sample_features --output-table train_sample_seeds --threshold 0.9
uv run python -m amzn.blocking --split train --expand --seeds-table train_sample_seeds --base-table train_sample_candidates --output-table train_sample_expanded_candidates --max-candidates 30
uv run python -m amzn.blocking --split train --combine --base-table train_sample_candidates --expanded-table train_sample_expanded_candidates --output-table train_sample_combined_candidates
uv run python -m amzn.train --candidate-table train_sample_combined_candidates --feature-table train_sample_combined_features --output-dir output/stage2 --rounds 900

# Apply both stages to every test entity.
uv run python -m amzn.blocking --split test --import-records --index-targets --candidates --max-candidates 30
uv run python -m amzn.features --split test --candidates test_candidates --output test_features
uv run python -m amzn.seeds --model-dir output/stage1 --feature-table test_features --output-table test_seeds --threshold 0.9
uv run python -m amzn.blocking --split test --expand --seeds-table test_seeds --base-table test_candidates --output-table test_expanded_candidates --max-candidates 30
uv run python -m amzn.blocking --split test --combine --base-table test_candidates --expanded-table test_expanded_candidates --output-table test_combined_candidates
uv run python -m amzn.infer --model-dir output/stage2 --candidate-table test_combined_candidates --feature-table test_combined_features
uv run python src/utils_validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
uv run python -m amzn.package --team-name "Claude's Plan"
~~~

The leaderboard upload is output/matching_results.tsv. The output/candidate_pairs.tsv file contains exactly the pairs scored by the final matcher. Both files have one row per test Source 1 entity, including entities with empty lists. The ZIP contains both outputs, source code, both model artifacts, pinned dependencies, and the methodology document.

## Method

The prepare step streams and normalizes the supplied records: case folding, Unicode transliteration, punctuation cleanup, common address abbreviations, sorted canonical fields, house number, and name/address token anchors. Country remains an open string label; France is included automatically.

The first blocking pass uses exact name/address, rare token, token-pair, and name-prefix/address keys. Keys with large target posting lists are dropped. It ranks remaining pairs by name/address Jaro-Winkler similarity and blocking evidence, keeping at most 30 per Source 1 entity.

The first GPU XGBoost classifier scores those pairs. Predictions above 0.9 become precise seeds for a second candidate pass. For each seed, the blocker finds other target records sharing a rare key (posting list at most 40), ranks them by similarity to both the seed and the Source 1 record, and keeps at most 30 additional pairs. The final classifier scores the union. In the 6% deterministic training sample, this raises pair-level candidate recall from 0.893211 to 0.924587 while averaging 53.25 candidates per Source 1 record.

The feature step computes 27 numeric features, including canonical and raw string similarities, character bigram and token Jaccard, edit similarity, house and address-number agreement, missingness, source, and blocking evidence. The final XGBoost 3.4.1 model (Apache-2.0 license) uses GPU histogram training. About 5% of training Source 1 entities fit each model; a disjoint 1% tunes the threshold and cap against the challenge's macro F0.5 metric, including singletons. The selected cap is 60 and threshold is 0.65. Held-out macro F0.5 is 0.924402 on 22,144 entities. The test set includes an unseen country, so leaderboard results may differ from this holdout.

Run uv run python -m unittest discover -s src/tests -v for the integration test. The supplied validator checks both output files and every referenced test ID.
