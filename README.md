# Business entity resolution pipeline

This project links every Source 1 test record to zero or more Source 2 and Source 3 records. It writes the two TSV files required by the ML Challenge. No external business data or lookup service is used.

## Environment and data

- Python 3.12, managed with `uv` (`.python-version`, `pyproject.toml`, and `uv.lock`).
- Training uses an NVIDIA GPU through XGBoost. Preprocessing and blocking use CPU cores and DuckDB.
- Put the supplied files in `dataset/train/` and `dataset/test/`, using the original `train_source1.tsv` through `train_source3.tsv`, `train_ground_truth.tsv`, and `test_source1.tsv` through `test_source3.tsv` names. The dataset is ignored by Git.
- Run commands from the project root. The local `cache/` and `output/` directories are ignored by Git.

```bash
uv sync --python 3.12
uv run python -m amzn.prepare --workers 6
uv run python -m amzn.blocking --split train --import-records --index-targets
uv run python -m amzn.blocking --split train --candidates --sample-per-thousand 60 --max-candidates 30
uv run python -m amzn.evaluate_blocking
uv run python -m amzn.train
uv run python -m amzn.blocking --split test --import-records --index-targets
uv run python -m amzn.blocking --split test --candidates --max-candidates 10
uv run python -m amzn.infer
uv run python src/utils_validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`output/matching_results.tsv` is the leaderboard upload. `output/candidate_pairs.tsv` contains the exact candidate set scored by the model. Both files include one row for every test Source 1 entity, including records with no candidates or matches. `output/pair_model.json` and `output/training_metrics.json` are reproducibility artifacts.

## Method

`amzn.prepare` streams all source files, transliterates non-ASCII text, standardizes common address terms, and derives canonical name, address, house-number, and token-anchor fields. Country remains an open string label. No US/India-only rule excludes France.

`amzn.blocking` creates exact-name, exact-address, name/address token, house/name token, and house/address token keys. It removes keys with large target posting lists, joins remaining keys in DuckDB, then ranks candidate pairs by inexpensive name and address similarities. The final cap is applied before model features and predictions. The candidate TSV therefore reflects exactly what the classifier receives.

`amzn.features` computes 18 pairwise features in DuckDB. `amzn.train` fits an XGBoost binary classifier on a deterministic 5% Source 1 training sample and tunes threshold and candidate cap on a disjoint 1% sample. Validation uses the challenge's macro F0.5 definition, including singleton records. The current run selected 10 candidates per Source 1 and threshold 0.575. Validation macro F0.5 is 0.884862 on 22,144 held-out Source 1 records. Candidate recall at 10 is 0.869125 on the combined 6% sample.

`amzn.infer` scores every final test candidate and writes both TSV files. The supplied validator checks file structure, membership, and the match-subset-of-candidates rule. Source code and lockfile are sufficient to rebuild the outputs from the supplied TSV data.

## Validation

```bash
uv run python -m unittest discover -s src/tests -v
```

The blocking test checks transliteration, misspelled names, cross-country exclusion, and empty output rows.
