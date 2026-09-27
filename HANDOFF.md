# Handoff: state of the work and how to continue

This is the working memory of the assistant that ran this project until 2026-09-27 03:30 IST. Any agent or teammate
picking up the work should read it first. It covers the goal, the user's standing instructions, what runs where, what
was learned, and what to do next. For a finding, `docs/experiments.md` has the full experiment log.
`Documentation_template.md` is the methodology deliverable for the organizers. `README.md` describes how to reproduce
the pipeline.

Times are IST (UTC+5:30). Paths are relative to the repo root `~/code/amzn` on hades unless they start with `~`.

---

## 1. First ten minutes (checklist)

1. **File 10 is unsubmitted and on hold.** The user prefers the first submission after the Kaggle fleet finishes.
   It is now `output/to_submit/10_HOLD_auto2_blend_composition_france_additions_matching_results.tsv`.
   Preserve all five daily slots. A file 10 score would combine blend, composition, and France changes and cannot
   isolate whether composition transfers.
2. **Check the cross-encoder (CE) fleet:** `./scripts/kaggle_status.sh`. Each line shows the training and scoring
   kernel status and whether the scores are already in `cache/ce_NAME`.
3. **Check that the background jobs are alive** (section 6):
   `pgrep -af "kaggle_chain|fleet_auto|gpu_chain|crossenc" | grep -v pgrep`.
   If the chains died (for example after a reboot), run `./scripts/kaggle_fleet.sh`. It is idempotent: finished
   chains are skipped, and scoring kernels that are already pushed are not pushed again.
4. **Check the disk:** `df -h /`. There was 68 GB free at 03:20. Each automatic re-blend writes about 2.8 GB, so
   prune older `output/*_autoN` dirs (section 9).
5. **Read section 7** (plan for the rest of the day) and **section 8** (open leads).

---

## 2. Goal and competition facts

- **Challenge:** Amazon/Unstop "Business Entity Resolution Challenge". Link every Source 1 business record to its
  Source 2/Source 3 records. Output `matching_results.tsv` (`source1_entity_id`, comma-separated
  `matched_entity_ids`, possibly empty) plus `candidate_pairs.tsv`. The final deliverable is a zip with code, a
  methodology doc and the files.
- **Metric:** macro F0.5 per Source 1 entity. Singletons count: an empty prediction for an entity with no links
  scores 1. Precision is weighted over recall.
- **Rules:** models must be MIT or Apache-2.0 licensed with at most 8B parameters. No external data or lookups.
- **Test data:** US, India and **France**. Training has labels for US and India only.
- **Deadline:** the competition ends at **midnight between 2026-09-27 and 2026-09-28 IST** (24:00 on Sep 27).
- **Submissions:** the user submits through the web UI. The daily quota resets at 00:00. On Sep 27 the team had
  **5 submissions**. File 10 uses the first one if the user submitted it before sleeping. Use them carefully.
- **User's target (their "goal"):** rank 1, roughly 0.992 or better. The leader was at 0.990788–0.990816 on the
  night of Sep 26. Our best is **0.987379** (file 7). The rank was 44 and later dropped to 69 as others improved.
  The latest user report is **rank 78, score 0.987379**, with an explicit target of **at least 0.992 and rank 1**.
- **Leaderboard history:**

| File | What | Score |
|---|---|---|
| v2 | first stage-2 | 0.896978 |
| v3 / v3_cal | stage 1 + label-shift calibration | 0.960932 / 0.968379 |
| hybrid_v7full | CE blend for US/India, stage 1 for France | 0.984806 |
| hybrid_L France raw / calibrated | + large CE; France uncalibrated / segment-calibrated | 0.983476 / 0.985292 |
| 6 | + France rules (patterns, conflict drops, swap drops, initials) | 0.987207 |
| 7 | + gen-feature blend with e5-base for US/India | **0.987379** (best) |
| 9 | diagnostic: file 7 with every France entity empty | 0.851 |
| 10 | auto2 blend + composition rescoring (US/India) + France additions | staged 03:11; score pending |

- **What file 9 tells us:** US/India F0.5 is about 0.990–0.991, matching the training holdout. France F0.5 is only
  about 0.966–0.971. France is 15% of entities, so **the gap to rank 1 is almost entirely France**.
  Bringing France to about 0.99 would add about 0.003 overall.

---

## 3. Standing instructions from the user

- **Commits:** never add AI attribution. No "Co-Authored-By" line, no "generated with" line. Commit often, one
  small meaningful commit per step, and **push after every commit** (`git push origin hmm`) so teammates always see
  the latest code. Work directly in `~/code/amzn` on hades.
  The user also requests **more than 100 meaningful commits on September 27** (at least 101, counted in IST).
  There were 17 commits dated September 27 at 04:05 IST. Do not pad the history with empty or cosmetic commits.
- **Submission files:** stage them in `output/to_submit/` as
  `<N>_<short_description>_matching_results.tsv`, so the user knows exactly what to upload. Keep `INDEX.txt` there
  up to date with the reasoning, the expected score and the leaderboard history. Move submitted files to
  `output/submissions/` with a hardlink (`ln -f`), then remove them from `to_submit`.
- **Storage:** watch the hades disk. It filled up once. Prune superseded, regenerable outputs, never submitted or
  current-best files, and say what was removed. Hardlink identical `candidate_pairs.tsv` files (1.4 GB each)
  instead of copying.
- **Kaggle:** use the teammates' accounts freely, but **respect GPU quotas** (section 5). Keep every Kaggle
  dataset **private**. Credentials may live on hades because it is the user's own PC. Do not overwrite the
  user's own `~/.kaggle` credentials.
- **Keep models running.** Do not kill Kaggle kernels or local training. The user wants compute kept busy until
  the end ("the competition still ends at 12am").
- **Chat style:** the user likes very terse replies (they use a "caveman" terse mode). Docs, commits and notes stay
  in normal prose.
- **Sleep:** the user sleeps about 04:00–12:00 and keeps the session alive. Submissions happen when they are awake.

---

## 4. Machines, environment, credentials

- **hades** (the user's PC): RTX 5060 Ti 16 GB, 31 GB RAM, 16 cores, root disk 275 GB. Repo at `~/code/amzn`,
  branch `hmm`, remote `git@github.com:AceofStades/Amazon-ML-Challenge.git`.
  The current agent session runs on hades itself; no SSH hop is needed.
  - Python via uv: `.venv/bin/python`. Modules run as `.venv/bin/python -m amzn.<module>`. Scripts that import
    `amzn` need `PYTHONPATH=src`.
  - Ignored by git: `dataset/`, `cache/`, `models/`, `logs/`, `output/`.
- **Kaggle CLI on hades:** `~/.local/bin/kaggle` (version 2.2.3, installed with `uv tool`). It is not on the
  non-interactive `PATH`, so use the full path.
- **Kaggle accounts.** Select an account with `KAGGLE_CONFIG_DIR`. Tokens are stored on hades at these paths.
  **Never commit tokens.**

| Account | Config dir on hades | Notes |
|---|---|---|
| aceofstades (user) | default `~/.kaggle` (credentials.json) | Owns dataset `aceofstades/amzn-er-data`, shared with all accounts. `datasets files` returns 403 (OAuth). |
| rekishu | `~/.kaggle/rid` | |
| hrishitachavan | `~/.config/kaggle/tiya` | |
| siddharthbokade | `~/.config/kaggle/sid` | |
| umedindulkar | `~/.config/kaggle/umed` | |
| cycl0p5 (zero, not o) | `~/.config/kaggle/vj` | The older token `cyclop5` had no GPU. |
| ayj2411 | `~/.config/kaggle/ansh` | |
| adityadavjekar | `~/.config/kaggle/dave` | |
| techno0 | `~/.config/kaggle/yash` | **No phone verification, so no GPU or internet.** CPU only, unused. |

  Each account has a private dataset `<user>/amzn-er-hard` with its hard-pair training files.

---

## 5. Kaggle GPU quota (important)

- Each account gets **30 GPU-hours per week**. A "GPU T4 x2" session counts **double**.
- At most **2 concurrent GPU sessions per account**. A push fails with "Maximum batch GPU session count of 2" when
  both are busy.
- **Every push creates a new kernel version that starts running.** Never push a kernel that is already running;
  it wastes quota.
- **Budget as of 03:20 on Sep 27.** Each teammate account has one 7.5 h training run (first run), one 3.75 h
  training run (second run) and two scorings of about 1.7 GPU-hours each. That puts them at about 28–29 of
  30 GPU-hours. **No quota is left for new runs on teammate accounts.** The user's own account had about 10 GPU-hours
  left before the XLM-R scoring (kernel score-7, running). Use its remainder only for emergencies, and ask the user
  first.
- A kernel that hits the quota is killed before it saves anything.
- After the fleet finishes, the only free GPU is hades itself (section 6).

---

## 6. What is running right now (03:25 IST)

### 6a. Kaggle CE fleet: `kaggle/fleet.tsv`, handled by `scripts/kaggle_chain.sh`

Each chain waits for its training kernel. It then pushes the scoring kernel (`kaggle/ce_score_N`, which reads the
trained model from the training kernel's output), waits for it, and downloads `scores_train.parquet` and
`scores_test.parquet` into `cache/ce_NAME/{train,test}_scores.parquet`. The chains run detached on hades
(`setsid nohup`) and log to `logs/kaggle/NAME.log`.

- **Status:** `./scripts/kaggle_status.sh`
- **Start or resume all:** `./scripts/kaggle_fleet.sh`
- **Stop the watchers only** (kernels keep running): `pkill -f "^/bin/sh ./scripts/kaggle_chain.sh"`

| Cache name | Kernel | Account | Model | Training pairs | State at 03:21 |
|---|---|---|---|---|---|
| xlmr7 | 7 | aceofstades | XLM-R large | hard seed 61 | scoring running |
| bgem3 | 9 | siddharthbokade | bge-m3 | hard seed 91 | scoring running |
| labse14 | 14 | ayj2411 | LaBSE | hard seed 151 | scoring running |
| large3 | 3 | rekishu | e5-large | kNN seed 31 | training (7.5 h) |
| bge | 4 | hrishitachavan | bge-reranker-v2-m3 | kNN seed 0 | training (7.5 h) |
| large10 | 10 | cycl0p5 | e5-large | hard seed 101 (version 2; v1 errored) | training (7.5 h) |
| xlmr12 | 12 | umedindulkar | XLM-R large | hard seed 121 | training (7.5 h) |
| large16 | 16 | adityadavjekar | e5-large | hard seed 171 | training (7.5 h) |
| bnd21 | 21 | rekishu | e5-large | boundary pairs | training (3.75 h) |
| bndbge22 | 22 | hrishitachavan | bge-reranker-v2-m3 | boundary pairs | training (3.75 h) |
| e5i23 | 23 | siddharthbokade | e5-large-instruct | hard seed 81 | training (3.75 h) |
| e5i24 | 24 | umedindulkar | e5-large-instruct | hard seed 131 | training (3.75 h) |
| bge25 | 25 | cycl0p5 | bge-reranker-v2-m3 | hard seed 111 | training (3.75 h) |
| bgem3_26 | 26 | ayj2411 | bge-m3 | hard seed 161 | training (3.75 h) |
| bge27 | 27 | adityadavjekar | bge-reranker-v2-m3 | hard seed 181 | training (3.75 h) |

- **Expected landing:** scores roughly between 05:00 and 11:30. The second runs were pushed around 02:00–02:40.
- **Training pairs:**
  - "hard" = stage-1 candidates with p1 ≥ 0.01 for 600k entities outside the blend buckets, plus two kNN neighbours
    and all true links (2.83M pairs, 73% positive).
  - "boundary" = only-scored pairs with 0.02 ≤ p1 < 0.98 (1.84M pairs, 29% positive).
  - Files are in `kaggle/data_hard/`.
- **Kernel sources** are in `kaggle/ce_train_N` and `kaggle/ce_score_N`. `kaggle/make_kernel.py N USER MODEL
  SEED_OR_STEM [BUDGET]` generates new ones.
- **Deleted or failed kernels:** 5, 6, 8, 11, 13, 15, 17, 18. They were deleted for quota, or ran without a GPU
  because phone verification was missing. Ignore them.

### 6b. hades local processes

- `scripts/fleet_auto.sh`: a loop. Whenever the set of `cache/ce_*/test_scores.parquet` changes, it waits 2 minutes
  and runs `scripts/fleet_blend.sh autoN`. That produces `output/blend_autoN` (holdout-evaluated),
  `output/hybrid_autoN` and `output/sub_autoN` (with the France rules, validated). The log is
  `logs/fleet_auto.log`, and each run logs to `logs/fleet_autoN.log`. auto1 and auto2 exist; the next run is auto3.
  - Note: `sub_autoN` uses the *old* France path (rules only). The submission path is
    `scripts/final_assemble.sh` (section 7).
- `scripts/gpu_chain.sh`: trains the e5-base CE on hard pairs seed 141 (`models/crossencoder_base_hard2`). It was at
  step 17,400 of 22,124 at 03:20. It then scores train and test into `cache/ce_base_hard2`. Log: `logs/gpu_chain.log`.
- `scripts/gpu_chain2.sh`: waits for gpu_chain, trains the e5-base boundary specialist
  (`models/crossencoder_base_bnd`), then scores into `cache/ce_base_bnd`. Expected to finish around 08:00–09:00.
  **After that the hades GPU is free.** One more local e5-base model (about 4 h train plus score) fits before evening.
- **Never overlap two GPU jobs on hades.** 16 GB overflows and the process segfaults. Chain them.

**Recovery update:** `gpu_chain.sh` terminated during hard2 test scoring with
`free(): invalid pointer` at 4.86M/8.83M pairs. Hard2's final model and train
scores are intact, but no test output exists. The boundary chain continues.
`scripts/gpu_recover_hard2.sh 124580` waits for that boundary shell and every
cross-encoder worker to finish, then recovers missing local outputs with
checkpointed 250k-pair subprocesses and batch 128. Check
`logs/gpu_recover_hard2.log`; never restart hard2 scoring alongside the boundary
chain. The helper also recovers boundary scores if its original scorer fails.

---

## 7. Plan for the rest of Sep 27

The resumed session runs `scripts/noon_build.sh fleetA` detached on this machine.
Starting at 11:00 IST, it waits for readable train and test score files from every
fleet model and both remaining local models, up to a 12:00 IST readiness cutoff.
It then waits for the shared pipeline lock, runs the final assembly,
and stages validated file 11 with hardlinks, an index entry, and an experiment
log commit. Check `logs/noon_fleetA.log` and the host process list before starting
another final assembly. Both pipeline scripts now use `logs/blend.lock` to avoid
overlap; an auto-blend started before this change finishes normally.
If models are still missing at the cutoff, it drafts with landed models and logs
the pending models. Drafting does not consume a submission; the user uploads only
after we review completeness and validation. Do not let a failed kernel block all progress.

1. **Around 11:00–12:00, build the noon submission** with every CE that has landed:
   `./scripts/final_assemble.sh fleetA` (about 25–30 min). It runs these steps:
   1. `scripts/blend_cv.py`: 10-fold CV over 220k training entities. Compare with pre1 0.99129 and auto2 0.99130.
   2. `amzn.blend`: full blend fit and test scores.
   3. `amzn.hybrid`: segment calibration, one winner per target, expected-F0.5 decode.
   4. `scripts/compose.py test`: composition rescoring for US/India.
   5. `scripts/merge_france.py`: France rows from `output/sub_auto2_frsafe`.
   6. The validator.

   The output is `output/final_fleetA/`.
   - Before running, make sure `fleet_auto.sh` is not in the middle of a blend (`pgrep -af fleet_blend`).
     31 GB RAM does not fit two blends plus a hybrid.
   - If one CE looks broken (fold logloss worse, or scores constant or NaN), move its dir out of the way
     (`mv cache/ce_X cache/off_ce_X`) and rerun. Every `cache/ce_*` that has both score files is included
     automatically.
   - Stage the result as `output/to_submit/11_<desc>_matching_results.tsv` (hardlink), update `INDEX.txt`, and
     tell the user.
2. **Hold file 10 and use the completed fleet as the first baseline.** File 10's estimated 0.9876–0.9878
   is unmeasured. Its simultaneous blend, composition and France changes cannot isolate composition transfer.
   - If composition needs a leaderboard check, compare otherwise identical fleet variants with and without it.
     Build the variant without composition from the same hybrid and France decisions, for example:
     `python scripts/merge_france.py --usin output/hybrid_NAME/matching_results.tsv --france output/sub_auto2_frsafe/matching_results.tsv --output output/final_NAME_nocomp/matching_results.tsv`,
     then copy or hardlink `candidate_pairs.tsv` and validate.
3. **Spend the remaining submissions on France** (section 8). The leaderboard is deterministic to 1e-6, so a
   France-only change measures the precision of the changed pairs exactly. For every pair added or removed, the
   score delta shows whether the pairs were mostly true.
4. **Final submission by about 23:00.** Submit the best-scoring file.
5. **Final package before midnight.** Update the README, `Documentation_template.md` and `docs/experiments.md`,
   then build the zip:
   `.venv/bin/python -m amzn.package --team-name "Claude's Plan" --submission-dir output/final_X --stage1-dir output/v7 --blend-dir output/blend_X`.
   An earlier test zip was 664 MB with 119 files. Commit and push.

---

## 8. Open leads (where the remaining score is)

### 8a. France: the big lever (≈ 0.02 France F0.5 = 0.003 overall)

**Per-country target budget** (`scripts/country_budget.py <matching_results.tsv>`, run on file 10, 03:15). All
columns except E are per Source 1 entity:

| Split | Country | E | T/E | T_noaddr/E | A/E | A_noaddr/E | U/E | U_noaddr/E |
|---|---|---|---|---|---|---|---|---|
| train (truth) | US | 1,323,633 | 4.674 | 0.168 | 3.459 | 0.164 | 1.215 | 0.004 |
| train (truth) | India | 883,188 | 4.680 | 0.139 | 3.465 | 0.136 | 1.216 | 0.003 |
| test (file 10) | US | 663,106 | 5.756 | 0.167 | 3.392 | 0.106 | 2.364 | 0.061 |
| test (file 10) | India | 809,986 | 5.824 | 0.138 | 3.394 | 0.079 | 2.431 | 0.059 |
| test (file 10) | France | 259,452 | 5.531 | 0.166 | 3.243 | 0.076 | 2.288 | 0.090 |

Column key:
- E: number of Source 1 entities.
- T: target records in S2/S3 of that country. T_noaddr: those without an address.
- A: accepted links. A_noaddr: accepted links to targets without an address.
- U: unassigned targets. U_noaddr: unassigned targets without an address.

Reading:
- In training, **97.7% of address-less targets are true links**, since the generator creates almost no
  address-less decoys.
- The test has the same number of address-less targets per entity as training. So the extra test decoys (T/E 5.8
  against 4.7) all carry addresses, and **an address-less test target is almost surely a true link of some entity
  of its country**.
- We assign only 46% of them in France (64% US, 57% India). The unassigned ones are names shared by several
  Source 1 entities, and the choice among them is ambiguous.
- If an assignment rule reached ≥ 0.75 precision, the France recall gain would be up to 0.09 links per entity.
  The break-even precision for a 3-link entity is about 0.73.
- Ideas for choosing the owner:
  - composition priors: an entity rarely has two address-less records (13% have one, 1% have two);
  - entities with fewer accepted links are likelier owners;
  - the source split between S2 and S3;
  - name-noise agreement.
- `scripts/compose.py` does this for US/India, but it is fitted on blend scores. On France it also removed true
  suffix and filler pairs. A France variant restricted to **address-less targets only** is untested.
  Validate it on the training holdout with stage-1 p1 scores first.

**Other France facts** (details in `docs/experiments.md`):
- France accepts 3.24 links per entity against 3.39 for US/India, while every labelled country has 3.46 true links
  per entity.
- France names are generic ("Lille Ecole SARL" repeats many times).
- The CEs do not understand French legal suffixes, so the France scorer is stage-1 p1 with segment calibration.
  The CE blend on France drops 24.5k true "drop word plus French suffix" pairs.
- France-only rules in `src/amzn/france_rule.py` gave +0.0019 overall:
  - restore raw-decision patterns (acronym, domain, alias, typo with JW ≥ 0.92, containment at the same address);
  - drop non-typo house conflicts;
  - drop one-word swaps to a common word;
  - add initials and domains on the Source 1 street.
- Measured decoy classes in France:
  - Same-house common-word swaps are about 23% true.
  - Pure made-up brand names at the same address: 0.0895 per entity against 0.034–0.038 in US/India, only about
    33% true, and inseparable.
- A covariate-shift reweighting of stage 1 was shelved (XGBoost segfaults, and the holdout got worse).

### 8b. US/India (≈ 0.0002–0.0005 left)

- Adding more CEs to the blend gives +0.0001 each at best. Composition rescoring gave +0.0002 CV.
- The holdout error audit on 22k entities: 1,836 false negatives, 78% of which are address-less targets with a name
  shared by several S1 entities (ambiguous). There were only 109 false positives. The remaining US/India error is
  mostly the same address-less ambiguity as France.

### 8c. Rejected ideas (do not retry without a new angle)

- Blend tuning: depth 8, eta 0.05, 5-model bagging.
- No-CE blend as the France scorer.
- Blend-scored France.
- No-address calibration segment.
- France precision push.
- Dropping France brands.
- Top-up of entities with few matches (45–73% precision on the holdout).
- Composition on France.
- Name-agreement calibration cells (too sparse).
- The v4 stage 2 for France (worse than stage 1).

---

## 9. Pipeline map (what produces what)

1. **Candidates:** DuckDB key blocking (`amzn.blocking`) plus bi-encoder kNN (`amzn.embed`), assembled by
   `amzn.v3 candidates`. Candidate recall is 0.9974.
2. **Features:** `amzn.v3 features` (name/address similarities, house numbers, duplicate counts and more) and
   `amzn.gen_features`. The latter adds generator-aware name-change descriptors: exact, add, drop, typo, swap class,
   acronym, domain, alias, brand. Written to `cache/gen/{train,test}.parquet`.
3. **Stage 1:** `amzn.train_v4 stage1 ... --model-dir output/v7`. XGBoost with out-of-fold p1 for every training
   pair.
4. **CEs:**
   - `amzn.crossenc` trains and scores locally.
   - Kaggle kernels cover the large models.
   - All score the same pair list, `cache/ce/{train,test}_scores.parquet` (stage-1 p ≥ 0.01).
   - Output goes to `cache/ce_NAME/`.
5. **Blend:** `amzn.blend --extras name,full,gen --extra-ce name=dir ... --depth 6`. A pairwise XGBoost on logit p1,
   the CE logits, 67 stage-1 features, name and gen features. Output in `output/blend_NAME`.
   `scripts/blend_cv.py` is the 10-fold CV version, which is far less noisy than the 22k holdout. It also writes
   `output/cv_TAG.parquet`, which compose uses.
6. **Hybrid:** `amzn.hybrid --blend-dir ...`.
   - US/India use blend scores with per-segment calibration (house relation equal/conflict/unknown; posterior
     a·n1(b)/m(b) to adjust for the test's extra decoys).
   - France uses stage-1 p1 with its own segment calibration.
   - Then one winner per target and an expected-F0.5 decode.
7. **France rules:** `amzn.france_rule --calibrated ... --raw ... --drop-conflicts --drop-swaps --add-initials`.
   `--raw` is the uncalibrated France decisions (`output/hybrid_L_frnone`).
8. **Composition:** `scripts/compose.py` (cv/test). It rescores ambiguous US/India pairs (0.01 ≤ p < 0.99) with the
   entity's composition.
9. **Merge:** `scripts/merge_france.py --usin A --france B [--france-extra C]` takes US/India rows from A and
   France rows from B. With C, it adds France pairs from C that fall in the safe classes (drop, filler swap, exact,
   domain, typo, acronym, add).
10. **Validate:** `.venv/bin/python src/utils_validate_submission.py -m matching_results.tsv -c candidate_pairs.tsv`.
    The last line must say PASS.

**Current best building blocks:**
- stage 1 `output/v7`;
- France decisions `output/sub_auto2_frsafe/matching_results.tsv` (file 7 rules + the acronym fix + 2.4k safe-class
  additions from the clean calibration);
- US/India `output/final_pre1` (file 10).

**Pruning guide (disk):**
- Each `fleet_auto` run creates `blend_autoN` (1.2 GB), `hybrid_autoN` (1.5 GB) and `sub_autoN` (94 MB).
  Keep the newest two and anything referenced by `to_submit` or `submissions`.
- `output/submissions` holds 34 hardlinks of one 1.4 GB `candidate_pairs.tsv`. Deleting one of them frees nothing.
- Never prune `output/v7` (stage 1), `cache/v3` (pair features), `cache/ce*`, `cache/gen`, `cache/embeddings` or
  `output/sub_auto2_frsafe`. Older `output/v2`–`v6` and `hybrid_*` experiment dirs are superseded. Check that
  nothing in `to_submit` or `submissions` links to them before deleting.

---

## 10. Key data facts (generator structure)

- Every labelled country: **3.46 true links per Source 1 entity**; 5.6% singletons.
- Records of an entity are spread over both S2 and S3.
- 13% of entities have one address-less record and 1% have two.
- At most one DBA/alias per entity. Acronyms 0.74% against 0.01% for two; brands 8.5% against 0.7%; domains 16.7%
  against 1.7%.
- The test has about **2× the decoys** of training: perturbed names a few doors away on the same street.
- Same-address, same-house one-word swaps in training:
  - to a filler word (center, services, service, partners): 98.7% true;
  - to a rare or garbled word: 95.9% true;
  - to a common vocabulary word: 45.3% true.
- House-number changes on the same street and name: a last-digit difference of 1–2 is 42.5% true, 3–10 is 13.5%
  true, and inner-digit or dropped-digit typos are 82–99% true.
- No leakage: row order and ids are random.

---

## 11. Pitfalls learned the hard way

- **Editing a running shell script in place** corrupts the running process: sh reads scripts incrementally.
  Write a new file and `mv` it over the old one.
- **`pgrep -f` / `pkill -f` match their own ssh or shell command line.** Anchor the pattern, for example
  `"^/bin/sh ./scripts/kaggle_chain.sh"`.
- **zsh** does not word-split `$var`. Heredocs containing `$` inside f-strings break quoting. Write files with an
  editor or scp instead.
- **polars:** joins do not keep row order unless you pass `maintain_order="left"`.
- **Memory:** `amzn.hybrid` once reached 19 GB RSS and was OOM-killed. `calibrate.house_segments` was patched to
  compute numbers only for the needed pairs. Do not run two heavy jobs at once.
- **GPU:** overlapping training and scoring on the 16 GB card segfaults. Use sequential chains.
- **Kaggle:**
  - GPU and internet need phone verification.
  - e5-large-instruct checkpoints are fp16. Load with `torch_dtype=torch.float32`, otherwise you get "Attempting to
    unscale FP16 gradients".
  - `kaggle kernels status` output looks like `... has status "KernelWorkerStatus.RUNNING"`. Match that string, not
    a bare "error", because transient API errors contain that word too.
- **Scoring subtleties:**
  - The holdout F0.5 "in candidates" (as printed by `eval_holdout.py`) ignores links missing from the candidates
    (candidate recall 0.9974), so it overstates by about 0.001. The CV figure from `blend_cv.py` is the one to
    compare blends with.
  - The one-winner-per-target tie bug was fixed early. Keep ties deterministic.
- **After a hades reboot:** `/tmp` is wiped, so keep helper scripts in `scripts/`. Restart `fleet_auto.sh`,
  `kaggle_fleet.sh` and any GPU chain that had not finished.

---

## 12. Useful one-liners

```sh
./scripts/kaggle_status.sh                                  # fleet status
tail -n 3 logs/kaggle/*.log                                 # chain progress
tail -f logs/fleet_auto.log                                 # auto re-blends
tail -n 2 logs/gpu_chain.log logs/gpu_chain2.log            # local GPU chain
.venv/bin/python scripts/cmp_submissions.py A.tsv B.tsv France   # what changed between two files, by name-change class
.venv/bin/python scripts/country_budget.py FILE.tsv         # per-country budget table (section 8a)
PYTHONPATH=src .venv/bin/python scripts/eval_holdout.py output/blend_A output/blend_B   # holdout comparison
df -h / ; du -sh output/* | sort -h | tail                  # disk
```

---

## Status update (2026-09-27 10:40 IST)

- Leaderboard: file 10 0.987648, **file 11 0.988413 (best)**, file 12 0.987802. Rank 1: 0.991483.
- Two submissions left, planned for **17:00** and **23:30**.
- Staged: `output/to_submit/13_file11_france_plus_all22_blend_usin_plus_owner_model_matching_results.tsv`
  (= `output/final_safe_own`). It combines file 11's France, the 22-model blend + composition for US/India, and 3,083
  US/India address-less owner-model assignments. Expected 0.9887-0.9889.
- Do **not** use the 22-model blend for France. Several new cross-encoders inflate France scores. Do **not** add
  same-house common-word swaps; file 12 showed they are ~25% true.
- The last local cross-encoder (`scripts/gpu_chain4.sh`, e5-base on hard pairs seed 161) lands around 15:00. Then:
  1. Run `./scripts/final_assemble.sh fleetB output/final_pre1_frcu/matching_results.tsv`.
  2. If the CV beats fleetA's 0.99167, rebuild file 13 from it:
     `merge_france.py` with France from `final_pre1_frcu`, then
     `add_owner.py --countries US,India --min-pm 0.9` (rerun `scripts/noaddr_owner.py` on the new base first).
- Optional: France owner-model picks at pm >= 0.9 (819). They are mostly unique-name records (q ~ 1). The spot check
  looks right, but they are not leaderboard-tested.
- Findings are logged in `docs/experiments.md` (section "Leaderboard: composition, France decoy structure ...").
- The session transcripts are in `.session/` (git-ignored; they contain the hades password, never commit them) and in
  `~/.claude/projects/-home-aceofstades-code-amzn/`, so `claude --resume 0eca6fab-c753-4b4a-872b-124a1ee7098e` works
  when started in `~/code/amzn`.

## Status update (2026-09-27 12:45 IST)

- Submissions left: 2, planned for **17:00** and **23:30**.
- Staged for 17:00: `output/to_submit/13_file11_france_plus_context_blend_usin_with_extended_composition_matching_results.tsv`
  (= `output/final_ctxA`): file 11's France + context blend (23 cross-encoders + full-competition target context)
  + composition with the same context for US/India. CV test-like 0.99327 vs 0.99269 for the plain 22-model blend.
  Expected about 0.9891-0.9893.
- `scripts/final_assemble.sh NAME output/final_pre1_frcu/matching_results.tsv` now builds exactly this pipeline
  (EXTRAS defaults to name,full,gen,ctx; composition uses --extra). Always pass file 11's France file.
- Before rebuilding on new cross-encoders, recompute nothing: `cache/ctx/{train,test}.parquet` depend only on stage 1.
- Kaggle: kernels 31-34 (pravinbokade, dipikabokade123) train until about 16:00-18:30, then chains score and download
  them into `cache/ce_*`. Kernels 35-36 (octopus25) failed: the account has no phone verification (0 GPUs, no
  internet); re-push them only after the user verifies it.
- The 23:30 file: rebuild with every landed model (`./scripts/final_assemble.sh fleetZ output/final_pre1_frcu/matching_results.tsv`),
  keep it only if its CV beats ctxA (0.99261 blend CV), stage it, then build the deliverable zip with `amzn.package`.
- A stale `scripts/noon_build.sh` (from an earlier session) fired at noon and would have staged a file with the old
  France; it was stopped and `logs/noon_fleetA.done` marks it finished.

## Status update (2026-09-27 17:50 IST)

- Leaderboard: file 13 **0.989261** (best, rank 55; context blend + context composition for US/India, file 11's
  France), file 14 0.989208 (France band 0.9-0.98 + owner picks: ~64% true, rejected). Two submissions left.
- `output/to_submit/15_FINAL_same_as_file13_best_matching_results.tsv` = file 13; re-submit it last if the leaderboard
  ranks the most recent submission. `output/claudes_plan_submission.zip` is the deliverable built from it.
- Everything else tested flat or negative (see docs/experiments.md, closing checks). Kaggle kernels 31-42 keep
  landing into cache/ce_*, but the ensemble is saturated (23 -> 25 models: CV flat).
