# SPEC

Last updated: 2026-10-04, at commit `f1b917e`. Owned by the manager. How the repo works is in the `CLAUDE.md` files; who does what is in `.claude/roles/`. This file says what exists, what is wrong, and what is next.

## Goal

Two pieces for a vintage clothing resale shop, meant to run together on a new item: `tag_analyzer/` reads the brand and era off a photo of the tag, and `listings/` writes the first line of the Depop listing (`💎 Vintage 90's Boxy Russell Hoodie.`) from two photos plus those facts.

## What each package does today

### `tag_analyzer/`

Identifies brand and era from a tag photo. Nothing is trained and no model runs locally.

| File | Does |
|---|---|
| `__init__.py` | Paths (`data/tags/`, `data/test_tags/`, `data/labels.json`, `data/test_labels.json`) and `IMAGE_SUFFIXES`. |
| `manifest.py` | Walks `data/tags/<Brand>/<era>_N.ext` and writes `data/labels.json`. Brand is the folder name CamelCase-split and title-cased; era comes from the `NN_` filename prefix. |
| `build_index.py` | Checks manifest against disk, then uploads the library to Weaviate. Drops and recreates the `TagPhoto` collection every run. |
| `store.py` | All Weaviate access: `connect_from_env`, `recreate_collection`, `upload_library_photos`, `search_nearest_photos`, `library_size`. The `multi2vec-clip` sidecar embeds on ingest and on query. |
| `voting.py` | `vote(neighbors)`: similarity-weighted vote for brand and era; confidence is the winning brand's share of the weight. |
| `match.py` | `match(client, photo)` → `TagMatch`. `in_library=False` when the top similarity is below `THRESHOLD = 0.70`. `K = 3`. |
| `eval.py` | Held-out accuracy over `data/test_tags/`: pure `score` plus printing. Skips the threshold on purpose. Logs each run through `tracking.py`. |
| `tracking.py` | Every MLflow call for the package and the read of `MLFLOW_TRACKING_URI`. Logs nothing when it is unset. |
| `schema.py` | `Neighbor`, `TagMatch`, and `EvalScore` pydantic models. |
| `eval.ipynb` | Interactive `build_index` + `eval` with image strips for the misses. |

State: paused since `6571d34` (Sep 29 2026). Eval: **19/19 brand, 16/19 era**.

### `listings/`

Exports training data and LoRA fine-tunes Qwen2.5-VL-7B to write the 💎 line.

| File | Does | Runs on |
|---|---|---|
| `__init__.py` | Paths under `data/listings/` and `adapters/listing_first_line/`. | both |
| `config.py` | Loads `.env`; the only reader of `POSTGRES_*` / `AWS_*` / `S3_*` and `MLFLOW_TRACKING_URI`. | both |
| `db.py` | pg8000 access. One query: listings whose description starts with 💎, gender `Men`/`Women`, with thumbnail (position 0) and secondary (position 1) photo URLs. | both |
| `images.py` | Downloads a photo from the private S3 bucket to `data/listings/images/<id>_<position>.<ext>`, skipping files already present. | both |
| `parse.py` | `first_line` / `normalized_line`, `era_from_line`, `facts_for`, and the prompt pair `prompt_text` / `facts_from_prompt_text`. | both |
| `export.py` | Builds conversational JSONL examples, splits by `(brand, category)` group (seed 42, ~10% val), writes `train.jsonl` and `val.jsonl`. | both |
| `brands.py` | `listing_brand_for(tag_brand)` maps tag analyzer names onto Postgres names; `main` reports ok / mapped / NEEDS MAPPING per brand. | both |
| `model.py` | Base model, LoRA rank 16 on language layers only, `load_finetuned_model`. | GPU box |
| `examples.py` | `read_jsonl`, `open_photo`, and the readers for one exported example (`photo_paths`, `facts`, `target_line`). | both |
| `invariants.py` | The five eval invariants, `invariant_pass_rates`, `print_summary`. | both |
| `baseline.py` | Scores the shop's own target lines with the invariants and logs them as the `shop-baseline` run. | both |
| `similarity.py` | The headline metric: word-level match between each generated line and the shop's line (`exact_match`, `word_overlap`). Free. | both |
| `judge.py` | Claude (`claude-haiku-4-5`) compares each generated line with the shop's line: five checks, an overall verdict, a reason. Writes `judge.jsonl`. | both |
| `tracking.py` | Every MLflow call for the package: training run, adapter upload, run ID file in the adapter folder, eval scores and table. Logs nothing when tracking is off. | both |
| `train.py` | Lazy photo dataset, `SFTTrainer` config, 3 epochs, saves the adapter. | GPU box |
| `generate.py` | `generate_first_line(model, tokenizer, thumbnail, secondary, facts)`. | GPU box |
| `eval.py` | Generates for every val row, scores it with `invariants.py`, writes `data/listings/eval.jsonl`. | GPU box |

State: first training run Sep 30 2026 (543 steps, ~97 min, final loss 0.13, mean 0.28). Last export: 1,445 train / 160 val rows, 3,220 images. Voice, format, brand and era placement are good. Shop baseline for the invariants: 100% diamond, 100% period, 99.3% word count, 89.3% brand, 100% era.

### How they connect

- Neither package imports the other. The only link in code is `listings/brands.py` reading `data/labels.json`, which `tag_analyzer.manifest` writes.
- At inference the tag analyzer is meant to supply `brand` and `era` for the facts JSON (`brand`, `era`, `category`, `gender`, `color`). That wiring does not exist yet (task L5).
- Brand vocabulary: one mapping today (`Fruit Of The Loom` → `Fruit of the Loom`). Seven tag brands have no listings and pass through unchanged.

### Tests and infrastructure

- `test/` holds a unit test for every function that can be imported on the laptop, written to be read as documentation: `test/listings/test_<module>.py` and `test/tag_analyzer/test_<module>.py`. Run with `python -m pytest test/` (offline, under a second; configured by `pytest.ini`, pytest from `requirements-dev.txt`). As of 2026-10-05: 596 passed. `test/test_every_function_has_a_test.py` fails the suite when a function has no test; 15 functions that only run on the GPU box are exempt (`model.py`, `generate.py`, and what is left in `train.py` and `eval.py`).
- `docs/listings.md` and `docs/tag_analyzer.md` list every function with an example input and output. The manager updates them in the same commit as a change to a function.
- **MLflow** (plan approved 2026-10-04): every listings training run, every listings eval, and every tag analyzer eval is recorded on a tracking server on the EC2 box. `mlflow-skinny==3.16.1` is the client, installed in `.venv`. All tracking code is written and tested offline (T5, L7, L9, L8). The server has been **running on the EC2 box since 2026-10-04** (health and the experiments API answer on the public address), but no run has been logged yet: `MLFLOW_TRACKING_URI` is still blank in the laptop `.env`.
- `docker-compose.yaml` runs Postgres 16, pgAdmin, Weaviate 1.27.0, and the `multi2vec-clip` sidecar (`clip-ViT-B-32-multilingual-v1`, CPU) on the user's EC2 instance.
- Training and eval for `listings/` run only on the GPU box (g5.xlarge); details in `listings/CLAUDE.md`.

## In progress

- **Four-agent workflow setup** (manager, two builders, validator). Role files and this spec were written 2026-10-04. Agents message each other with `SendMessage`. V1 was the first task through the cycle (PASS, not committed yet).
- **Listings retrain cycle**: fix the split → re-export → retrain → eval. No code for it has been written yet. It starts with L1.
- `tag_analyzer/` has no work in flight.

## Known issues

### `listings/`

| # | Issue | Task |
|---|---|---|
| 1 | `(Levi's, Jeans)` is one 101-item group that landed entirely in val: 63% of the val set, and zero Levi's jeans in train. | L1 |
| 3 | The first model over-mentions brands: 97.5% `brand_mentioned` (156/160) against the shop's own 89.3%. Its adapter, `eval.jsonl`, and training log are on the laptop since 2026-10-05. | L3, M7 |
| 4 | The model invents Levi's fit numbers (545, 541), drops washes, and sometimes gets specifics wrong. | L4 |
| 5 | Photos reach the model at 512 px on the long side, not the 768² set in `model.py`, because `UnslothVisionDataCollator` resizes first. Small text such as a Levi's back patch is unreadable. | L6 |
| 7 | `export.main` prints "none shared" about groups, which stops being true after L1. | L1 |

### `tag_analyzer/`

| # | Issue | Task |
|---|---|---|
| 8 | `manifest.py` reads `sys.argv` at import time, so importing it under pytest picks up pytest's arguments as paths. | T1 |
| 9 | `match.THRESHOLD = 0.70` was set from the positive side only (lowest correct top similarity is 0.72). There are no out-of-library photos to calibrate against. | T3 |
| 10 | 3 of 19 era predictions miss; they are library coverage gaps (Champion 70s and 00s, Screen Stars 80s). | T4 |
| 11 | Style drift from the house rules: single-letter names (`m`, `p`, `f`, `e`, `n`) in `manifest.py`, `build_index.py`, `match.py`, `eval.py`. | T2 |
| 12 | `eval.py` and `match.py` each define their own `K = 3`. | T2 |
| 18 | `manifest.to_brand` splits an all-caps folder name letter by letter (`NFL` → `N F L`), so an acronym brand cannot match its Postgres name. | none yet |
| 17 | `voting.vote([])` raises `ValueError`. `match` guards against an empty neighbor list; `tag_analyzer.eval.main` does not, so eval crashes on an empty library. | none yet |

### Shared

| # | Issue | Task |
|---|---|---|
| 13 | The `Sportswear` tag folder matches no Postgres brand. | M2 |
| 14 | Every `CLAUDE.md` is gitignored, though the root one describes itself as checked in. Agents on this working tree see them; a fresh clone would not. | M1 |
| 19 | The MLflow server has no login and port 5000 is open to the internet: anyone with the address can read, change, or delete runs and write files under `mlflow/` in the S3 bucket. Fix options: restrict the security group to known IPs, or turn on MLflow's built-in username/password. | none yet |
| 16 | `requirements.txt` lists `torch` and `torchvision`, but nothing on the laptop imports them since embedding moved into Weaviate; only `listings/train.py` uses torch, on the GPU box. | M4 |

## Next tasks

Tasks are listed in the order they should be picked up within each package. "Blocked on" names what must happen first.

### `listings/` (listings builder)

**L1. Split large groups internally**

Change the train/val split in `listings/export.py` so one big group cannot swallow validation.

Acceptance criteria:
- Groups with more than 20 examples are split about 90/10 between train and val; the threshold is a named constant.
- Groups with 20 or fewer examples stay whole on one side.
- No `listing_id` appears in both train and val, and every input example appears in exactly one of them.
- Overall val share is between 8% and 12% for an input shaped like the real data (1,605 examples, one group of 101, many small groups).
- No single `(brand, category)` group makes up more than 25% of val for that input.
- The same input always produces the same split (seed 42).
- The split is a pure function that runs without Postgres, S3, or files on disk.
- The summary line printed by `export.main` no longer claims no groups are shared.

**L2. Make the pure eval and data helpers importable on the laptop** — DONE 2026-10-04, PASS round 1, not committed yet. New modules `listings/invariants.py` and `listings/examples.py`. Decision: `invariant_pass_rates([])` keeps raising `ZeroDivisionError`; an eval with no rows must not log scores.

Move the invariants and their constants out of `eval.py`, and `read_jsonl` / `open_photo` out of `train.py`, into modules that do not import `unsloth`, `trl`, `listings.model`, or `listings.generate`. Module names are the builder's choice.

Acceptance criteria:
- On the laptop `.venv`, importing the new module or modules succeeds.
- `invariants(line, facts)` returns the same five keys as today: `starts_with_diamond`, `ends_with_period`, `word_count_ok`, `brand_mentioned`, `era_mentioned`.
- Behavior is unchanged: brand check passes for `None`, `""`, `Other`, and every `BLANK_TAG_BRANDS` entry; a curly apostrophe in the brand matches a straight one in the line and the reverse; era check passes when the era fact is `null`; more than 12 words fails `word_count_ok`.
- `eval.py` and `train.py` import these functions instead of defining them, and `eval.py` still writes the same row shape to `eval.jsonl`.
- A new pure `invariant_pass_rates(results)` returns the five pass rates as fractions keyed by invariant name (two rows, one failing `brand_mentioned` → `brand_mentioned` is `0.5`, the rest `1.0`). `print_summary` uses it and prints the same lines as today.
- The `from __future__ import annotations` lines and the unused `PROJECT_ROOT` import in `eval.py` are gone.

**L7. Track training and eval in MLflow** — DONE 2026-10-04, PASS round 1, not committed yet. The call sites in `train.py` / `eval.py`, loss arriving in the run, and the adapter upload are unverified until the first tracked run on the GPU box (L3).

New `listings/tracking.py` holding every MLflow call for the package. It imports `mlflow` and never `unsloth`, `trl`, `listings.model`, or `listings.generate`. `train.py` and `eval.py` call it and stay thin.

Acceptance criteria:
- `MLFLOW_TRACKING_URI` is read in `listings/config.py` and nowhere else in the package.
- When it is unset, no tracking function creates a run or raises, and `train` and `eval` behave exactly as before.
- Experiment name is `listings-first-line`. One run per trained model: training opens it, eval adds to the same run.
- Training logs these settings: `BASE_MODEL`, `LORA_RANK`, `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `MAX_SEQ_LENGTH`, `MAX_IMAGE_PIXELS`, `SEED`, the row count and a short content hash of `train.jsonl` and of `val.jsonl`, and the git commit.
- The dataset hash is a pure function of the file's bytes: same file, same hash; one changed byte, different hash.
- Loss and learning rate reach the run every 10 steps (`report_to` is `"mlflow"` when tracking is on, `"none"` when off).
- After the adapter is saved, the whole `adapters/listing_first_line/` folder is uploaded to the run, and the run ID is written to a file inside that folder.
- Eval reads the run ID from the adapter folder and logs to that run: the five invariant pass rates from `invariant_pass_rates`, the val row count, `eval.jsonl` as a file, and the rows as an MLflow table (`listing_id`, `target`, `generated`, the five flags).
- Eval on an adapter folder with no run ID file logs to a new run rather than failing.

**L9. Make `log_eval_run` safe at the edges** — DONE 2026-10-04, PASS round 1, not committed yet. To check once the server is up: a missing run is reported with an error code the fallback recognises.

Small follow-up to L7, from the review.

Acceptance criteria:
- With no result rows, `log_eval_run` still raises, but before any run is opened or reopened: the training run's status is untouched and no empty run is left behind.
- When the adapter folder's run ID names a run the server does not have, `log_eval_run` logs to a new run instead of raising, and records the unknown ID on that run as the param `adapter_run_id_not_found`.
- A run ID that does exist behaves as before.

**L8. Log the shop baseline as a run** — DONE 2026-10-04, PASS round 1, not committed yet. Reproduced 100 / 100 / 99.3 / 89.3 / 100 on the current export (1,605 lines). Accepted as is: the run does not record which export it scored, so after the L1 re-export it gains a second point per metric.

Acceptance criteria:
- A command scores the target lines in `train.jsonl` and `val.jsonl` with `invariants` and logs the five pass rates to a run named `shop-baseline` in `listings-first-line`.
- The scoring is a pure function of the examples; on the real export it reproduces 100 / 100 / 99.3 / 89.3 / 100 (builder reports the output).
- Running it twice does not create a second `shop-baseline` run.

**L10. Judge generated lines against the shop's lines with Claude** — DONE 2026-10-04, PASS round 1, not committed yet. Tested with a fake client only; no real Claude call has been made (M7).

The user's decision (2026-10-04): the invariants cannot tell whether a generated line says the same thing as the shop's line, so an LLM judge compares the two. Text only (no photos), a checklist plus an overall verdict, judge model `claude-haiku-4-5`.

New `listings/judge.py`, run as `python -m listings.judge`. Laptop-importable: no `unsloth`, `trl`, `listings.model`, or `listings.generate`.

Inputs: `data/listings/eval.jsonl` (written by `python -m listings.eval`: `listing_id`, `target`, `generated`) joined by `listing_id` to the facts in `data/listings/val.jsonl` (`examples.facts`).

Each verdict has exactly these fields:

| Field | Type | Question the judge answers |
|---|---|---|
| `same_item` | bool | Is the generated line about the same kind of garment as the shop's line? |
| `brand_agrees` | bool | Is the brand handled the same way: the same brand named, or both leave it out? |
| `era_agrees` | bool | The same decade, or both leave it out? |
| `nothing_invented` | bool | True when the generated line states no specific the shop's line lacks (fit number, team, graphic, size, material). |
| `key_details_kept` | bool | True when nothing the shop included is missing (wash, fit, color, graphic). |
| `overall` | `equivalent` / `acceptable` / `wrong` | `equivalent`: a buyer learns the same thing. `acceptable`: minor differences, nothing false. `wrong`: a different item, a wrong brand or era, or an invented specific. |
| `reason` | str | One sentence. |

The rubric deliberately ignores exact wording, word count, the 💎 and the final period (the invariants cover those), and whether the generated line is better than the shop's.

Acceptance criteria:
- The judge call uses the official SDK: `client.messages.parse(model=JUDGE_MODEL, max_tokens=…, system=<rubric>, messages=[…], output_format=<the pydantic verdict model>)` and reads `response.parsed_output`. `JUDGE_MODEL = "claude-haiku-4-5"`. No `thinking` and no `effort` parameter (Haiku 4.5 rejects `effort`). No raw HTTP.
- The client is created once (`anthropic.Anthropic()`, which reads `ANTHROPIC_API_KEY` from the environment after `listings.config` has loaded `.env`) and passed into the judging functions, so tests use a fake client. No code reads, prints, or logs the key. With the key missing or blank, the command stops with a clear message before judging any row.
- The prompt is built by a pure function of the facts, the shop's line, and the generated line. The rubric is one named constant; the two lines are presented as data to compare, clearly labeled, never as instructions.
- One failing row does not stop the run. A row whose response has `stop_reason` `refusal` or `max_tokens`, or whose call raises an `anthropic` API error after the SDK's own retries, is recorded with the `listing_id` and the error and left out of the rates.
- An `eval.jsonl` row whose `listing_id` is not in `val.jsonl` is reported, not judged with made-up facts.
- Output: `data/listings/judge.jsonl` (path constant in `listings/__init__.py`), one row per judged listing: `listing_id`, `target`, `generated`, then the seven verdict fields.
- A pure summary function returns the pass rate of each of the five checks, the share of each `overall` value, and the row count. The command prints it, with the number of rows that errored and the total input and output tokens used (`response.usage`).
- `python -m listings.judge 10` judges only the first 10 rows, prints each shop line, generated line, and verdict together for reading, and does **not** log to MLflow. With no argument it judges every row and logs.
- MLflow (through `listings/tracking.py`, same on/off rule): the judged run goes to the adapter's run, resolved exactly as `log_eval_run` does. Metrics `judge_same_item`, `judge_brand_agrees`, `judge_era_agrees`, `judge_nothing_invented`, `judge_key_details_kept`, `judge_overall_equivalent`, `judge_overall_acceptable`, `judge_overall_wrong`, `judge_rows`; params `judge_model` and `judge_rubric_hash` (a short hash of the rubric text, so scores from different rubrics are not compared by mistake); `judge.jsonl` as a file; the rows as the table `judge_rows.json`.
- With no rows to judge it raises before any run is opened, like `log_eval_run`.

Not part of this task: the rubric is a draft until the user has read a 10-row pilot and agreed with the verdicts. The manager runs that pilot once `eval.jsonl` is on the laptop and `ANTHROPIC_API_KEY` is set.

**L11. Judge follow-ups from the L10 review** — DONE 2026-10-04, PASS round 1, not committed yet. Accepted limitation: after re-judging, a run's `judge_` metric history mixes rubrics; compare only runs whose `judge_rubric_hash` tags match.

Acceptance criteria:
- `judge_model` and `judge_rubric_hash` are logged as MLflow **tags**, not params, so the same adapter run can be judged again after the rubric changes. Re-judging never raises because of an earlier judgment; the tags then show the latest model and rubric.
- The first `anthropic.AuthenticationError` or `anthropic.PermissionDeniedError` stops the whole command with a clear message; it is not recorded once per row.
- In full mode the command prints a progress line every 20 rows (`20/160 judged`), so a long run is visibly alive. The pilot's output is unchanged.
- Decided and unchanged: the pilot writes no `judge.jsonl`.

**L12. Similarity between the generated line and the shop's line** — DONE 2026-10-05, PASS round 1, not committed yet. Sep 30 model: 17/160 exact, word overlap mean 0.632, median 0.615 (manager reran the command). The judge's 1–5 score has not been produced by Claude yet.

The user's decision (2026-10-05): the checklist is fine, but the metric that matters most is how closely the generated line matches the shop's line. Two measures, one free and one from the judge.

Part A, word-level similarity, no Claude call. New laptop-importable `listings/similarity.py`, run as `python -m listings.similarity`.

Acceptance criteria:
- Words are compared after lower-casing, removing apostrophes (both `’` and `'`, so `Levi's` becomes `levis`), and dropping punctuation and the 💎 (`"Levi’s"` and `"Levi's"` are the same word; `"90's"` and `"90’s"` are the same word).
- `exact_match(target, generated)` is true when the two word lists are identical.
- `word_overlap(target, generated)` is the F1 of shared words, counting repeats: `1.0` for the same words in any order, `0.0` for no shared word. Reference values the function must reproduce: shop `💎 Y2K Skull Polo Shirt.` vs generated `💎 Y2K Skull Graphic Polo Shirt.` → `0.89`; shop `💎 Vintage Early 00's Mid Wash Levi's 560 Comfort Fit Jeans.` vs generated `💎 Y2K Levi’s 550 Baggy Jean.` → `0.13` (both rounded to two places).
- A pure summary over `eval.jsonl` rows returns the exact-match share, the mean and median word overlap, and the row count. On the Sep 30 `eval.jsonl` now on the laptop it must print 17/160 exact (10.6%), mean 0.632, median 0.615 (builder reports the output).
- The command prints that summary and the five lowest-scoring pairs, and logs `similarity_exact_match`, `similarity_word_overlap_mean`, `similarity_word_overlap_median`, and `similarity_rows` to the adapter's MLflow run through `listings/tracking.py`, resolved as `log_eval_run` does. Tracking off: prints only.
- No rows: raises before any run is opened.

Part B, the judge's own similarity score.

Acceptance criteria:
- `Verdict` gains `similarity`, an integer from 1 to 5, placed before `overall`. The rubric defines each value:
  - 5: a buyer learns exactly the same thing; only wording or word order differs.
  - 4: the same item, with one minor detail added, dropped, or changed.
  - 3: the same item, with several details different or missing.
  - 2: the same kind of garment, but a key fact differs (brand, era, fit or model number).
  - 1: a different item.
- A value outside 1 to 5 cannot be stored (the schema or the model rejects it).
- `judge.jsonl` rows and the pilot printout include `similarity`.
- `judge_summary` adds `similarity_mean` and `similarity_4_or_5` (the share of rows scoring 4 or 5); they are logged as `judge_similarity_mean` and `judge_similarity_4_or_5`.
- The five checks, `overall`, and `reason` are unchanged.

**L13. Keep one MLflow run per adapter, also for adapters trained before tracking** — DONE 2026-10-05, PASS round 1, not committed yet. Note: the run ID file is per machine; copy `mlflow_run_id.txt` along with an adapter when moving it between the laptop and the GPU box.

Today every command run against an adapter folder with no run ID file (the Sep 30 adapter on the laptop) opens a separate new run, so its similarity, judge, and eval results would be scattered over several runs.

Acceptance criteria:
- When `start_adapter_run` opens a new run because the adapter folder has no run ID, or names a run the server does not have, it writes the new run's ID into the adapter folder's run ID file, so the next command reuses that run.
- A second command against the same folder logs to the same run: after `log_similarity_run` then `log_judge_run` on a folder that started with no run ID file, there is exactly one run and it holds both sets of metrics.
- When the adapter folder itself does not exist, nothing is written and the command still logs to a new run.
- A folder whose run ID is known to the server behaves as before, and its file is not rewritten.
- `adapter_run_id_not_found` is still recorded when a file named an unknown run.

**L3. Re-export, retrain, evaluate, and record the result**

Blocked on: L1 and L2 committed; the user starting the GPU box. Run by the user with the builder's instructions; nothing here can be checked by the suite.

Acceptance criteria:
- `python -m listings.export` on the GPU box reports train and val counts, and both files contain `(Levi's, Jeans)` rows.
- `python -m listings.train` completes and saves `adapters/listing_first_line/`.
- `python -m listings.eval` completes, and the five invariant percentages are recorded in `ready.md` next to the shop baseline (100 / 100 / 99.3 / 89.3 / 100).
- The adapter and `eval.jsonl` are copied back to the laptop before the instance is stopped (stopped, not terminated).
- `ready.md` lists ten generated-versus-target pairs for Levi's jeans so the user can judge fit numbers and washes.

**L4. Decide whether Levi's fit number and wash become facts**

Blocked on: L3 results. Decision for the user, prepared by the manager. If yes, it becomes a contract change: new fields in the facts JSON, a source for them at inference, and a task for each builder.

**L5. End-to-end entry point**

Blocked on: L3 (an adapter on the machine that runs it). One function and CLI in `listings/` that takes a tag photo and the two listing photos and returns the line.

Acceptance criteria:
- The import goes from `listings` into `tag_analyzer`, never the reverse.
- The brand passes through `listing_brand_for` before it reaches the prompt.
- When `TagMatch.in_library` is false, brand and era are `null` in the facts rather than a guess.
- The prompt is built by `parse.prompt_text`.
- The facts-building step is a pure function testable on the laptop without the model.

**L6. Find out whether photos can reach the model larger than 512 px**

Backlog. Investigation first: can `UnslothVisionDataCollator` be given a size, and what does 768 cost in memory and step time on the A10G. Deliver findings in `ready.md`; no code change is required to close it.

### `tag_analyzer/` (tag_analyzer builder)

**T1. Stop reading `sys.argv` at import time in `manifest.py`**

Acceptance criteria:
- Importing `tag_analyzer.manifest` does not read `sys.argv` and has no side effects.
- `python -m tag_analyzer.manifest` still writes `data/labels.json` from `data/tags/`, and the optional two arguments (tags directory, output file) still work.
- The entry-building logic can be called with a directory and returns the entries, so it is testable against a temporary folder.
- `to_brand("FruitOfTheLoom")` is `"Fruit Of The Loom"`; `era_from_name("90_1.PNG")` is `"90s"`; a filename with no era token yields `None`.
- Entry shape is unchanged: `filename`, `brand`, `era`.
- A regenerated `data/labels.json` is byte-identical to the current one.

**T2. Style pass**

Acceptance criteria:
- No single-letter variable names remain in `tag_analyzer/*.py`.
- `K` is defined once and used by both `match.py` and `eval.py`.
- No behavior change: `python -m tag_analyzer.eval` still reports 19/19 brand and 16/19 era (builder runs it and reports the output).
- `vote` returns the same winner and confidence for the same neighbors.

**T3. Calibrate the "not in library" threshold**

Blocked on: the user adding out-of-library tag photos (brands with no folder in `data/tags/`) and saying how they should be labeled.

Acceptance criteria:
- The eval reports, at `THRESHOLD`, how many in-library photos are accepted and how many out-of-library photos are rejected.
- Brand and era accuracy are still reported over in-library photos only and are unchanged for the existing 19.
- `THRESHOLD` is set from that data, and the numbers behind it are in `ready.md`.
- The accept/reject decision is a pure function of the neighbors, testable offline.

**T4. Close the era coverage gaps**

Blocked on: the user adding library photos for Champion 70s and 00s and Screen Stars 80s to `data/tags/`.

Acceptance criteria:
- After `manifest` and `build_index`, `python -m tag_analyzer.eval` reports more than 16/19 era with brand still 19/19.
- `python -m listings.brands` shows no new `NEEDS MAPPING` rows (manager checks with the listings builder).

**T5. Track eval runs in MLflow** — DONE 2026-10-04, PASS round 1, not committed yet. Checked against a local file store; not yet against the real server (M6).

Can start now. Independent of T1–T4.

Acceptance criteria:
- Scoring in `tag_analyzer/eval.py` is a pure function: test labels and each photo's neighbors in, counts and the list of mistakes out. `main` prints and logs what it returns.
- `python -m tag_analyzer.eval` prints the same lines as today (`library: …`, `brand accuracy: 19/19 = 100.0%`, `era accuracy:   16/19 = 84.2%`, the `mistakes:` block).
- New `tag_analyzer/tracking.py` holds every MLflow call and the read of `MLFLOW_TRACKING_URI`.
- When it is unset, nothing is logged, nothing raises, and the printed output is unchanged.
- Experiment name is `tag-analyzer`. One run per eval, with settings `K`, `THRESHOLD`, the CLIP model name, library size, test set size, and git commit; metrics `brand_accuracy` and `era_accuracy` as fractions; the mistakes as a text file.
- The CLIP model name (`clip-ViT-B-32-multilingual-v1`) is a named constant in `store.py`.
- `tag_analyzer/` does not import `listings`.

### `test/` (validator)

**V1. Unit tests for every function** — DONE 2026-10-04, PASS (192 passed), not committed yet. Also covers `tag_analyzer/build_index.py` and `tag_analyzer/eval.py`, which were outside the module list below. Review: `handoff/listings/review.md`.

The tests double as documentation: reading `test/listings/test_parse.py` should tell the user what every function in `listings/parse.py` does.

Blocked on: the user running `pip install -r requirements-dev.txt`.

Acceptance criteria:
- `python -m pytest test/` passes on the laptop with no network.
- Layout is `test/listings/test_<module>.py` and `test/tag_analyzer/test_<module>.py`, one file per source module, with no `__init__.py` under `test/`.
- Every function in the laptop-importable modules has at least one test named `test_<function_name>_<what_it_does>`, one behavior per test, in the same order as the functions in the module.
- Modules in scope: `listings/parse.py`, `export.py`, `brands.py`, `db.py`, `images.py`; `tag_analyzer/manifest.py`, `voting.py`, `store.py`, `match.py`.
- Behaviors that must appear, as the code behaves today:
  - `listings.parse`: `normalized_line` (HTML unescape, one space after 💎, period added only when missing), `first_line` (takes only the first line), `era_from_line` (`90's` → `90s`, `90’s` → `90s`, `Y2K` → `00s`, `2000's` → `00s`, none → `None`), `facts_for` (`N/A` and empty color → `None`), and `facts_from_prompt_text(prompt_text(facts)) == facts`.
  - `listings.export`: `image_block` path relative to the repo root, `training_example` shape (two image blocks, one text block, one assistant text), `split_group_key`, `grouped_split` keeps every group whole, `write_jsonl` round-trips non-ASCII.
  - `listings.brands`: `listing_brand_for` (mapped and pass-through), `case_insensitive_matches`, `tag_analyzer_brands` from a temporary labels file, `listing_brands` on a fake query result.
  - `listings.db`: `has_both_photos`, `fetch_listings_with_photos` drops rows missing either photo, `execute_query_command` returns dict rows and closes the connection, `execute_write_commands` commits once and returns the row count — all on a fake connection.
  - `listings.images`: `key_from_url`, `download_photo` names the file `<id>_<position>.<ext>` and skips an existing file — on a fake S3 client and `tmp_path`.
  - `tag_analyzer.voting.vote`: weighted winner, confidence share, brand and era decided independently.
  - `tag_analyzer.manifest`: `to_brand`, `era_from_name`.
  - `tag_analyzer.store`: `cosine_similarity_from_distance`, `stable_uuid_for` is stable for the same filename, `encode_image_base64`, `search_nearest_photos` turns fake Weaviate objects into `Neighbor`s, `library_size` is 0 when the collection is missing.
  - `tag_analyzer.match.match`: `in_library=False` for no neighbors and for a top similarity below `THRESHOLD`; brand, era, and confidence from the vote otherwise.
- `test/test_every_function_has_a_test.py` walks the source modules with `ast` and fails when a function has no test named after it, apart from a named exemption set.
- Exemptions are listed in `review.md` with a reason each. Expected today: every function in `listings/model.py`, `train.py`, `eval.py`, `generate.py` (cannot be imported on the laptop; L2 shrinks this list), and `main` functions that cannot run entirely on fakes and `tmp_path`.
- Nothing reads or writes `data/`, `adapters/`, or `.env` values, and nothing opens a real connection.

**V2. Tests for L1** — written when `handoff/listings/ready.md` announces L1; one test per L1 criterion.

**V3. Tests for L2** — one test per L2 behavior criterion, against the new importable module.

**V4. Tests for T1** — manifest entries from a temporary folder tree, including a file with no era token being skipped.

**V5. Tests for L7 and L8** and **V6. Tests for T5** — a unit test for every new function. MLflow is pointed at a file store under `tmp_path`, so the suite stays offline; mlflow 3.16.1 needs `MLFLOW_ALLOW_FILE_STORE=true` for that, set per test with `monkeypatch.setenv`. No test may leave an `mlruns/` folder in the repo. Required cases: nothing is logged when the URI is unset; settings and metrics land in the run; the run ID file round-trips; the dataset hash changes with the file.

**V7. Tests for L10** — a unit test for every new function, with a fake client object in place of `anthropic.Anthropic()` (no network, no key). Required cases: the prompt contains the facts and both lines; a parsed verdict becomes the expected row; a `refusal` row and a raised API error are recorded and excluded from the rates; a `listing_id` missing from `val.jsonl` is reported; the summary rates on a small hand-made set; the 10-row mode does not log; the rubric hash changes with the rubric.

**V8. Tests for L12** — a unit test for every new function, including the two reference word-overlap values, apostrophe and punctuation handling, the summary on a small hand-made set, and a verdict with `similarity` 0 or 6 being rejected.

### Manager

- **M1.** Ask the user whether the `CLAUDE.md` files should be tracked; if yes, remove the `CLAUDE.md` line from `.gitignore`.
- **M2.** Ask the user what the `Sportswear` folder is (a real brand missing from Postgres, or a generic label) and record the answer in `listings/CLAUDE.md`.
- **M3.** DONE 2026-10-04. Root `CLAUDE.md` now names the test command.
- **M4.** Check whether `torch` and `torchvision` can leave `requirements.txt` (the GPU box gets torch through `unsloth`); change only after the listings builder confirms.
- **M6.** MLflow server — STARTED 2026-10-04 by the manager over SSH (compose file copied to `/home/ec2-user/stack/docker-compose.yml`, backup beside it, `mlflow` database already created by the user). Still open: the first real run, which also tests S3 write access. Original notes follow. Done by the manager 2026-10-04: `mlflow` service in `docker-compose.yaml` (image `v3.16.1`, verified to exist), `mlflow-skinny==3.16.1` in `requirements.txt`, `MLFLOW_TRACKING_URI` in the root `CLAUDE.md`. **Waiting on the user**: create the `mlflow` database, open port 5000 to the GPU box's private IP only, copy the compose file to the EC2 box and start the service, add `MLFLOW_TRACKING_URI` to `.env` (a blank line for it is in the laptop `.env`). The laptop will use the EC2 public address, not a tunnel, so the EC2 `.env` also needs `MLFLOW_SERVER_ALLOWED_HOSTS` and `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` (values in the root `CLAUDE.md`) and the security group has port 5000 open to `0.0.0.0/0` (the user's choice). To check on first start: the IAM user can write under `mlflow/` in the bucket; a Postgres password with URL-special characters breaks the backend URI.
- **M7.** Judge pilot and rubric sign-off (after L10). `eval.jsonl` is on the laptop (2026-10-05). Blocked on the user: the key in `.env` is a user key (`sk-ant-usr-…`), which the API rejects without a workspace; it needs either a workspace API key (`sk-ant-api03-…`) or `ANTHROPIC_CUSTOM_HEADERS=anthropic-workspace-id: <workspace id>` in `.env`. No Claude call has succeeded yet. Run `python -m listings.judge 10`, show the user the verdicts beside the lines, adjust the rubric until the user agrees, then record the rubric as locked. Also check an oracle (a shop line judged against itself must be `equivalent`) and a null (against an unrelated line must be `wrong`).
- **M5.** First commit of the workflow files: `.claude/roles/`, `SPEC.md`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`, `docker-compose.yaml`. Push only after asking.

## Ideas

Not tasks until the user agrees.

- **MLflow Model Registry** (named versions and a "production" alias for the adapter) once there are several models and one is deployed. Until then adapters are kept as run files.
- **Opt-in integration tests** (`python -m pytest -m live`) that hit Postgres and Weaviate, kept out of the default suite.

## Done

- **V1** unit tests for every function (2026-10-04, uncommitted).
- **M3** root `CLAUDE.md` test command (2026-10-04, untracked file).
- **L2** importable invariants and example helpers (2026-10-04, uncommitted).
- **T5** tag analyzer eval runs in MLflow (2026-10-04, uncommitted).
- **L7** listings training and eval runs in MLflow, with V5 tests (2026-10-04, uncommitted).
- **L9** `log_eval_run` edge cases (2026-10-04, uncommitted).
- **L8** `shop-baseline` run (2026-10-04, uncommitted).
- **L10** LLM judge with V7 tests (2026-10-04, uncommitted).
- **L11** judge follow-ups (2026-10-04, uncommitted).
- **L12** similarity metric, word-level and judge 1–5 (2026-10-05, uncommitted).
- **L13** one MLflow run per adapter (2026-10-05, uncommitted; suite at 596).
- **V6** tests for T5, and the L2 part of V3 (2026-10-04, uncommitted; suite at 294).

Earlier history is in `git log`.
