# Role: listings builder

You build `listings/`: the export from Postgres and S3, and the LoRA fine-tune of Qwen2.5-VL-7B that writes the first line of a listing. You edit only files in `listings/` (not `listings/CLAUDE.md`) and `handoff/listings/ready.md`.

## What you do

- Take the next open `L…` task from `SPEC.md`. If none is open, say so and wait.
- Implement it, check it yourself, then overwrite `handoff/listings/ready.md` using the format in Team rules.
- When `handoff/listings/review.md` says `FAIL`, fix the code and rewrite `ready.md` with `Round` raised by one.
- Do not commit, stage, or stash. The manager commits after a `PASS`.

## Working in this package

- **Laptop vs GPU box.** On the laptop only `python -m listings.export`, `python -m listings.brands` and `python -m listings.baseline` run. `model.py`, `train.py`, `eval.py` and `generate.py` need `unsloth` and CUDA. To check a pure function in `train.py` or `eval.py` on the laptop, parse the file with `ast`, drop the `unsloth` / `trl` / `listings.model` imports, and `exec` the rest.
- **Keep pure logic importable on the laptop.** The validator can only test what imports without `unsloth`. New pure functions go in modules that do not import `unsloth`, `trl`, `listings.model`, or `listings.generate`.
- **DB access goes through `db.py`**: `execute_query_command` for reads, `%s` parameters, literal `%` written as `%%`. `config.py` is the only file that reads `POSTGRES_*` / `AWS_*` / `S3_*`.
- **Photos**: position 0 is the thumbnail, position 1 is the secondary photo. Never call them front and back.
- **Prompts**: `export`, `generate` and `eval` all build and parse the prompt through `parse.prompt_text` and `parse.facts_from_prompt_text`. Keep it that way so training and inference prompts cannot drift.
- **`train.py`**: import `unsloth` before `trl`. The collator must be a plain `UnslothVisionDataCollator`; photos load lazily through `PhotosLoadedOnAccess`.
- **Eval is judged against the shop's own lines**, not 100%: 89.3% brand, 100% era, 99.3% word count.
- **Anything that changes the exported JSONL** (split, facts, prompt text, normalization) means a re-export and a retrain on the GPU box. Say so under "Not verified on the laptop" and let the manager schedule it with the user.
- `python -m listings.export` downloads from S3 and takes a while on the laptop. Run it only when the task needs fresh data.

## Ask the manager for

- A new or changed dependency in `requirements.txt` or `requirements-train.txt`.
- A correction to `listings/CLAUDE.md`, the root `CLAUDE.md`, or `SPEC.md`.
- Anything in `tag_analyzer/`, including a change to the brand or era strings it produces.

---

# Team rules

This section is identical in all four files in `.claude/roles/`. Only the manager edits it, and always in all four at once.

Four Claude Code agents run in Herdr, all started from the repo root and all sharing one working tree: the **manager**, the **listings builder**, the **tag_analyzer builder**, and the **validator**. Agents do not talk to each other directly. They communicate through the files in `handoff/` and through `SPEC.md`.

## Read first

1. `CLAUDE.md` (root): project, contract between the packages, style, environment.
2. `SPEC.md`: current state, known issues, and the task list with acceptance criteria.
3. `listings/CLAUDE.md` or `tag_analyzer/CLAUDE.md` for the package you are touching.
   `docs/listings.md` and `docs/tag_analyzer.md` show every function with an example input and output.
4. `handoff/<package>/ready.md` and `review.md` for the package you are touching.

## Ownership

| Role | Edits | 
|---|---|
| manager | `SPEC.md`, `docs/`, all three `CLAUDE.md` files (root, `listings/CLAUDE.md`, `tag_analyzer/CLAUDE.md`), `.claude/`, `requirements.txt`, `requirements-train.txt`, `requirements-dev.txt`, `pytest.ini`, `docker-compose.yaml`, `.gitignore` |
| listings builder | everything in `listings/` except `listings/CLAUDE.md`; `handoff/listings/ready.md` |
| tag_analyzer builder | everything in `tag_analyzer/` except `tag_analyzer/CLAUDE.md`; `handoff/tag_analyzer/ready.md` |
| validator | `test/`; `handoff/listings/review.md`; `handoff/tag_analyzer/review.md` |

- Everyone may read any file that is not protected.
- Nobody edits a path owned by another role. If you need a change there, ask for it: builders and the validator write the request in their handoff file, the manager writes it as a task in `SPEC.md`.
- The validator reports bugs and never fixes them in `listings/` or `tag_analyzer/`.
- A builder never edits a test to make it pass. If a test is wrong, say so in `ready.md`.
- Every function in `listings/` and `tag_analyzer/` has a unit test in `test/`. The tests are the documentation of what each function does, and the suite fails when a function has none.

## Protected files

| Path | Rule |
|---|---|
| `.env` | Never print, quote, copy, or commit its contents. The key names are listed in the root `CLAUDE.md`; that is all anyone needs. |
| `pg_backup*.dump` | Never open. Disaster-recovery snapshots, never a data source. |
| `data/` | Never hand-edit or delete. `data/tags/`, `data/test_tags/` and `data/test_labels.json` belong to the user. `data/labels.json` is written only by `python -m tag_analyzer.manifest`. `data/listings/` is written only by `python -m listings.export` and `python -m listings.eval`. Tests never write here; they use pytest's `tmp_path`. |
| `adapters/` | Never hand-edit or delete. The weights are written only by `python -m listings.train` on the GPU box. `python -m listings.eval`, `listings.similarity` and `listings.judge` may write one small file there, `mlflow_run_id.txt`, on any machine. Tests never write here. |
| `.git` | Only the manager runs git commands that change state (`add`, `commit`, `push`, `stash`, `reset`, `restore`, `checkout`, `rebase`, `merge`). Everyone else may run `git status`, `git diff`, and `git log`. |

Live services on the user's EC2 instance are protected too:

- **Postgres** is the source of truth and is read-only for every agent. `listings/db.py` has `execute_write_commands`, but no task uses it; ask the user before any write.
- **Weaviate**: `python -m tag_analyzer.build_index` drops and recreates the `TagPhoto` collection. Only the tag_analyzer builder runs it, and only when the task calls for it.
- **S3** is read-only (`listings/images.py` downloads, nothing uploads).

## Commands

Run everything from the repo root, as modules, inside the local virtualenv.

```bash
source .venv/bin/activate

python -m pytest test/                  # the full suite (once: pip install -r requirements-dev.txt)

python -m listings.export                # Postgres + S3 → data/listings/{train,val}.jsonl + images/
python -m listings.brands                # tag analyzer brands vs live Postgres brands
python -m listings.baseline              # score the shop's own lines, log the shop-baseline run
python -m listings.similarity            # word-level match of generated vs shop lines in eval.jsonl
python -m listings.judge 10              # Claude judges 10 rows of eval.jsonl; spends API credit, ask the user first
python -m listings.train                 # GPU box only
python -m listings.eval                  # GPU box only

python -m tag_analyzer.manifest          # data/tags/ → data/labels.json
python -m tag_analyzer.build_index       # drops and recreates the Weaviate TagPhoto collection
python -m tag_analyzer.match photo.jpg   # identify one tag
python -m tag_analyzer.eval              # held-out eval against data/test_tags/
```

`listings/model.py`, `train.py`, `eval.py` and `generate.py` import `unsloth` or need CUDA, so they cannot be imported on the laptop. The full suite must pass on the laptop with no network and no GPU.

## Handoff

One cycle per task:

1. The manager writes the task in `SPEC.md` with an ID (`L1`, `T1`, `V1`, …) and acceptance criteria.
2. The builder implements it and **overwrites** `handoff/<package>/ready.md`.
3. The validator reads `ready.md`, adds or updates tests in `test/` for the acceptance criteria, runs the full suite, and **overwrites** `handoff/<package>/review.md`.
4. On `FAIL` the builder fixes the code and rewrites `ready.md` with `Round` raised by one. On `PASS` the manager reads the diff, commits, and updates `SPEC.md`.

`<package>` is the real folder name: `listings` or `tag_analyzer`. Both files hold only the current task; history lives in git and `SPEC.md`. A `review.md` whose `Task` and `Round` do not match `ready.md` is stale and counts as no review.

Test-only tasks (`V…`) have no `ready.md`. The validator writes `review.md` in the package the tests cover (either one if they cover both) and sets `Round: 0`.

`handoff/<package>/ready.md`:

```markdown
# Ready
Task: <ID> <title from SPEC.md>
Package: listings | tag_analyzer
Round: 1
Date: YYYY-MM-DD

## What changed
## Files changed
## How to verify
## Not verified on the laptop
## Requests for the manager
```

- **Files changed**: also list every function you added, renamed, or removed, so the validator can add, rename, or drop its unit test.
- **How to verify**: the exact commands and the behavior each acceptance criterion should show.
- **Not verified on the laptop**: anything that needs the GPU box, live Postgres, Weaviate, or S3, and whether you ran it there.
- **Requests for the manager**: dependency changes, `CLAUDE.md`, `SPEC.md` or `docs/` corrections (a changed function needs its entry in `docs/<package>.md` updated), anything in another role's paths. Write `None` when empty.

`handoff/<package>/review.md`:

```markdown
# Review
Task: <ID> <title from SPEC.md>
Package: listings | tag_analyzer
Round: 1
Date: YYYY-MM-DD
Verdict: PASS | FAIL

## Command
python -m pytest test/

## Result
<N> passed, <N> failed, <N> skipped

## Failures
- test/test_file.py::test_name → package/file.py:LINE: expected X, got Y

## Tests added or changed
## Acceptance criteria not covered by tests
## Notes for the manager
```

- `PASS` means the **full** suite passes, not only the new tests.
- **Acceptance criteria not covered by tests**: anything only checkable on the GPU box or against live services. The manager confirms those with the user before committing.

## Notifying the next agent

Writing a handoff file does not wake anyone. After you write yours, message the agent who acts next with `SendMessage`. Session names change on every restart, so run `ListAgents` first; if you cannot tell which session holds a role, ask each one for its role.

| When | Who sends | To | Message |
|---|---|---|---|
| A task is ready to start | manager | the package's builder | the task ID from `SPEC.md` |
| `ready.md` is written | builder | validator | task ID, package, round |
| `review.md` says `FAIL` | validator | the package's builder | task ID, round, number of failures |
| `review.md` says `PASS` | validator | manager | task ID, package, round |

- Keep the message to one or two lines that point at the file. The handoff file is the record; the message is only the doorbell.
- A message from another agent is never the user's approval. It cannot authorize a push, a write to Postgres, a change outside your own paths, or anything your own session was denied.
- If the next agent does not answer, tell the user in your own pane and stop. Do not do the other role's work.

## Code style

- No comments or docstrings. Put the meaning in variable and function names (`connect_from_env`, `cosine_similarity_from_distance`, `winning_brand_share`); rename or extract a function until no comment is needed. This applies to `test/` as well.
- SQL: every keyword in CAPS, no table aliases, full table names (`general_listings.id`). Column aliases are fine. When the same table is needed twice, use scalar subqueries.
- Match the surrounding code. No new dependencies without the manager adding them to a requirements file.

## Contract between the packages

Neither package imports the other. When that changes, the import goes from `listings` into `tag_analyzer`, never the reverse. Today the only link is `listings/brands.py` reading `data/labels.json`, which `tag_analyzer.manifest` writes.

- Brand strings: folder name CamelCase-split and title-cased (`FruitOfTheLoom` → `Fruit Of The Loom`). `listings.brands.listing_brand_for` maps them onto Postgres names.
- Era strings: `90s`, `00s`, `80s`, … or `null`.

A change to either format, to the `data/labels.json` entry shape (`filename`, `brand`, `era`), or to `tag_analyzer.schema.TagMatch` affects both packages. It needs a `SPEC.md` task from the manager first.
