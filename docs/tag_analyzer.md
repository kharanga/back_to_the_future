# `tag_analyzer/` function reference

Every function in the package, with an example input and the output it gives. Written 2026-10-04 against commit `f1b917e`; updated the same day for task T5 (`eval.py` split, `tracking.py`).

- **Ran** means the output was produced by running the function on the laptop.
- **Illustrative** means the function needs Weaviate or the photo library, so the output shows the shape, not a captured run.

The unit tests in `test/tag_analyzer/` are the checked version of these examples.

## How the functions fit together

```
BUILD THE LIBRARY                               once per library change

  data/tags/Champion/90_1.PNG ...
        │
        ▼  python -m tag_analyzer.manifest
  manifest.to_brand("Champion")                 folder name → brand
  manifest.era_from_name("90_1.PNG")            filename → era
        │
        ▼
  data/labels.json                              [{filename, brand, era}, ...]
        │
        ▼  python -m tag_analyzer.build_index
  store.upload_library_photos()                 drops and recreates TagPhoto,
                                                Weaviate embeds each photo with CLIP

IDENTIFY ONE TAG                                python -m tag_analyzer.match photo.jpg

  store.search_nearest_photos(photo, k=3)       3 most similar library photos
        │
        ▼
  top similarity < 0.70 ?  ──yes──▶  TagMatch(in_library=False)
        │ no
        ▼
  voting.vote(neighbors)                        brand, era, confidence
        │
        ▼
  TagMatch(brand, era, confidence, neighbors, in_library=True)

MEASURE ACCURACY                                python -m tag_analyzer.eval

  eval.neighbors_for_each_photo()               search once per held-out photo
  eval.score(labels, neighbors)                 EvalScore: counts + mistakes
  eval.accuracy_lines(score)                    printed
  tracking.log_eval_run(...)                    one MLflow run, when tracking is on
```

The three sample neighbors used throughout:

```python
neighbors = [
    Neighbor(filename="Champion/90_1.PNG", brand="Champion", era="90s", similarity=0.91),
    Neighbor(filename="Champion/80_2.PNG", brand="Champion", era="80s", similarity=0.84),
    Neighbor(filename="Russell/90_1.PNG",  brand="Russell",  era="90s", similarity=0.80),
]
```

---

## `schema.py` — the two result shapes

Not functions, but every other file passes these around.

**`Neighbor`**: one library photo that came back from a search.

| Field | Example |
|---|---|
| `filename` | `"Champion/90_1.PNG"` |
| `brand` | `"Champion"` |
| `era` | `"90s"` |
| `similarity` | `0.91` (1.0 is identical) |

**`TagMatch`**: the answer for one query photo.

| Field | Example | Notes |
|---|---|---|
| `brand` | `"Champion"` | `None` when not in library |
| `era` | `"90s"` | `None` when not in library |
| `confidence` | `0.686` | the winning brand's share of the vote |
| `neighbors` | the list above | always included |
| `in_library` | `True` | `False` when the best match is below the threshold |

**`EvalScore`**: the result of scoring the held-out set.

| Field | Example |
|---|---|
| `test_set_size` | `19` |
| `correct_brand` | `19` |
| `correct_era` | `16` |
| `mistakes` | `["some_test_photo.jpg -> Champion/80s", …]` |

---

## `manifest.py` — labels from folder and file names

### `to_brand(token)` — Ran

Splits CamelCase and underscores, then title-cases.

| Input | Output |
|---|---|
| `"Champion"` | `"Champion"` |
| `"FruitOfTheLoom"` | `"Fruit Of The Loom"` |
| `"ScreenStars"` | `"Screen Stars"` |
| `"polo_ralph_lauren"` | `"Polo Ralph Lauren"` |
| `"NFL"` | `"N F L"` |

The last row is a trap: an all-caps folder name is split letter by letter. A folder for an acronym brand has to be named `Nfl` to come out as `Nfl`, and neither form matches the `NFL` that Postgres uses without a mapping in `listings/brands.py`.

### `era_from_name(fname)` — Ran

Reads the two digits before the first underscore.

| Input | Output |
|---|---|
| `"90_1.PNG"` | `"90s"` |
| `"80s_2.jpg"` | `"80s"` |
| `"00_3.webp"` | `"00s"` |
| `"tag.jpg"` | `None` |

### `main()` — Illustrative

`python -m tag_analyzer.manifest`. Walks `data/tags/`, skips files with no era in the name, and writes `data/labels.json`.

Input on disk:
```
data/tags/Champion/90_1.PNG
data/tags/Champion/80_2.PNG
data/tags/FruitOfTheLoom/90_1.jpg
data/tags/Champion/tag.jpg
```

Output file:
```json
[
  {"filename": "Champion/80_2.PNG", "brand": "Champion", "era": "80s"},
  {"filename": "Champion/90_1.PNG", "brand": "Champion", "era": "90s"},
  {"filename": "FruitOfTheLoom/90_1.jpg", "brand": "Fruit Of The Loom", "era": "90s"}
]
```

Printed:
```
SKIP (no era token): Champion/tag.jpg
wrote 3 entries to /…/data/labels.json
brands (2): ['Champion', 'Fruit Of The Loom']
eras: ['80s', '90s']
```

Two optional arguments override the tags folder and the output file. They are read when the module is imported, not when `main` runs (task T1).

---

## `store.py` — everything that talks to Weaviate

### `connect_from_env()` — Illustrative

No arguments. Returns an open Weaviate client built from the `WEAVIATE_*` values. Used as `with store.connect_from_env() as client:`.

### `recreate_collection(client)` — Illustrative

Deletes the `TagPhoto` collection if it exists and creates it empty, with properties `filename`, `brand`, `era`, `image` and CLIP vectorizing on `image`. Returns the new collection.

### `stable_uuid_for(filename)` — Ran

The same filename always gives the same ID.

| Input | Output |
|---|---|
| `"Champion/90_1.PNG"` | `"9db38c16-02f1-5724-941c-0cf07bee5581"` |
| `"Champion/90_1.PNG"` again | `"9db38c16-02f1-5724-941c-0cf07bee5581"` |

### `encode_image_base64(photo_path)` — Illustrative

| Input | Output |
|---|---|
| `data/tags/Champion/90_1.PNG` | the file's bytes as a base64 string, `"iVBORw0KGgo…"` |

### `upload_library_photos(client, entries, library_dir)` — Illustrative

Recreates the collection, then uploads every manifest entry in batches of 20. Raises if any upload fails.

| Input | Effect |
|---|---|
| client, the `labels.json` entries, `data/tags/` | prints `uploaded Champion/90_1.PNG` per photo; `TagPhoto` now holds one object per entry |

### `cosine_similarity_from_distance(distance)` — Ran

| Input | Output |
|---|---|
| `0.09` | `0.91` |
| `0.0` | `1.0` |

### `search_nearest_photos(client, query_photo, k)` — Illustrative

Sends the query photo to Weaviate and returns the `k` closest library photos, best first, as `Neighbor`s. The photo blobs are never returned.

| Input | Output |
|---|---|
| client, `"photo.jpg"`, `3` | the three sample `neighbors` above |

### `library_size(client)` — Illustrative

| Situation | Output |
|---|---|
| `TagPhoto` holds 120 photos | `120` |
| `TagPhoto` does not exist | `0` |

---

## `voting.py` — picking the winner

### `vote(neighbors)` — Ran

Each neighbor votes for its brand and its era with a weight equal to its similarity. Brand and era are decided separately.

Input: the three sample `neighbors`.

| | Weights | Winner |
|---|---|---|
| brand | Champion 0.91 + 0.84 = 1.75, Russell 0.80 | Champion |
| era | 90s 0.91 + 0.80 = 1.71, 80s 0.84 | 90s |

Output: `("Champion", "90s", 0.686)`

Confidence is the winning brand's share: 1.75 / (1.75 + 0.80) = 0.686.

`vote([])` raises `ValueError`; callers must check for an empty list first.

---

## `match.py` — the answer for one photo

### `match(client, query_photo)` — Ran (with the search replaced by fixed neighbors)

**Best match at or above 0.70** (the three sample neighbors, top similarity 0.91):

```json
{
  "brand": "Champion",
  "era": "90s",
  "confidence": 0.6862745098039216,
  "neighbors": [
    {"filename": "Champion/90_1.PNG", "brand": "Champion", "era": "90s", "similarity": 0.91},
    {"filename": "Champion/80_2.PNG", "brand": "Champion", "era": "80s", "similarity": 0.84},
    {"filename": "Russell/90_1.PNG", "brand": "Russell", "era": "90s", "similarity": 0.8}
  ],
  "in_library": true
}
```

**Best match below 0.70** (one neighbor at 0.55):

```json
{
  "brand": null,
  "era": null,
  "confidence": 0.0,
  "neighbors": [
    {"filename": "Russell/90_1.PNG", "brand": "Russell", "era": "90s", "similarity": 0.55}
  ],
  "in_library": false
}
```

An empty neighbor list gives the same not-in-library result with `"neighbors": []`.

Run as `python -m tag_analyzer.match photo.jpg`, it prints `matching: photo.jpg` and then the JSON. With no argument it uses the first photo in `data/test_tags/`.

---

## `build_index.py` — upload the library

### `main()` — Illustrative

`python -m tag_analyzer.build_index`. **Drops and recreates the `TagPhoto` collection.**

| Situation | Result |
|---|---|
| manifest and folder agree | uploads everything, prints the summary below |
| a photo on disk is not in the manifest | prints `WARNING: photos not in manifest (skipped): […]` and continues |
| a manifest entry has no photo on disk | raises `FileNotFoundError` before uploading anything |

```
uploaded Champion/80_2.PNG
uploaded Champion/90_1.PNG
…
uploaded 120 photos -> TagPhoto (120 total)
brands: ['Champion', 'Fruit Of The Loom', …]
eras:   ['00s', '70s', '80s', '90s']
```

---

## `eval.py` — accuracy on held-out photos

The sample test set, two photos with the neighbors Weaviate returned for each:

```python
test_labels = [
    {"filename": "champion_test_1.jpg", "brand": "Champion", "era": "90s"},
    {"filename": "russell_test_1.jpg",  "brand": "Russell",  "era": "90s"},
]
neighbors_by_filename = {
    "champion_test_1.jpg": [the three sample neighbors],
    "russell_test_1.jpg": [
        Neighbor(filename="Russell/80_1.PNG", brand="Russell", era="80s", similarity=0.88),
        Neighbor(filename="Russell/80_2.PNG", brand="Russell", era="80s", similarity=0.86),
        Neighbor(filename="Russell/90_1.PNG", brand="Russell", era="90s", similarity=0.80),
    ],
}
```

### `neighbors_for_each_photo(client, test_labels)` — Illustrative

Searches Weaviate once per test photo (`K = 3`) and returns the `neighbors_by_filename` dict above.

### `score(test_labels, neighbors_by_filename)` — Ran

Votes on each photo's neighbors and compares with the label, ignoring case. The 0.70 threshold is not applied, so every photo gets a prediction.

Input: the sample test set above.

Output:
```python
EvalScore(test_set_size=2, correct_brand=2, correct_era=1, mistakes=["russell_test_1.jpg -> Russell/80s"])
```

The Russell photo is a 90s tag, but two of its three neighbors are 80s, so the era vote goes to 80s. Each mistake reads `filename -> predicted brand/predicted era`.

### `accuracy(correct, total)` — Ran

| Input | Output |
|---|---|
| `16`, `19` | `0.8421052631578947` |

`total = 0` raises `ZeroDivisionError`.

### `accuracy_lines(eval_score)` — Ran

Input: the `EvalScore` above.

Output:
```python
["brand accuracy: 2/2 = 100.0%", "era accuracy:   1/2 = 50.0%", "\nmistakes:", "  russell_test_1.jpg -> Russell/80s"]
```

With no mistakes, only the two accuracy lines are returned.

### `run_settings(library_size, test_set_size)` — Ran

The settings recorded with an eval run.

| Input | Output |
|---|---|
| `170`, `2` | `{"K": 3, "THRESHOLD": 0.7, "clip_model": "clip-ViT-B-32-multilingual-v1", "library_size": 170, "test_set_size": 2}` |

### `run_metrics(eval_score)` — Ran

| Input | Output |
|---|---|
| the `EvalScore` above | `{"brand_accuracy": 1.0, "era_accuracy": 0.5}` |

### `main()` — Illustrative

`python -m tag_analyzer.eval`. Reads `data/test_labels.json`, searches, scores, prints, then logs the run to MLflow when tracking is on.

Output on the real library (run by the builder 2026-10-04; the mistake line shows the format):
```
library: 170 photos | test set: 19
brand accuracy: 19/19 = 100.0%
era accuracy:   16/19 = 84.2%

mistakes:
  some_test_photo.jpg -> Champion/80s
```

---

## `tracking.py` — recording eval runs in MLflow

Every MLflow call for the package, and the only place `MLFLOW_TRACKING_URI` is read.

### `tracking_enabled()` — Ran

| Situation | Output |
|---|---|
| `MLFLOW_TRACKING_URI` is not in `.env` (today) | `False` |
| `MLFLOW_TRACKING_URI=http://localhost:5000` | `True` |

### `git_commit()` — Ran

No arguments. Returns the 40-character hash of the current commit, for example `"f1b917e2…"`, or `"unknown"` when git is not available.

### `log_eval_run(settings, metrics, mistakes)` — Illustrative

Records one eval as one run in the `tag-analyzer` experiment.

| Input | Output |
|---|---|
| `run_settings(…)`, `run_metrics(…)`, the list of mistake strings; tracking on | the new run's ID, for example `"3f2a9c…"` |
| the same, tracking off | `None`; nothing is logged and `mlflow` is not even imported |

What the run holds in the MLflow UI:

| Kind | Values |
|---|---|
| Parameters | `K`, `THRESHOLD`, `clip_model`, `library_size`, `test_set_size`, `git_commit` (MLflow stores them as text: `"3"`, `"0.7"`) |
| Metrics | `brand_accuracy`, `era_accuracy` |
| Files | `mistakes.txt`, one mistake per line, empty when there are none |
