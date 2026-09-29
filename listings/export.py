import json
import random
from collections import defaultdict
from pathlib import Path

from listings import PROJECT_ROOT, LISTINGS_DIR, TRAIN_FILE, VAL_FILE
from listings.db import fetch_listings_with_photos, THUMBNAIL_POSITION, SECONDARY_POSITION
from listings.images import s3_connect, download_photo
from listings.parse import first_line, facts_for, prompt_text

VAL_FRACTION = 0.1
SEED = 42
PROGRESS_EVERY = 100


def image_block(path: Path) -> dict:
    return {"type": "image", "image": str(path.relative_to(PROJECT_ROOT))}


def training_example(row: dict, thumbnail: Path, secondary: Path) -> dict:
    line = first_line(row["description"])
    prompt = prompt_text(facts_for(row, line))
    return {
        "listing_id": row["id"],
        "messages": [
            {"role": "user", "content": [image_block(thumbnail), image_block(secondary), {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": line}]},
        ],
    }


def split_group_key(row: dict) -> tuple:
    return (row["brand"] or "", row["category"] or "")


def grouped_split(examples_by_group: dict[tuple, list[dict]]) -> tuple[list[dict], list[dict]]:
    total = sum(len(group) for group in examples_by_group.values())
    val_target = round(total * VAL_FRACTION)
    keys = list(examples_by_group)
    random.Random(SEED).shuffle(keys)
    train, val = [], []
    for key in keys:
        destination = val if len(val) < val_target else train
        destination.extend(examples_by_group[key])
    return train, val


def write_jsonl(path: Path, examples: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    rows = fetch_listings_with_photos()
    print(f"listings with thumbnail + secondary photo and 💎 line: {len(rows)}")

    s3 = s3_connect()
    examples_by_group = defaultdict(list)
    for n, row in enumerate(rows, 1):
        thumbnail = download_photo(s3, row["thumbnail_url"], row["id"], THUMBNAIL_POSITION)
        secondary = download_photo(s3, row["secondary_url"], row["id"], SECONDARY_POSITION)
        examples_by_group[split_group_key(row)].append(training_example(row, thumbnail, secondary))
        if n % PROGRESS_EVERY == 0:
            print(f"  {n}/{len(rows)} photos downloaded")

    train, val = grouped_split(examples_by_group)
    LISTINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_FILE, train)
    write_jsonl(VAL_FILE, val)
    print(f"train: {len(train)} -> {TRAIN_FILE}")
    print(f"val:   {len(val)} -> {VAL_FILE}  ({len(examples_by_group)} brand/category groups, none shared)")


if __name__ == "__main__":
    main()
