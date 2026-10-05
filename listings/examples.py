import json
from pathlib import Path

from PIL import Image

from listings import PROJECT_ROOT
from listings.parse import facts_from_prompt_text


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def open_photo(relative_path: str) -> Image.Image:
    return Image.open(PROJECT_ROOT / relative_path).convert("RGB")


def user_blocks(example: dict) -> list[dict]:
    return example["messages"][0]["content"]


def photo_paths(example: dict) -> list[str]:
    return [block["image"] for block in user_blocks(example) if block["type"] == "image"]


def facts(example: dict) -> dict:
    text_block = next(block for block in user_blocks(example) if block["type"] == "text")
    return facts_from_prompt_text(text_block["text"])


def target_line(example: dict) -> str:
    return example["messages"][1]["content"][0]["text"]
