import html
import json
import re

DIAMOND = "💎"
INSTRUCTION = "Write the first line of the Depop listing for this item."
ERA_PATTERN = re.compile(r"\b(?:19|20)?(\d0)['’]?s\b|\b(y2k)\b", re.IGNORECASE)
MISSING_COLOR_VALUES = {None, "", "N/A"}


def first_line(description: str) -> str:
    return normalized_line(description.split("\n", 1)[0])


def normalized_line(line: str) -> str:
    text = html.unescape(line).strip()
    if text.startswith(DIAMOND):
        text = DIAMOND + " " + text[len(DIAMOND):].lstrip()
    if not text.endswith("."):
        text = text + "."
    return text


def era_from_line(line: str) -> str | None:
    match = ERA_PATTERN.search(line)
    if not match:
        return None
    if match.group(2):
        return "00s"
    return match.group(1) + "s"


def facts_for(row: dict, line: str) -> dict:
    return {
        "brand": row["brand"],
        "era": era_from_line(line),
        "category": row["category"],
        "gender": row["gender"],
        "color": None if row["color"] in MISSING_COLOR_VALUES else row["color"],
    }


def prompt_text(facts: dict) -> str:
    return json.dumps(facts, ensure_ascii=False) + "\n" + INSTRUCTION


def facts_from_prompt_text(text: str) -> dict:
    return json.loads(text.split("\n", 1)[0])
