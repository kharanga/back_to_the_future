import json

from listings import EVAL_FILE, PROJECT_ROOT, VAL_FILE
from listings.generate import generate_first_line
from listings.model import load_finetuned_model
from listings.parse import DIAMOND, era_from_line, facts_from_prompt_text
from listings.train import open_photo, read_jsonl

MAX_WORDS = 12
UNBRANDED_VALUES = {None, "", "Other"}
BLANK_TAG_BRANDS = {
    "AAA",
    "Anvil",
    "Fruit of the Loom",
    "Gildan",
    "Hanes",
    "Jansport",
    "Jerzees",
    "Lee",
    "Majestic Athletic",
    "NFL",
    "Screen Stars",
    "Soffe",
}
MIN_BRAND_WORD_LENGTH = 3
CURLY_APOSTROPHE = "\u2019"
INVARIANT_NAMES = ["starts_with_diamond", "ends_with_period", "word_count_ok", "brand_mentioned", "era_mentioned"]


def user_blocks(example: dict) -> list[dict]:
    return example["messages"][0]["content"]


def photo_paths(example: dict) -> list[str]:
    return [block["image"] for block in user_blocks(example) if block["type"] == "image"]


def facts(example: dict) -> dict:
    text_block = next(block for block in user_blocks(example) if block["type"] == "text")
    return facts_from_prompt_text(text_block["text"])


def target_line(example: dict) -> str:
    return example["messages"][1]["content"][0]["text"]


def straight_apostrophes(text: str) -> str:
    return text.lower().replace(CURLY_APOSTROPHE, "'")


def brand_mentioned(line: str, brand: str | None) -> bool:
    if brand in UNBRANDED_VALUES or brand in BLANK_TAG_BRANDS:
        return True
    brand_words = [word for word in straight_apostrophes(brand).split() if len(word) >= MIN_BRAND_WORD_LENGTH]
    return any(word in straight_apostrophes(line) for word in brand_words)


def era_mentioned(line: str, era: str | None) -> bool:
    return era is None or era_from_line(line) == era


def invariants(line: str, item_facts: dict) -> dict:
    return {
        "starts_with_diamond": line.startswith(DIAMOND),
        "ends_with_period": line.rstrip().endswith("."),
        "word_count_ok": len(line.split()) <= MAX_WORDS,
        "brand_mentioned": brand_mentioned(line, item_facts["brand"]),
        "era_mentioned": era_mentioned(line, item_facts["era"]),
    }


def evaluate_example(model, tokenizer, example: dict) -> dict:
    thumbnail, secondary = (open_photo(path) for path in photo_paths(example))
    item_facts = facts(example)
    generated = generate_first_line(model, tokenizer, thumbnail, secondary, item_facts)
    return {
        "listing_id": example["listing_id"],
        "target": target_line(example),
        "generated": generated,
        **invariants(generated, item_facts),
    }


def print_summary(results: list[dict]):
    total = len(results)
    for name in INVARIANT_NAMES:
        passed = sum(result[name] for result in results)
        print(f"{name:<20} {passed}/{total} = {passed / total:.1%}")


def main():
    examples = read_jsonl(VAL_FILE)
    model, tokenizer = load_finetuned_model()

    results = []
    for example in examples:
        result = evaluate_example(model, tokenizer, example)
        results.append(result)
        print(f"{result['listing_id']:>5}  target:    {result['target']}")
        print(f"       generated: {result['generated']}\n")

    print_summary(results)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nwritten to {EVAL_FILE}")


if __name__ == "__main__":
    main()
