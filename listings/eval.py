import json

from listings import EVAL_FILE, PROJECT_ROOT, VAL_FILE
from listings.generate import generate_first_line
from listings.model import load_finetuned_model
from listings.parse import DIAMOND, facts_from_prompt_text
from listings.train import open_photo, read_jsonl

MAX_WORDS = 12
UNBRANDED_VALUES = {None, "", "Other"}
MIN_BRAND_WORD_LENGTH = 3


def user_blocks(example: dict) -> list[dict]:
    return example["messages"][0]["content"]


def photo_paths(example: dict) -> list[str]:
    return [block["image"] for block in user_blocks(example) if block["type"] == "image"]


def facts(example: dict) -> dict:
    text_block = next(block for block in user_blocks(example) if block["type"] == "text")
    return facts_from_prompt_text(text_block["text"])


def target_line(example: dict) -> str:
    return example["messages"][1]["content"][0]["text"]


def brand_mentioned(line: str, brand: str | None) -> bool:
    if brand in UNBRANDED_VALUES:
        return True
    brand_words = [word for word in brand.lower().split() if len(word) >= MIN_BRAND_WORD_LENGTH]
    return any(word in line.lower() for word in brand_words)


def invariants(line: str, item_facts: dict) -> dict:
    return {
        "starts_with_diamond": line.startswith(DIAMOND),
        "ends_with_period": line.rstrip().endswith("."),
        "word_count_ok": len(line.split()) <= MAX_WORDS,
        "brand_mentioned": brand_mentioned(line, item_facts["brand"]),
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
    for name in ["starts_with_diamond", "ends_with_period", "word_count_ok", "brand_mentioned"]:
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
