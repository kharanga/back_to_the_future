import json

from listings import EVAL_FILE, VAL_FILE, tracking
from listings.examples import facts, open_photo, photo_paths, read_jsonl, target_line
from listings.generate import generate_first_line
from listings.invariants import invariants, print_summary
from listings.model import load_finetuned_model


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
    tracking.log_eval_run(results)


if __name__ == "__main__":
    main()
