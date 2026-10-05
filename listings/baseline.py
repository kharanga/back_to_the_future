from listings import TRAIN_FILE, VAL_FILE, tracking
from listings.examples import facts, read_jsonl, target_line
from listings.invariants import invariants, print_summary


def target_line_invariants(example: dict) -> dict:
    return invariants(target_line(example), facts(example))


def baseline_results(examples: list[dict]) -> list[dict]:
    return [target_line_invariants(example) for example in examples]


def main():
    examples = read_jsonl(TRAIN_FILE) + read_jsonl(VAL_FILE)
    print(f"shop target lines: {len(examples)}")
    results = baseline_results(examples)
    print_summary(results)
    tracking.log_baseline_run(results)


if __name__ == "__main__":
    main()
