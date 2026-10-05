import re
import statistics
from collections import Counter

from listings import EVAL_FILE, semantic, tracking
from listings.examples import read_jsonl

APOSTROPHES = str.maketrans("", "", "'\u2019")
WORD_PATTERN = re.compile(r"[^\W_]+")
LOWEST_PAIRS_SHOWN = 5


def words(line: str) -> list[str]:
    return WORD_PATTERN.findall(line.lower().translate(APOSTROPHES))


def exact_match(target: str, generated: str) -> bool:
    return words(target) == words(generated)


def word_overlap(target: str, generated: str) -> float:
    target_words, generated_words = Counter(words(target)), Counter(words(generated))
    shared = sum((target_words & generated_words).values())
    if shared == 0:
        return 0.0
    precision = shared / sum(generated_words.values())
    recall = shared / sum(target_words.values())
    return 2 * precision * recall / (precision + recall)


def scored_row(row: dict) -> dict:
    return {
        "listing_id": row["listing_id"],
        "target": row["target"],
        "generated": row["generated"],
        "exact_match": exact_match(row["target"], row["generated"]),
        "word_overlap": word_overlap(row["target"], row["generated"]),
    }


def scored_rows(rows: list[dict]) -> list[dict]:
    return [scored_row(row) for row in rows]


def similarity_summary(rows: list[dict]) -> dict:
    scored = scored_rows(rows)
    overlaps = [row["word_overlap"] for row in scored]
    return {
        "exact_match": sum(row["exact_match"] for row in scored) / len(scored),
        "word_overlap_mean": statistics.mean(overlaps),
        "word_overlap_median": statistics.median(overlaps),
        "rows": len(scored),
    }


def lowest_scoring(rows: list[dict], count: int = LOWEST_PAIRS_SHOWN) -> list[dict]:
    return sorted(scored_rows(rows), key=lambda row: row["word_overlap"])[:count]


def summary_lines(summary: dict) -> list[str]:
    exact_count = round(summary["exact_match"] * summary["rows"])
    return [
        f"{'exact match':<20} {exact_count}/{summary['rows']} = {summary['exact_match']:.1%}",
        f"{'word overlap mean':<20} {summary['word_overlap_mean']:.3f}",
        f"{'word overlap median':<20} {summary['word_overlap_median']:.3f}",
    ]


def pair_lines(row: dict) -> list[str]:
    return [
        f"{row['listing_id']:>5}  shop:      {row['target']}",
        f"       generated: {row['generated']}",
        f"       overlap:   {row['word_overlap']:.2f}",
        "",
    ]


def main():
    rows = read_jsonl(EVAL_FILE)
    summary = similarity_summary(rows)
    semantic_scored = semantic.semantic_rows(semantic.load_embedder(), rows)
    summary = {**summary, **semantic.semantic_summary(semantic_scored)}
    print("\n".join(summary_lines(summary) + semantic.summary_lines(summary)))
    print(f"\nlowest {LOWEST_PAIRS_SHOWN} by word overlap:\n")
    for row in lowest_scoring(rows):
        print("\n".join(pair_lines(row)))
    print(f"lowest {semantic.LOWEST_PAIRS_SHOWN} by semantic score:\n")
    for row in semantic.lowest_scoring(semantic_scored):
        print("\n".join(semantic.pair_lines(row)))
    tracking.log_similarity_run(summary)


if __name__ == "__main__":
    main()
