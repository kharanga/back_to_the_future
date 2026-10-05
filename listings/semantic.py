import math
import statistics

from listings.parse import DIAMOND

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CURLY_APOSTROPHE = "’"
HIGHEST_SCORE = 100.0
LOWEST_PAIRS_SHOWN = 5


def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def text_to_embed(line: str) -> str:
    return line.replace(DIAMOND, "").replace(CURLY_APOSTROPHE, "'").strip()


def embedded(embedder, lines: list[str]):
    return embedder.encode([text_to_embed(line) for line in lines], normalize_embeddings=True)


def vector_length(vector) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def cosine_similarity(first_vector, second_vector) -> float:
    lengths_multiplied = vector_length(first_vector) * vector_length(second_vector)
    if lengths_multiplied == 0:
        return 0.0
    dot_product = sum(float(first) * float(second) for first, second in zip(first_vector, second_vector))
    return dot_product / lengths_multiplied


def semantic_score(first_vector, second_vector) -> float:
    return min(HIGHEST_SCORE, max(0.0, cosine_similarity(first_vector, second_vector) * HIGHEST_SCORE))


def rotated_by_one(items) -> list:
    return list(items[1:]) + list(items[:1])


def mismatched_scores(target_vectors, generated_vectors) -> list[float]:
    if len(target_vectors) < 2:
        return []
    another_rows_target = rotated_by_one(target_vectors)
    return [semantic_score(target, generated) for target, generated in zip(another_rows_target, generated_vectors)]


def with_mismatched_score(row: dict, mismatched: list[float], index: int) -> dict:
    return {**row, "semantic_mismatched": mismatched[index]} if mismatched else row


def semantic_rows(embedder, rows: list[dict]) -> list[dict]:
    target_vectors = embedded(embedder, [row["target"] for row in rows])
    generated_vectors = embedded(embedder, [row["generated"] for row in rows])
    mismatched = mismatched_scores(target_vectors, generated_vectors)
    return [
        with_mismatched_score(
            {**row, "semantic": semantic_score(target_vectors[index], generated_vectors[index])}, mismatched, index
        )
        for index, row in enumerate(rows)
    ]


def semantic_summary(scored_rows: list[dict]) -> dict:
    scores = [row["semantic"] for row in scored_rows]
    mismatched = [row["semantic_mismatched"] for row in scored_rows if "semantic_mismatched" in row]
    summary = {"semantic_mean": statistics.mean(scores), "semantic_median": statistics.median(scores)}
    if mismatched:
        summary["semantic_mismatched_mean"] = statistics.mean(mismatched)
    return summary


def lowest_scoring(scored_rows: list[dict], count: int = LOWEST_PAIRS_SHOWN) -> list[dict]:
    return sorted(scored_rows, key=lambda row: row["semantic"])[:count]


def summary_lines(summary: dict) -> list[str]:
    lines = [
        f"{'semantic mean':<20} {summary['semantic_mean']:.1f} out of 100",
        f"{'semantic median':<20} {summary['semantic_median']:.1f}",
    ]
    if "semantic_mismatched_mean" in summary:
        lines.append(f"{'semantic mismatched':<20} {summary['semantic_mismatched_mean']:.1f} (each line against another row's shop line)")
    return lines


def pair_lines(row: dict) -> list[str]:
    return [
        f"{row['listing_id']:>5}  shop:      {row['target']}",
        f"       generated: {row['generated']}",
        f"       semantic:  {row['semantic']:.1f}",
        "",
    ]
