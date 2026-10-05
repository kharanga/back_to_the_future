import json

from tag_analyzer import TEST_LABELS_FILE, QUERIES_DIR
from tag_analyzer.match import THRESHOLD
from tag_analyzer.schema import EvalScore, Neighbor
from tag_analyzer.voting import vote
from tag_analyzer import store, tracking

K = 3


def neighbors_for_each_photo(client, test_labels: list[dict]) -> dict[str, list[Neighbor]]:
    return {
        label["filename"]: store.search_nearest_photos(client, QUERIES_DIR / label["filename"], K)
        for label in test_labels
    }


def score(test_labels: list[dict], neighbors_by_filename: dict[str, list[Neighbor]]) -> EvalScore:
    correct_brand = correct_era = 0
    mistakes = []

    for label in test_labels:
        predicted_brand, predicted_era, _ = vote(neighbors_by_filename[label["filename"]])
        brand_ok = predicted_brand.lower() == label["brand"].lower()
        era_ok = predicted_era.lower() == label["era"].lower()
        correct_brand += brand_ok
        correct_era += era_ok
        if not (brand_ok and era_ok):
            mistakes.append(f"{label['filename']} -> {predicted_brand}/{predicted_era}")

    return EvalScore(
        test_set_size=len(test_labels),
        correct_brand=correct_brand,
        correct_era=correct_era,
        mistakes=mistakes,
    )


def accuracy(correct: int, total: int) -> float:
    return correct / total


def accuracy_lines(eval_score: EvalScore) -> list[str]:
    total = eval_score.test_set_size
    lines = [
        f"brand accuracy: {eval_score.correct_brand}/{total} = {accuracy(eval_score.correct_brand, total):.1%}",
        f"era accuracy:   {eval_score.correct_era}/{total} = {accuracy(eval_score.correct_era, total):.1%}",
    ]
    if eval_score.mistakes:
        lines.append("\nmistakes:")
        lines.extend(f"  {mistake}" for mistake in eval_score.mistakes)
    return lines


def run_settings(library_size: int, test_set_size: int) -> dict:
    return {
        "K": K,
        "THRESHOLD": THRESHOLD,
        "clip_model": store.CLIP_MODEL,
        "library_size": library_size,
        "test_set_size": test_set_size,
    }


def run_metrics(eval_score: EvalScore) -> dict[str, float]:
    return {
        "brand_accuracy": accuracy(eval_score.correct_brand, eval_score.test_set_size),
        "era_accuracy": accuracy(eval_score.correct_era, eval_score.test_set_size),
    }


def main():
    with open(TEST_LABELS_FILE) as test_labels_file:
        test_labels = json.load(test_labels_file)

    with store.connect_from_env() as client:
        library_size = store.library_size(client)
        print(f"library: {library_size} photos | test set: {len(test_labels)}")
        neighbors_by_filename = neighbors_for_each_photo(client, test_labels)

    eval_score = score(test_labels, neighbors_by_filename)
    for line in accuracy_lines(eval_score):
        print(line)

    tracking.log_eval_run(
        run_settings(library_size, eval_score.test_set_size),
        run_metrics(eval_score),
        eval_score.mistakes,
    )


if __name__ == "__main__":
    main()
