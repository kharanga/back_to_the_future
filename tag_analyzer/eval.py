import json

from tag_analyzer import TEST_LABELS_FILE, QUERIES_DIR
from tag_analyzer.voting import vote
from tag_analyzer import store

K = 3


def main():
    with open(TEST_LABELS_FILE) as f:
        test_labels = json.load(f)

    correct_brand = correct_era = 0
    mistakes = []

    with store.connect_from_env() as client:
        print(f"library: {store.library_size(client)} photos | test set: {len(test_labels)}")
        for label in test_labels:
            neighbors = store.search_nearest_photos(client, QUERIES_DIR / label["filename"], K)
            predicted_brand, predicted_era, _ = vote(neighbors)
            brand_ok = predicted_brand.lower() == label["brand"].lower()
            era_ok = predicted_era.lower() == label["era"].lower()
            correct_brand += brand_ok
            correct_era += era_ok
            if not (brand_ok and era_ok):
                mistakes.append(f"{label['filename']} -> {predicted_brand}/{predicted_era}")

    n = len(test_labels)
    print(f"brand accuracy: {correct_brand}/{n} = {correct_brand / n:.1%}")
    print(f"era accuracy:   {correct_era}/{n} = {correct_era / n:.1%}")
    if mistakes:
        print("\nmistakes:")
        for mistake in mistakes:
            print(f"  {mistake}")


if __name__ == "__main__":
    main()
