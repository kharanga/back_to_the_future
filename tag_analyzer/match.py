import sys

from tag_analyzer import QUERIES_DIR, IMAGE_SUFFIXES
from tag_analyzer.voting import vote
from tag_analyzer.schema import TagMatch
from tag_analyzer import store

THRESHOLD = 0.70
K = 3


def match(client, query_photo) -> TagMatch:
    neighbors = store.search_nearest_photos(client, query_photo, K)

    if not neighbors or neighbors[0].similarity < THRESHOLD:
        return TagMatch(confidence=0.0, neighbors=neighbors, in_library=False)

    brand, era, confidence = vote(neighbors)
    return TagMatch(
        brand=brand,
        era=era,
        confidence=confidence,
        neighbors=neighbors,
        in_library=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        photo = sys.argv[1]
    else:
        candidates = sorted(
            p for p in QUERIES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not candidates:
            print(f"no query photos found in {QUERIES_DIR} — drop one in, or pass a path")
            sys.exit(1)
        photo = candidates[0]

    print(f"matching: {photo}\n")
    with store.connect_from_env() as client:
        result = match(client, photo)
    print(result.model_dump_json(indent=2))
