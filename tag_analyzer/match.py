import json
import sys

import numpy as np

from tag_analyzer import QUERIES_DIR, VECTORS_FILE, LABELS_FILE
from tag_analyzer.embed import embed_image
from tag_analyzer.voting import top_k, vote
from tag_analyzer.schema import TagMatch

THRESHOLD = 0.75
K = 3

matrix = np.load(VECTORS_FILE)["vectors"]
with open(LABELS_FILE) as f:
    labels = json.load(f)


def match(query_photo) -> TagMatch:
    q = embed_image(query_photo)
    sims = matrix @ q
    neighbors = top_k(sims, labels, K)

    if neighbors[0].similarity < THRESHOLD:
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
            p for p in QUERIES_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not candidates:
            print(f"no query photos found in {QUERIES_DIR} — drop one in, or pass a path")
            sys.exit(1)
        photo = candidates[0]

    print(f"matching: {photo}\n")
    result = match(photo)
    print(result.model_dump_json(indent=2))