import json

import numpy as np

from tag_analyzer import (
    LIBRARY_DIR,
    INDEX_DIR,
    LABELS_MANIFEST,
    VECTORS_FILE,
    LABELS_FILE,
)
from tag_analyzer.embed import embed_image

def main():
    INDEX_DIR.mkdir(exist_ok=True)

    with open(LABELS_MANIFEST) as f:
        manifest = json.load(f)

    manifest_names = {entry["filename"] for entry in manifest}
    folder_names = {
        str(p.relative_to(LIBRARY_DIR))
        for p in LIBRARY_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    missing_files = manifest_names - folder_names
    unlabeled = folder_names - manifest_names

    if missing_files:
        raise FileNotFoundError(
            f"manifest entries with no photo on disk: {sorted(missing_files)}"
        )
    if unlabeled:
        print(f"WARNING: photos not in manifest (skipped): {sorted(unlabeled)}")

    vectors = []
    labels = []
    for entry in manifest:
        path = LIBRARY_DIR / entry["filename"]
        vectors.append(embed_image(path))
        labels.append(entry)
        print(f"embedded {entry['filename']}")

    matrix = np.stack(vectors)

    np.savez(VECTORS_FILE, vectors=matrix)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nindexed {len(labels)} photos")
    print(f"brands: {sorted({e['brand'] for e in labels})}")
    print(f"eras:   {sorted({e['era'] for e in labels})}")


if __name__ == "__main__":
    main()