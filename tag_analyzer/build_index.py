import json

from tag_analyzer import LIBRARY_DIR, LABELS_MANIFEST, IMAGE_SUFFIXES
from tag_analyzer import store


def main():
    with open(LABELS_MANIFEST) as f:
        manifest = json.load(f)

    manifest_names = {entry["filename"] for entry in manifest}
    folder_names = {
        str(p.relative_to(LIBRARY_DIR))
        for p in LIBRARY_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    }
    missing_files = manifest_names - folder_names
    unlabeled = folder_names - manifest_names

    if missing_files:
        raise FileNotFoundError(
            f"manifest entries with no photo on disk: {sorted(missing_files)}"
        )
    if unlabeled:
        print(f"WARNING: photos not in manifest (skipped): {sorted(unlabeled)}")

    with store.connect_from_env() as client:
        store.upload_library_photos(client, manifest, LIBRARY_DIR)
        total = store.library_size(client)

    print(f"\nuploaded {len(manifest)} photos -> {store.COLLECTION} ({total} total)")
    print(f"brands: {sorted({e['brand'] for e in manifest})}")
    print(f"eras:   {sorted({e['era'] for e in manifest})}")


if __name__ == "__main__":
    main()
