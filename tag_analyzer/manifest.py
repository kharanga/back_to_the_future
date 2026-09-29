import json
import re
import sys
from pathlib import Path

from tag_analyzer import LIBRARY_DIR, LABELS_MANIFEST, IMAGE_SUFFIXES

TAGS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else LIBRARY_DIR
OUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else LABELS_MANIFEST

def to_brand(token: str) -> str:
    token = token.replace("_", " ")
    token = re.sub(r"(?<!^)(?=[A-Z])", " ", token)
    return re.sub(r"\s+", " ", token).strip().title()


def era_from_name(fname: str) -> str | None:
    m = re.search(r"(\d{2})s?_", fname)
    return m.group(1) + "s" if m else None


def main():
    entries = []
    for path in sorted(TAGS_DIR.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(TAGS_DIR)
        era = era_from_name(rel.name)
        if era is None:
            print(f"SKIP (no era token): {rel}")
            continue

        if len(rel.parts) > 1:
            brand = to_brand(rel.parts[0])
        else:
            m = re.match(r"(.+?)_\d{2}s?_", rel.name)
            if not m:
                print(f"SKIP (no brand token): {rel}")
                continue
            brand = to_brand(m.group(1))

        entries.append({"filename": str(rel), "brand": brand, "era": era})

    with open(OUT_FILE, "w") as f:
        json.dump(entries, f, indent=2)

    brands = sorted({e["brand"] for e in entries})
    print(f"wrote {len(entries)} entries to {OUT_FILE}")
    print(f"brands ({len(brands)}): {brands}")
    print(f"eras: {sorted({e['era'] for e in entries})}")


if __name__ == "__main__":
    main()