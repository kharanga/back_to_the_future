import json

from listings import PROJECT_ROOT
from listings.db import execute_query_command

TAG_LABELS_FILE = PROJECT_ROOT / "data" / "labels.json"

LISTING_BRAND_FOR_TAG_BRAND = {
    "Fruit Of The Loom": "Fruit of the Loom",
}

LISTING_BRANDS = """
SELECT DISTINCT general_listings.brand
  FROM general_listings
 WHERE general_listings.brand IS NOT NULL
"""


def listing_brand_for(tag_brand: str) -> str:
    return LISTING_BRAND_FOR_TAG_BRAND.get(tag_brand, tag_brand)


def tag_analyzer_brands() -> set[str]:
    with open(TAG_LABELS_FILE, encoding="utf-8") as f:
        return {entry["brand"] for entry in json.load(f)}


def listing_brands() -> set[str]:
    return {row["brand"] for row in execute_query_command(LISTING_BRANDS)}


def case_insensitive_matches(brand: str, candidates: set[str]) -> list[str]:
    return sorted(candidate for candidate in candidates if candidate.lower() == brand.lower())


def main():
    known = listing_brands()
    for tag_brand in sorted(tag_analyzer_brands()):
        mapped = listing_brand_for(tag_brand)
        if mapped in known:
            status = "ok" if mapped == tag_brand else f"mapped -> {mapped}"
        elif case_insensitive_matches(tag_brand, known):
            status = f"NEEDS MAPPING -> {case_insensitive_matches(tag_brand, known)}"
        else:
            status = "no listings, passed through"
        print(f"{tag_brand:<20} {status}")


if __name__ == "__main__":
    main()
