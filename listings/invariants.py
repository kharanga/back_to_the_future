from listings.parse import DIAMOND, era_from_line

MAX_WORDS = 12
UNBRANDED_VALUES = {None, "", "Other"}
BLANK_TAG_BRANDS = {
    "AAA",
    "Anvil",
    "Fruit of the Loom",
    "Gildan",
    "Hanes",
    "Jansport",
    "Jerzees",
    "Lee",
    "Majestic Athletic",
    "NFL",
    "Screen Stars",
    "Soffe",
}
MIN_BRAND_WORD_LENGTH = 3
CURLY_APOSTROPHE = "’"
INVARIANT_NAMES = ["starts_with_diamond", "ends_with_period", "word_count_ok", "brand_mentioned", "era_mentioned"]


def straight_apostrophes(text: str) -> str:
    return text.lower().replace(CURLY_APOSTROPHE, "'")


def brand_mentioned(line: str, brand: str | None) -> bool:
    if brand in UNBRANDED_VALUES or brand in BLANK_TAG_BRANDS:
        return True
    brand_words = [word for word in straight_apostrophes(brand).split() if len(word) >= MIN_BRAND_WORD_LENGTH]
    return any(word in straight_apostrophes(line) for word in brand_words)


def era_mentioned(line: str, era: str | None) -> bool:
    return era is None or era_from_line(line) == era


def invariants(line: str, item_facts: dict) -> dict:
    return {
        "starts_with_diamond": line.startswith(DIAMOND),
        "ends_with_period": line.rstrip().endswith("."),
        "word_count_ok": len(line.split()) <= MAX_WORDS,
        "brand_mentioned": brand_mentioned(line, item_facts["brand"]),
        "era_mentioned": era_mentioned(line, item_facts["era"]),
    }


def pass_count(results: list[dict], invariant_name: str) -> int:
    return sum(result[invariant_name] for result in results)


def invariant_pass_rates(results: list[dict]) -> dict:
    return {name: pass_count(results, name) / len(results) for name in INVARIANT_NAMES}


def print_summary(results: list[dict]):
    pass_rates = invariant_pass_rates(results)
    for name in INVARIANT_NAMES:
        print(f"{name:<20} {pass_count(results, name)}/{len(results)} = {pass_rates[name]:.1%}")
