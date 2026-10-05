import json

from listings import brands
from listings.brands import case_insensitive_matches, listing_brand_for, listing_brands, tag_analyzer_brands


def printed_status_for(tag_brand: str, printed: str) -> str:
    line_for_brand = next(line for line in printed.splitlines() if line.startswith(tag_brand))
    return line_for_brand.removeprefix(tag_brand).strip()


def run_main_with(monkeypatch, tag_brands: set, known_listing_brands: set):
    monkeypatch.setattr(brands, "tag_analyzer_brands", lambda: tag_brands)
    monkeypatch.setattr(brands, "listing_brands", lambda: known_listing_brands)
    brands.main()


def test_listing_brand_for_maps_a_tag_brand_onto_its_postgres_spelling():
    assert listing_brand_for("Fruit Of The Loom") == "Fruit of the Loom"


def test_listing_brand_for_passes_an_unmapped_brand_through_unchanged():
    assert listing_brand_for("Russell") == "Russell"


def test_tag_analyzer_brands_is_the_set_of_brands_in_the_labels_file(tmp_path, monkeypatch):
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(
        json.dumps(
            [
                {"filename": "Russell/90_1.jpg", "brand": "Russell", "era": "90s"},
                {"filename": "Russell/80_1.jpg", "brand": "Russell", "era": "80s"},
                {"filename": "Champion/90_1.jpg", "brand": "Champion", "era": "90s"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(brands, "TAG_LABELS_FILE", labels_file)
    assert tag_analyzer_brands() == {"Russell", "Champion"}


def test_listing_brands_is_the_set_of_brands_the_query_returns(monkeypatch):
    monkeypatch.setattr(brands, "execute_query_command", lambda sql: [{"brand": "Russell"}, {"brand": "Levi's"}])
    assert listing_brands() == {"Russell", "Levi's"}


def test_case_insensitive_matches_finds_candidates_that_differ_only_in_case():
    assert case_insensitive_matches("Fruit Of The Loom", {"Fruit of the Loom", "Russell"}) == ["Fruit of the Loom"]


def test_case_insensitive_matches_is_empty_when_nothing_matches():
    assert case_insensitive_matches("Sportswear", {"Fruit of the Loom", "Russell"}) == []


def test_main_reports_ok_for_a_tag_brand_spelled_like_postgres(monkeypatch, capsys):
    run_main_with(monkeypatch, tag_brands={"Russell"}, known_listing_brands={"Russell"})
    assert printed_status_for("Russell", capsys.readouterr().out) == "ok"


def test_main_reports_mapped_for_a_tag_brand_with_a_mapping(monkeypatch, capsys):
    run_main_with(monkeypatch, tag_brands={"Fruit Of The Loom"}, known_listing_brands={"Fruit of the Loom"})
    assert printed_status_for("Fruit Of The Loom", capsys.readouterr().out) == "mapped -> Fruit of the Loom"


def test_main_reports_needs_mapping_for_an_unmapped_case_difference(monkeypatch, capsys):
    run_main_with(monkeypatch, tag_brands={"Tommy Hilfiger"}, known_listing_brands={"TOMMY HILFIGER"})
    assert printed_status_for("Tommy Hilfiger", capsys.readouterr().out) == "NEEDS MAPPING -> ['TOMMY HILFIGER']"


def test_main_reports_passed_through_for_a_tag_brand_with_no_listings(monkeypatch, capsys):
    run_main_with(monkeypatch, tag_brands={"Sportswear"}, known_listing_brands={"Russell"})
    assert printed_status_for("Sportswear", capsys.readouterr().out) == "no listings, passed through"
