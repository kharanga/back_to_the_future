import json

import pytest

from tag_analyzer import manifest
from tag_analyzer.manifest import era_from_name, to_brand


@pytest.fixture
def tags_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "TAGS_DIR", tmp_path / "tags")
    monkeypatch.setattr(manifest, "OUT_FILE", tmp_path / "labels.json")
    (tmp_path / "tags").mkdir()
    return tmp_path / "tags"


def add_file(tags_dir, relative_name: str):
    (tags_dir / relative_name).parent.mkdir(parents=True, exist_ok=True)
    (tags_dir / relative_name).write_bytes(b"fake photo")


def written_entries(tags_dir) -> list[dict]:
    manifest.main()
    return json.loads((tags_dir.parent / "labels.json").read_text())


def test_to_brand_splits_a_camel_case_folder_name_into_title_cased_words():
    assert to_brand("FruitOfTheLoom") == "Fruit Of The Loom"


def test_to_brand_leaves_a_single_word_alone():
    assert to_brand("Russell") == "Russell"


def test_to_brand_reads_underscores_as_spaces():
    assert to_brand("Screen_Stars") == "Screen Stars"


def test_to_brand_title_cases_a_lowercase_name():
    assert to_brand("champion") == "Champion"


def test_era_from_name_reads_the_two_digit_prefix_as_a_decade():
    assert era_from_name("90_1.PNG") == "90s"


def test_era_from_name_accepts_a_prefix_that_already_ends_in_s():
    assert era_from_name("80s_2.jpg") == "80s"


def test_era_from_name_is_none_when_the_name_has_no_era_token():
    assert era_from_name("tag.jpg") is None


def test_main_writes_filename_brand_and_era_for_a_photo_in_a_brand_folder(tags_dir):
    add_file(tags_dir, "FruitOfTheLoom/90_1.jpg")
    assert written_entries(tags_dir) == [
        {"filename": "FruitOfTheLoom/90_1.jpg", "brand": "Fruit Of The Loom", "era": "90s"}
    ]


def test_main_reads_the_brand_from_the_filename_for_a_photo_outside_a_brand_folder(tags_dir):
    add_file(tags_dir, "Russell_80s_1.jpg")
    assert written_entries(tags_dir) == [{"filename": "Russell_80s_1.jpg", "brand": "Russell", "era": "80s"}]


def test_main_skips_a_photo_with_no_era_token(tags_dir):
    add_file(tags_dir, "Russell/tag.jpg")
    assert written_entries(tags_dir) == []


def test_main_skips_files_that_are_not_images(tags_dir):
    add_file(tags_dir, "Russell/90_1.txt")
    assert written_entries(tags_dir) == []


def test_main_lists_entries_in_sorted_path_order(tags_dir):
    add_file(tags_dir, "Russell/90_1.jpg")
    add_file(tags_dir, "Champion/80_1.jpg")
    assert [entry["filename"] for entry in written_entries(tags_dir)] == ["Champion/80_1.jpg", "Russell/90_1.jpg"]
