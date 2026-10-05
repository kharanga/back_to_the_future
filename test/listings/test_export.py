import json

from listings import PROJECT_ROOT, export
from listings.export import grouped_split, image_block, split_group_key, training_example, write_jsonl
from listings.parse import facts_from_prompt_text

THUMBNAIL = PROJECT_ROOT / "data" / "listings" / "images" / "7_0.jpg"
SECONDARY = PROJECT_ROOT / "data" / "listings" / "images" / "7_1.jpg"
RUSSELL_HOODIE_ROW = {
    "id": 7,
    "brand": "Russell",
    "category": "Hoodies",
    "gender": "Men",
    "color": "Grey",
    "description": "💎 Vintage 90's Boxy Russell Hoodie.\n📏 Pit to pit 24\n⚠️ No flaws\n⛔️ No returns",
}


def russell_hoodie_example() -> dict:
    return training_example(RUSSELL_HOODIE_ROW, THUMBNAIL, SECONDARY)


def groups_of(group_count: int, examples_per_group: int) -> dict:
    return {
        (f"Brand {group}", "Hoodies"): [
            {"listing_id": group * examples_per_group + item} for item in range(examples_per_group)
        ]
        for group in range(group_count)
    }


def listing_ids(examples: list[dict]) -> set:
    return {example["listing_id"] for example in examples}


def test_image_block_holds_the_photo_path_relative_to_the_repo_root():
    assert image_block(THUMBNAIL) == {"type": "image", "image": "data/listings/images/7_0.jpg"}


def test_training_example_carries_the_listing_id():
    assert russell_hoodie_example()["listing_id"] == 7


def test_training_example_has_a_user_message_then_an_assistant_message():
    roles = [message["role"] for message in russell_hoodie_example()["messages"]]
    assert roles == ["user", "assistant"]


def test_training_example_user_message_is_thumbnail_then_secondary_then_text():
    user_content = russell_hoodie_example()["messages"][0]["content"]
    assert user_content[0] == {"type": "image", "image": "data/listings/images/7_0.jpg"}
    assert user_content[1] == {"type": "image", "image": "data/listings/images/7_1.jpg"}
    assert [block["type"] for block in user_content] == ["image", "image", "text"]


def test_training_example_prompt_holds_the_facts_for_the_row():
    prompt = russell_hoodie_example()["messages"][0]["content"][2]["text"]
    assert facts_from_prompt_text(prompt) == {
        "brand": "Russell",
        "era": "90s",
        "category": "Hoodies",
        "gender": "Men",
        "color": "Grey",
    }


def test_training_example_assistant_message_is_only_the_diamond_line():
    assistant_content = russell_hoodie_example()["messages"][1]["content"]
    assert assistant_content == [{"type": "text", "text": "💎 Vintage 90's Boxy Russell Hoodie."}]


def test_split_group_key_is_brand_and_category():
    assert split_group_key({"brand": "Levi's", "category": "Jeans"}) == ("Levi's", "Jeans")


def test_split_group_key_uses_empty_strings_for_missing_brand_and_category():
    assert split_group_key({"brand": None, "category": None}) == ("", "")


def test_grouped_split_puts_about_ten_percent_in_val():
    train, val = grouped_split(groups_of(group_count=100, examples_per_group=1))
    assert (len(train), len(val)) == (90, 10)


def test_grouped_split_keeps_every_group_whole_on_one_side():
    examples_by_group = groups_of(group_count=30, examples_per_group=3)
    train, val = grouped_split(examples_by_group)
    for group in examples_by_group.values():
        assert listing_ids(group) <= listing_ids(train) or listing_ids(group) <= listing_ids(val)


def test_grouped_split_puts_every_example_on_exactly_one_side():
    train, val = grouped_split(groups_of(group_count=30, examples_per_group=3))
    assert listing_ids(train) | listing_ids(val) == set(range(90))
    assert listing_ids(train) & listing_ids(val) == set()


def test_grouped_split_gives_the_same_split_for_the_same_input():
    examples_by_group = groups_of(group_count=30, examples_per_group=3)
    assert grouped_split(examples_by_group) == grouped_split(examples_by_group)


def test_write_jsonl_writes_one_json_object_per_line(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [{"listing_id": 1}, {"listing_id": 2}])
    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"listing_id": 1}, {"listing_id": 2}]


def test_write_jsonl_writes_non_ascii_characters_unescaped(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [{"text": "💎 Vintage 90’s Tee."}])
    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8") == '{"text": "💎 Vintage 90’s Tee."}\n'


def test_main_writes_every_listing_with_photos_to_train_or_val(tmp_path, monkeypatch):
    rows = [{**RUSSELL_HOODIE_ROW, "id": listing_id, "brand": f"Brand {listing_id}"} for listing_id in range(20)]
    for row in rows:
        row["thumbnail_url"] = f"https://fake-bucket/photos/{row['id']}_a.jpg"
        row["secondary_url"] = f"https://fake-bucket/photos/{row['id']}_b.jpg"
    monkeypatch.setattr(export, "fetch_listings_with_photos", lambda: rows)
    monkeypatch.setattr(export, "s3_connect", lambda: "fake s3 client")
    monkeypatch.setattr(
        export,
        "download_photo",
        lambda s3, s3_url, listing_id, position: tmp_path / "images" / f"{listing_id}_{position}.jpg",
    )
    monkeypatch.setattr(export, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(export, "LISTINGS_DIR", tmp_path / "listings")
    monkeypatch.setattr(export, "TRAIN_FILE", tmp_path / "listings" / "train.jsonl")
    monkeypatch.setattr(export, "VAL_FILE", tmp_path / "listings" / "val.jsonl")

    export.main()

    written_lines = (
        (tmp_path / "listings" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        + (tmp_path / "listings" / "val.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert sorted(json.loads(line)["listing_id"] for line in written_lines) == list(range(20))
