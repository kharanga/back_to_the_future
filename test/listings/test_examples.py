from PIL import Image

from listings import examples
from listings.examples import facts, open_photo, photo_paths, read_jsonl, target_line, user_blocks
from listings.parse import prompt_text

RUSSELL_HOODIE_FACTS = {"brand": "Russell", "era": "90s", "category": "Hoodies", "gender": "Men", "color": "Grey"}
THUMBNAIL_BLOCK = {"type": "image", "image": "data/listings/images/7_0.jpg"}
SECONDARY_BLOCK = {"type": "image", "image": "data/listings/images/7_1.jpg"}
PROMPT_BLOCK = {"type": "text", "text": prompt_text(RUSSELL_HOODIE_FACTS)}
RUSSELL_HOODIE_EXAMPLE = {
    "listing_id": 7,
    "messages": [
        {"role": "user", "content": [THUMBNAIL_BLOCK, SECONDARY_BLOCK, PROMPT_BLOCK]},
        {"role": "assistant", "content": [{"type": "text", "text": "💎 Vintage 90's Boxy Russell Hoodie."}]},
    ],
}


def test_read_jsonl_reads_one_dict_per_line(tmp_path):
    (tmp_path / "val.jsonl").write_text('{"listing_id": 1}\n{"listing_id": 2}\n', encoding="utf-8")
    assert read_jsonl(tmp_path / "val.jsonl") == [{"listing_id": 1}, {"listing_id": 2}]


def test_read_jsonl_reads_non_ascii_characters(tmp_path):
    (tmp_path / "val.jsonl").write_text('{"text": "💎 Vintage 90’s Tee."}\n', encoding="utf-8")
    assert read_jsonl(tmp_path / "val.jsonl") == [{"text": "💎 Vintage 90’s Tee."}]


def test_open_photo_opens_a_path_relative_to_the_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(examples, "PROJECT_ROOT", tmp_path)
    Image.new("RGB", (4, 2)).save(tmp_path / "7_0.png")
    assert open_photo("7_0.png").size == (4, 2)


def test_open_photo_converts_the_photo_to_rgb(tmp_path, monkeypatch):
    monkeypatch.setattr(examples, "PROJECT_ROOT", tmp_path)
    Image.new("L", (4, 2)).save(tmp_path / "7_0.png")
    assert open_photo("7_0.png").mode == "RGB"


def test_user_blocks_is_the_content_of_the_user_message():
    assert user_blocks(RUSSELL_HOODIE_EXAMPLE) == [THUMBNAIL_BLOCK, SECONDARY_BLOCK, PROMPT_BLOCK]


def test_photo_paths_is_the_thumbnail_path_then_the_secondary_path():
    assert photo_paths(RUSSELL_HOODIE_EXAMPLE) == ["data/listings/images/7_0.jpg", "data/listings/images/7_1.jpg"]


def test_facts_reads_the_facts_back_out_of_the_prompt_block():
    assert facts(RUSSELL_HOODIE_EXAMPLE) == RUSSELL_HOODIE_FACTS


def test_target_line_is_the_text_of_the_assistant_message():
    assert target_line(RUSSELL_HOODIE_EXAMPLE) == "💎 Vintage 90's Boxy Russell Hoodie."
