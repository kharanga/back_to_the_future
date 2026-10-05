import json
from contextlib import nullcontext

import pytest

from tag_analyzer import build_index, store

RUSSELL_ENTRY = {"filename": "Russell/90_1.jpg", "brand": "Russell", "era": "90s"}


class SimpleLibrary:
    def __init__(self, root, uploads):
        self.tags_dir = root / "tags"
        self.labels_file = root / "labels.json"
        self.uploads = uploads

    def add_photo(self, relative_name: str):
        (self.tags_dir / relative_name).parent.mkdir(parents=True, exist_ok=True)
        (self.tags_dir / relative_name).write_bytes(b"fake photo")

    def write_manifest(self, entries: list[dict]):
        self.labels_file.write_text(json.dumps(entries))


@pytest.fixture
def library(tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setattr(build_index, "LIBRARY_DIR", tmp_path / "tags")
    monkeypatch.setattr(build_index, "LABELS_MANIFEST", tmp_path / "labels.json")
    monkeypatch.setattr(store, "connect_from_env", lambda: nullcontext("fake client"))
    monkeypatch.setattr(
        store,
        "upload_library_photos",
        lambda client, entries, library_dir: uploads.append((entries, library_dir)),
    )
    monkeypatch.setattr(store, "library_size", lambda client: len(uploads[0][0]))
    (tmp_path / "tags").mkdir()
    return SimpleLibrary(tmp_path, uploads)


def test_main_uploads_the_manifest_entries_from_the_library_folder(library):
    library.add_photo("Russell/90_1.jpg")
    library.write_manifest([RUSSELL_ENTRY])
    build_index.main()
    assert library.uploads == [([RUSSELL_ENTRY], library.tags_dir)]


def test_main_raises_when_a_manifest_entry_has_no_photo_on_disk(library):
    library.write_manifest([RUSSELL_ENTRY])
    with pytest.raises(FileNotFoundError):
        build_index.main()
    assert library.uploads == []


def test_main_warns_about_photos_missing_from_the_manifest(library, capsys):
    library.add_photo("Russell/90_1.jpg")
    library.add_photo("Russell/80_1.jpg")
    library.write_manifest([RUSSELL_ENTRY])
    build_index.main()
    assert "WARNING: photos not in manifest (skipped): ['Russell/80_1.jpg']" in capsys.readouterr().out
