from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tag_analyzer import store
from tag_analyzer.schema import Neighbor
from tag_analyzer.store import (
    cosine_similarity_from_distance,
    encode_image_base64,
    library_size,
    recreate_collection,
    search_nearest_photos,
    stable_uuid_for,
    upload_library_photos,
)

FAKE_PHOTO_BYTES = b"tag photo"
FAKE_PHOTO_BASE64 = "dGFnIHBob3Rv"


class FakeBatch:
    def __init__(self):
        self.added_objects = []
        self.failed_objects = []
        self.batch_sizes = []

    @contextmanager
    def fixed_size(self, batch_size):
        self.batch_sizes.append(batch_size)
        yield self

    def add_object(self, properties, uuid):
        self.added_objects.append({"properties": properties, "uuid": uuid})


class FakeQuery:
    def __init__(self, found_objects):
        self.found_objects = found_objects
        self.near_image_arguments = None

    def near_image(self, **arguments):
        self.near_image_arguments = arguments
        return SimpleNamespace(objects=self.found_objects)


class FakeCollection:
    def __init__(self, found_objects=(), total_count=0):
        self.batch = FakeBatch()
        self.query = FakeQuery(list(found_objects))
        self.photo_count = total_count
        self.aggregate = SimpleNamespace(over_all=self.counted)

    def counted(self, total_count):
        return SimpleNamespace(total_count=self.photo_count)


class FakeCollections:
    def __init__(self, existing_collection=None):
        self.collection = existing_collection
        self.deleted_names = []
        self.created = []

    def exists(self, name):
        return self.collection is not None

    def delete(self, name):
        self.deleted_names.append(name)
        self.collection = None

    def create(self, name, vector_config, properties):
        self.created.append({"name": name, "property_names": [each.name for each in properties]})
        self.collection = FakeCollection()
        return self.collection

    def get(self, name):
        return self.collection


def fake_client(existing_collection=None) -> SimpleNamespace:
    return SimpleNamespace(collections=FakeCollections(existing_collection))


def found_object(filename: str, brand: str, era: str, distance: float) -> SimpleNamespace:
    return SimpleNamespace(
        properties={"filename": filename, "brand": brand, "era": era},
        metadata=SimpleNamespace(distance=distance),
    )


@pytest.fixture
def photo(tmp_path):
    photo_path = tmp_path / "90_1.jpg"
    photo_path.write_bytes(FAKE_PHOTO_BYTES)
    return photo_path


@pytest.fixture
def connect_arguments(monkeypatch):
    recorded = {}
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda **arguments: recorded.update(arguments))
    monkeypatch.setattr(store, "WEAVIATE_HOST", "fake.host")
    monkeypatch.setattr(store, "WEAVIATE_PORT", 18080)
    monkeypatch.setattr(store, "WEAVIATE_GRPC_PORT", 15051)
    monkeypatch.setattr(store, "WEAVIATE_API_KEY", None)
    return recorded


def test_connect_from_env_connects_to_the_configured_host_and_ports(connect_arguments):
    store.connect_from_env()
    assert (connect_arguments["host"], connect_arguments["port"], connect_arguments["grpc_port"]) == (
        "fake.host",
        18080,
        15051,
    )


def test_connect_from_env_sends_no_credentials_without_an_api_key(connect_arguments):
    store.connect_from_env()
    assert connect_arguments["auth_credentials"] is None


def test_connect_from_env_sends_the_api_key_when_one_is_set(connect_arguments, monkeypatch):
    monkeypatch.setattr(store, "WEAVIATE_API_KEY", "fake-key")
    store.connect_from_env()
    assert connect_arguments["auth_credentials"].api_key == "fake-key"


def test_recreate_collection_deletes_an_existing_collection_first():
    client = fake_client(existing_collection=FakeCollection())
    recreate_collection(client)
    assert client.collections.deleted_names == ["TagPhoto"]


def test_recreate_collection_deletes_nothing_when_the_collection_is_missing():
    client = fake_client()
    recreate_collection(client)
    assert client.collections.deleted_names == []


def test_recreate_collection_creates_tag_photo_with_label_and_image_properties():
    client = fake_client()
    recreate_collection(client)
    assert client.collections.created == [
        {"name": "TagPhoto", "property_names": ["filename", "brand", "era", "image"]}
    ]


def test_recreate_collection_returns_the_new_collection():
    client = fake_client()
    assert recreate_collection(client) is client.collections.collection


def test_stable_uuid_for_is_the_same_for_the_same_filename():
    assert stable_uuid_for("Russell/90_1.jpg") == stable_uuid_for("Russell/90_1.jpg")


def test_stable_uuid_for_differs_between_filenames():
    assert stable_uuid_for("Russell/90_1.jpg") != stable_uuid_for("Russell/90_2.jpg")


def test_encode_image_base64_is_the_photo_bytes_as_base64_text(photo):
    assert encode_image_base64(photo) == FAKE_PHOTO_BASE64


def test_upload_library_photos_uploads_each_entry_with_its_labels_and_encoded_photo(photo):
    client = fake_client()
    upload_library_photos(client, [{"filename": "90_1.jpg", "brand": "Russell", "era": "90s"}], photo.parent)
    assert [added["properties"] for added in client.collections.collection.batch.added_objects] == [
        {"filename": "90_1.jpg", "brand": "Russell", "era": "90s", "image": FAKE_PHOTO_BASE64}
    ]


def test_upload_library_photos_gives_each_photo_its_stable_uuid(photo):
    client = fake_client()
    upload_library_photos(client, [{"filename": "90_1.jpg", "brand": "Russell", "era": "90s"}], photo.parent)
    assert client.collections.collection.batch.added_objects[0]["uuid"] == stable_uuid_for("90_1.jpg")


def test_upload_library_photos_replaces_the_existing_collection(photo):
    client = fake_client(existing_collection=FakeCollection())
    upload_library_photos(client, [{"filename": "90_1.jpg", "brand": "Russell", "era": "90s"}], photo.parent)
    assert client.collections.deleted_names == ["TagPhoto"]


def test_upload_library_photos_raises_when_weaviate_reports_failed_objects(photo, monkeypatch):
    collection_with_failures = FakeCollection()
    collection_with_failures.batch.failed_objects = ["fake failure"]
    monkeypatch.setattr(store, "recreate_collection", lambda client: collection_with_failures)
    with pytest.raises(RuntimeError):
        upload_library_photos(
            fake_client(), [{"filename": "90_1.jpg", "brand": "Russell", "era": "90s"}], photo.parent
        )


def test_cosine_similarity_from_distance_is_one_minus_the_distance():
    assert cosine_similarity_from_distance(0.25) == 0.75


def test_search_nearest_photos_turns_each_found_object_into_a_neighbor(photo):
    client = fake_client(FakeCollection(found_objects=[found_object("Russell/90_1.jpg", "Russell", "90s", 0.25)]))
    assert search_nearest_photos(client, photo, 3) == [
        Neighbor(filename="Russell/90_1.jpg", brand="Russell", era="90s", similarity=0.75)
    ]


def test_search_nearest_photos_asks_for_k_photos_nearest_the_encoded_query_photo(photo):
    client = fake_client(FakeCollection())
    search_nearest_photos(client, photo, 3)
    near_image_arguments = client.collections.collection.query.near_image_arguments
    assert (near_image_arguments["near_image"], near_image_arguments["limit"]) == (FAKE_PHOTO_BASE64, 3)


def test_search_nearest_photos_is_empty_when_nothing_is_found(photo):
    assert search_nearest_photos(fake_client(FakeCollection()), photo, 3) == []


def test_library_size_is_zero_when_the_collection_is_missing():
    assert library_size(fake_client()) == 0


def test_library_size_is_the_total_count_of_the_collection():
    assert library_size(fake_client(FakeCollection(total_count=142))) == 142
