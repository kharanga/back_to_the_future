import pytest

from listings import images
from listings.images import download_photo, key_from_url, s3_connect


class FakeS3:
    def __init__(self):
        self.checked_buckets = []
        self.downloads = []

    def head_bucket(self, Bucket):
        self.checked_buckets.append(Bucket)

    def download_file(self, bucket, key, target):
        self.downloads.append((bucket, key))
        with open(target, "wb") as downloaded:
            downloaded.write(b"fake photo")


@pytest.fixture
def images_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(images, "S3_BUCKET", "fake-bucket")
    return tmp_path / "images"


def test_s3_connect_returns_an_s3_client_for_the_configured_region(monkeypatch):
    fake_s3 = FakeS3()
    client_arguments = []
    monkeypatch.setattr(images, "AWS_REGION", "fake-region-1")
    monkeypatch.setattr(
        images.boto3,
        "client",
        lambda service, region_name: client_arguments.append((service, region_name)) or fake_s3,
    )
    assert s3_connect() is fake_s3
    assert client_arguments == [("s3", "fake-region-1")]


def test_s3_connect_checks_that_the_bucket_is_reachable(monkeypatch):
    fake_s3 = FakeS3()
    monkeypatch.setattr(images, "S3_BUCKET", "fake-bucket")
    monkeypatch.setattr(images.boto3, "client", lambda service, region_name: fake_s3)
    s3_connect()
    assert fake_s3.checked_buckets == ["fake-bucket"]


def test_key_from_url_is_the_url_path_without_the_leading_slash():
    assert key_from_url("https://fake-bucket.s3.amazonaws.com/depop/123/photo.jpg") == "depop/123/photo.jpg"


def test_key_from_url_reads_an_s3_scheme_url_the_same_way():
    assert key_from_url("s3://fake-bucket/depop/123/photo.jpg") == "depop/123/photo.jpg"


def test_download_photo_names_the_file_by_listing_id_position_and_extension(images_dir):
    target = download_photo(FakeS3(), "https://fake-bucket.s3.amazonaws.com/depop/123/photo.jpg", 123, 0)
    assert target == images_dir / "123_0.jpg"


def test_download_photo_lowercases_the_extension(images_dir):
    target = download_photo(FakeS3(), "https://fake-bucket.s3.amazonaws.com/depop/123/photo.JPG", 123, 1)
    assert target == images_dir / "123_1.jpg"


def test_download_photo_downloads_the_key_from_the_configured_bucket(images_dir):
    fake_s3 = FakeS3()
    target = download_photo(fake_s3, "https://fake-bucket.s3.amazonaws.com/depop/123/photo.jpg", 123, 0)
    assert fake_s3.downloads == [("fake-bucket", "depop/123/photo.jpg")]
    assert target.read_bytes() == b"fake photo"


def test_download_photo_skips_a_file_that_is_already_there(images_dir):
    images_dir.mkdir()
    (images_dir / "123_0.jpg").write_bytes(b"downloaded earlier")
    fake_s3 = FakeS3()
    target = download_photo(fake_s3, "https://fake-bucket.s3.amazonaws.com/depop/123/photo.jpg", 123, 0)
    assert fake_s3.downloads == []
    assert target.read_bytes() == b"downloaded earlier"
