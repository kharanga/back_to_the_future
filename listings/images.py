from pathlib import Path
from urllib.parse import urlparse

import boto3

from listings import IMAGES_DIR
from listings.config import AWS_REGION, S3_BUCKET


def s3_connect():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.head_bucket(Bucket=S3_BUCKET)
    return s3


def key_from_url(s3_url: str) -> str:
    return urlparse(s3_url).path.lstrip("/")


def download_photo(s3, s3_url: str, listing_id: int, position: int) -> Path:
    key = key_from_url(s3_url)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target = IMAGES_DIR / f"{listing_id}_{position}{Path(key).suffix.lower()}"
    if not target.exists():
        s3.download_file(S3_BUCKET, key, str(target))
    return target
