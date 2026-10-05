import base64
import os
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.util import generate_uuid5

from tag_analyzer import PROJECT_ROOT
from tag_analyzer.schema import Neighbor

load_dotenv(PROJECT_ROOT / ".env")

COLLECTION = "TagPhoto"
CLIP_MODEL = "clip-ViT-B-32-multilingual-v1"
IMAGE_PROPERTY = "image"
LABEL_PROPERTIES = ["filename", "brand", "era"]
UPLOAD_BATCH_SIZE = 20
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.environ.get("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", 50051))
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")


def connect_from_env() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None,
    )


def recreate_collection(client: weaviate.WeaviateClient):
    if client.collections.exists(COLLECTION):
        client.collections.delete(COLLECTION)
    return client.collections.create(
        COLLECTION,
        vector_config=Configure.Vectors.multi2vec_clip(image_fields=[IMAGE_PROPERTY]),
        properties=[
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="brand", data_type=DataType.TEXT),
            Property(name="era", data_type=DataType.TEXT),
            Property(name=IMAGE_PROPERTY, data_type=DataType.BLOB),
        ],
    )


def stable_uuid_for(filename: str) -> str:
    return generate_uuid5(filename)


def encode_image_base64(photo_path: Path) -> str:
    return base64.b64encode(photo_path.read_bytes()).decode("ascii")


def upload_library_photos(client: weaviate.WeaviateClient, entries: list[dict], library_dir: Path):
    collection = recreate_collection(client)
    with collection.batch.fixed_size(batch_size=UPLOAD_BATCH_SIZE) as batch:
        for entry in entries:
            batch.add_object(
                properties={
                    "filename": entry["filename"],
                    "brand": entry["brand"],
                    "era": entry["era"],
                    IMAGE_PROPERTY: encode_image_base64(library_dir / entry["filename"]),
                },
                uuid=stable_uuid_for(entry["filename"]),
            )
            print(f"uploaded {entry['filename']}")
    if collection.batch.failed_objects:
        raise RuntimeError(f"weaviate upload errors: {collection.batch.failed_objects}")


def cosine_similarity_from_distance(distance: float) -> float:
    return 1.0 - distance


def search_nearest_photos(client: weaviate.WeaviateClient, query_photo: Path, k: int) -> list[Neighbor]:
    collection = client.collections.get(COLLECTION)
    response = collection.query.near_image(
        near_image=encode_image_base64(Path(query_photo)),
        limit=k,
        return_properties=LABEL_PROPERTIES,
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        Neighbor(
            filename=obj.properties["filename"],
            brand=obj.properties["brand"],
            era=obj.properties["era"],
            similarity=cosine_similarity_from_distance(obj.metadata.distance),
        )
        for obj in response.objects
    ]


def library_size(client: weaviate.WeaviateClient) -> int:
    if not client.collections.exists(COLLECTION):
        return 0
    return client.collections.get(COLLECTION).aggregate.over_all(total_count=True).total_count
