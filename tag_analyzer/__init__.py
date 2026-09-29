from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIBRARY_DIR = PROJECT_ROOT / "data" / "tags"
QUERIES_DIR = PROJECT_ROOT / "data" / "test_tags"
LABELS_MANIFEST = PROJECT_ROOT / "data" / "labels.json"
TEST_LABELS_FILE = PROJECT_ROOT / "data" / "test_labels.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
