from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIBRARY_DIR = PROJECT_ROOT / "data" / "tags"
QUERIES_DIR = PROJECT_ROOT / "data" / "test_tags"
INDEX_DIR = PROJECT_ROOT / "index"
RESULTS_DIR = PROJECT_ROOT / "results"
LABELS_MANIFEST = PROJECT_ROOT/ "data"/ "labels.json"

VECTORS_FILE = INDEX_DIR / "vectors.npz"
LABELS_FILE = INDEX_DIR / "labels.json"
TEST_LABELS_FILE = PROJECT_ROOT / "data" / "test_labels.json"