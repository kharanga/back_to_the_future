from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LISTINGS_DIR = PROJECT_ROOT / "data" / "listings"
IMAGES_DIR = LISTINGS_DIR / "images"
TRAIN_FILE = LISTINGS_DIR / "train.jsonl"
VAL_FILE = LISTINGS_DIR / "val.jsonl"
EVAL_FILE = LISTINGS_DIR / "eval.jsonl"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "listing_first_line"
