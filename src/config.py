from pathlib import Path

# ========================
# PROJECT PATHS
# ========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_IMAGES_DIR = DATASET_DIR / "train/images"
TRAIN_MASKS_DIR  = DATASET_DIR / "train/masks"

VAL_IMAGES_DIR = DATASET_DIR / "val/images"
VAL_MASKS_DIR  = DATASET_DIR / "val/masks"

TEST_IMAGES_DIR = DATASET_DIR / "test/images"