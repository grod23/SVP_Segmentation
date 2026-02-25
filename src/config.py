from pathlib import Path
import torch


# ========================
# SEEDING FOR REPRODUCIBILITY
# ========================
SEED = 67

# ========================
# PROJECT PATHS
# ========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SPLIT = PROJECT_ROOT / 'fundus_dataset/train_split.joblib'
METADATA = r'C:/Users/gabe7/Downloads/Fundus_Dataset/Labels/Temporal/Peak_and_trough/Metadata.json'
CACHE_DIR = 'cache'
TENSORBOARD_DIR = './runs'

# ========================
# MONAI Dictionary Keys
# ========================
IMAGE_KEY = 'Image'
MASK_KEY = 'Mask'
METADATA_KEY = 'Metadata'
# ========================
# Metadata Dictionary Keys
# ========================
ViDEO_TITLE_KEY = 'Video Title'
FRAME_KEY = 'Frame'
MIN_MAX_KEY = 'Min/Max'
DISEASE_KEY = 'Disease'
SVP_CLASS_KEY = 'SVP'

# ========================
# Hyperparameters
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0
EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 12
WEIGHT_DECAY = 0.01
IMAGE_SIZE = (512, 512)

# ========================
# Preprocessing Hyperparameters
# ========================
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)

# ========================
# Model Parameters
# ========================
IN_CHANNELS = 1
OUT_CHANNELS = 1
SPATIAL_DIMS = 2


