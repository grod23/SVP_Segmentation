from pathlib import Path
import torch

# ========================
# PROJECT PATHS
# ========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SPLIT = 'fundus_dataset/train_split.joblib'
METADATA = r'C:\Users\gabe7\Downloads\Fundus_Dataset\Labels\Temporal\Peak_and_trough\Metadata.json'

# ========================
# Hyperparameters
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 12
WEIGHT_DECAY = 0.01
IMAGE_SIZE = (224, 224)

