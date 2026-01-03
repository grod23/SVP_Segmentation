from pathlib import Path
import torch

# ========================
# PROJECT PATHS
# ========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = r"C:\Users\gabe7\Downloads\Fundus_Dataset\Labels\Temporal\Peak_and_trough\images"
MASK_DIR = r"C:\Users\gabe7\Downloads\Fundus_Dataset\Labels\Temporal\Peak_and_trough\masks"
METADATA = r"C:\Users\gabe7\Downloads\Fundus_Dataset\Labels\Temporal\Peak_and_trough\Metadata.json"

# ========================
# Hyperparameters
# ========================
print(f'Device Available: {torch.cuda.is_available()}')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 12
