from fundus_dataset import IMAGE_DIR, MASK_DIR, METADATA
from monai.data import DataLoader, PersistentDataset
import json

class DataUtils:
    def __init__(self):
        self.metadata = self.load_data()

    def load_data(self):
        # Load JSON into a Python dict
        with open(METADATA, 'r') as f:
            metadata = json.load(f)
        return metadata

    def create_datasets(self):
        return

    def create_dataloaders(self):
        return

