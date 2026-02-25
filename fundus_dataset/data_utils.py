import torch
from src.config import METADATA, TRAIN_SPLIT, BATCH_SIZE, NUM_WORKERS, CACHE_DIR
from fundus_dataset import Transform, FundusDataset
from monai.data import DataLoader, PersistentDataset, Dataset
import json
import joblib
import shutil
import os


"""Utility class for loading datasets and creating PyTorch DataLoaders."""
class DataUtils:
    def __init__(self):
        self.train_split = TRAIN_SPLIT
        self.metadata = METADATA
        self.transform = Transform()

    def load_metadata(self):
        # Load JSON into a Python dict
        with open(self.metadata, 'r') as f:
            metadata = json.load(f)
        return metadata

    def clear_cache(self):
        # Reset cache directory
        if os.path.exists(CACHE_DIR):
            print('Clearing Cache Directory')
            shutil.rmtree(CACHE_DIR)

    def load_split(self):
        train, validation, test = joblib.load(self.train_split)
        return train, validation, test

    def create_datasets(self):
        train, validation, test = self.load_split()
        # Load Datasets
        train_dataset = FundusDataset(train, transform=self.transform.train_transform)
        validation_dataset = FundusDataset(validation, transform=self.transform.test_transform)
        test_dataset = FundusDataset(test, transform=self.transform.test_transform)
        return train_dataset, validation_dataset, test_dataset

    def create_dataloaders(self):
        train_dataset, validation_dataset, test_dataset = self.create_datasets()
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=None,
            persistent_workers=NUM_WORKERS > 0
        )
        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=None,
            persistent_workers=NUM_WORKERS > 0
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=None,
            persistent_workers=NUM_WORKERS > 0
        )
        return train_loader, validation_loader, test_loader

