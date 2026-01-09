import torch.cuda
from fundus_dataset import TRAIN_SPLIT, METADATA, BATCH_SIZE, NUM_WORKERS
from monai.data import DataLoader, PersistentDataset
import json
import joblib

class DataUtils:
    def __init__(self):
        self.train_split = TRAIN_SPLIT

    def load_split(self):
        train, validation, test = joblib.load(self.train_split)
        return train, validation, test

    def create_datasets(self):
        train, validation, test = self.load_split()
        train_dataset = PersistentDataset(
        )
        validation_dataset = PersistentDataset(
        )
        test_dataset = PersistentDataset(
        )
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
            persistent_workers=True
        )
        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=None,
            persistent_workers=True
        )
        test_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            collate_fn=None,
            persistent_workers=True
        )
        return train_loader, validation_loader, test_loader

