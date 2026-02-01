from src.config import IMAGE_SIZE, IMAGE_KEY, MASK_KEY
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureTyped, NormalizeIntensityd, Resized, ToTensord,
    DeleteItemsd, RandRotate90d,  RandFlipd, ScaleIntensityRanged, RandShiftIntensityd, RandGaussianNoised,
    RandGaussianSmoothd, RandScaleIntensityd)

"""Transformation class for loading MONAI transform compositions to PyTorch datasets"""
class Transform:
    def __init__(self):
        self.train_transform = self.load_train_transforms()
        self.test_transform = self.load_test_transforms()

    def load_train_transforms(self):
        train_transformations = Compose([
            # ─────────────────────────────────────────────────────────────
            # STAGE 1: LOADING & BASIC PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            LoadImaged(
                keys=[IMAGE_KEY, MASK_KEY],
                dtype=torch.float32,
                # dtype=np.uint8,
                reader='PILReader',
                image_only=True,  # Image_only provides metadata
                ensure_channel_first=True  # Ensures correct channel format (Channels, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            Resized(
                keys=[IMAGE_KEY, MASK_KEY],
                spatial_size=IMAGE_SIZE,
                mode=['bilinear', 'nearest']
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: DATA AUGMENTATION (TRAINING ONLY)
            # ─────────────────────────────────────────────────────────────
            # RandRotate90d(
            #     keys=[IMAGE_KEY, MASK_KEY],
            #     prob=0.5
            # )
        ])

        return train_transformations

    def load_test_transforms(self):
        # Test and validation transformations
        test_transformations = Compose([
            # ─────────────────────────────────────────────────────────────
            # STAGE 1: LOADING & BASIC PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            LoadImaged(
                keys=[IMAGE_KEY, MASK_KEY],
                dtype=torch.float32,
                reader='PILReader',
                image_only=True,  # Image_only provides metadata
                ensure_channel_first=True  # Ensures correct channel format (Channels, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            Resized(
                keys=[IMAGE_KEY, MASK_KEY],
                spatial_size=IMAGE_SIZE,
                mode=['bilinear', 'nearest']
            ),
        ])

        return test_transformations

