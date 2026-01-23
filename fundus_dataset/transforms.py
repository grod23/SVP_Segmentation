from src.config import IMAGE_SIZE, IMAGE_KEY, MASK_KEY
import torch
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
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            # NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
            #     keys=['Image'],
            #     nonzero=True,
            #     channel_wise=False
            # ),
            ScaleIntensityRanged(
                keys=[IMAGE_KEY, MASK_KEY],
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 4: DATA AUGMENTATION (TRAINING ONLY)
            # ─────────────────────────────────────────────────────────────
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
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            # NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
            #     keys=['Image'],
            #     nonzero=True,
            #     channel_wise=False
            # ),
            ScaleIntensityRanged(
                keys=['Mask'],
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
        ])

        return test_transformations

