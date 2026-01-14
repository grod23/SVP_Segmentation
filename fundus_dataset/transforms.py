from src.config import IMAGE_SIZE, IMAGE_KEY, MASK_KEY
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureTyped, NormalizeIntensityd, Resized,
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
                # meta_keys='Metadata',  # Stores metadata in this key
                reader='PILReader',
                image_only=True,  # Image_only provides metadata
                ensure_channel_first=True  # Ensures correct channel format (Channels, Height, Width)
            ),
            # EnsureChannelFirstd(
            #     keys=['Image', 'Mask']  # Ensures correct channel format (Channels, Height, Width)
            # ),
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
            # ScaleIntensityRanged(
            #     keys=['Image'],
            #     a_min=0, a_max=255,
            #     b_min=0.0, b_max=1.0,
            #     clip=True
            # ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 4: DATA AUGMENTATION (TRAINING ONLY)
            # ─────────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=[IMAGE_KEY, MASK_KEY],
                dtype=torch.float32,
                track_meta=False,
                allow_missing_keys=False
            )
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
                # meta_keys='Metadata',  # Stores metadata in this key
                reader='PILReader',
                image_only=True,  # Image_only provides metadata
                ensure_channel_first=True  # Ensures correct channel format (Channels, Height, Width)
            ),
            # EnsureChannelFirstd(
            #     keys=['Image', 'Mask']  # Ensures correct channel format (Channels, Depth, Height, Width)
            # ),
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
            # ScaleIntensityRanged(
            #     keys=['Image'],
            #     a_min=0, a_max=255,
            #     b_min=0.0, b_max=1.0,
            #     clip=True
            # ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=[IMAGE_KEY, MASK_KEY],
                dtype=torch.float32,
                track_meta=False,
                allow_missing_keys=False
            )
        ])

        return test_transformations

