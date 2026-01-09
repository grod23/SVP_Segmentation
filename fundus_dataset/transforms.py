import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, EnsureTyped, NormalizeIntensityd, Resized,
    DeleteItemsd, RandRotate90d,  RandFlipd, ScaleIntensityRanged, RandShiftIntensityd, RandGaussianNoised,
    RandGaussianSmoothd, RandScaleIntensityd)
from fundus_dataset import DEVICE, IMAGE_SIZE

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
                keys=['Image', 'Mask'],
                dtype=torch.float32,
                meta_keys='Metadata',
                reader='ITKReader',
                image_only=False,  # Image_only provides metadata for spacing info
                ensure_channel_first=True
            ),
            EnsureChannelFirstd(
                keys=['Image', 'Mask']  # Ensures correct channel format (Channels, Depth, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            Resized(
                keys=['Image', 'Mask'],
                spatial_size=IMAGE_SIZE
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=['Image'],
                nonzero=True,
                channel_wise=False
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 4: DATA AUGMENTATION (TRAINING ONLY)
            # ─────────────────────────────────────────────────────────────
            # Spatial augmentations
            # RandRotate90d(
            #     keys=["Folder Path"],
            #     prob=0.3,
            #     spatial_axes=(0, 1)  # Only rotate in axial plane
            # ),
            # RandFlipd(
            #     keys=["Folder Path"],
            #     prob=0.3,
            #     spatial_axis=0  # Left-right flip
            # ),
            # # Intensity augmentations (helps with scanner variability)
            # RandScaleIntensityd(
            #     keys=["Folder Path"],
            #     factors=0.2,  # ±20% intensity scaling
            #     prob=0.2
            # ),
            # RandShiftIntensityd(
            #     keys=["Folder Path"],
            #     offsets=0.1,  # Small intensity shifts
            #     prob=0.2
            # ),
            # RandGaussianNoised(
            #     keys=["Folder Path"],
            #     prob=0.2,
            #     mean=0.0,
            #     std=0.05  # Small random noise
            # ),
            # RandGaussianSmoothd(  # Random smoothing
            #     keys=["Folder Path"],
            #     prob=0.2,
            #     sigma_x=(0.5, 1.0),
            #     sigma_y=(0.5, 1.0),
            #     sigma_z=(0.5, 1.0)
            # ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=['Image', 'Mask'],
                dtype=torch.float32,
                device=DEVICE,
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
                keys=['Image', 'Mask'],
                dtype=torch.float32,
                meta_keys='Metadata',
                reader='ITKReader',
                image_only=False,  # Image_only provides metadata for spacing info
                ensure_channel_first=True
            ),
            EnsureChannelFirstd(
                keys=['Image', 'Mask']  # Ensures correct channel format (Channels, Depth, Height, Width)
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 2: SPATIAL PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            Resized(
                keys=['Image', 'Mask'],
                spatial_size=IMAGE_SIZE
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 3: INTENSITY PREPROCESSING
            # ─────────────────────────────────────────────────────────────
            NormalizeIntensityd(  # Z-Score Normalization (data - mean) / std_dev
                keys=['Image'],
                nonzero=True,
                channel_wise=False
            ),
            # ─────────────────────────────────────────────────────────────
            # STAGE 5: Tensor Conversion
            # ─────────────────────────────────────────────────────────────
            EnsureTyped(
                keys=['Image', 'Mask'],
                dtype=torch.float32,
                device=DEVICE,
                track_meta=False,
                allow_missing_keys=False
            )
        ])

        return test_transformations

