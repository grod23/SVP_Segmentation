from src.config import IMAGE_KEY, MASK_KEY, CLIP_LIMIT, TILE_GRID_SIZE
from torch.utils.data import Dataset
import torch
import cv2
import matplotlib.pyplot as plt
import sys

class FundusDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('Data Type')
        # print(type(self.data[index]))
        batch = self.data[index]
        if self.transform:
            batch = self.transform(batch)
        image = batch[IMAGE_KEY]
        mask = batch[MASK_KEY]
        # print(f'Image Shape: {image.shape}')
        # print(f'Image Type: {image.dtype}')
        # print("Image min/max:", image.min(), image.max())
        # print(f'Mask Shape: {mask.shape}')
        # print(f'Mask Type: {mask.dtype}')
        # print("Mask min/max:", mask.min(), mask.max())
        # print('\n')

        # Image Preprocessing
        # plt.figure(figsize=(10, 10))
        # plt.subplot(1, 2, 1)
        # plt.title('Image Before Preprocessing')
        # plt.axis('off')
        # plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy() / 255.0)
        # plt.subplot(1, 2, 2)
        # plt.title('Mask Before Preprocessing')
        # plt.axis('off')
        # plt.imshow(mask.permute(1, 2, 0).detach().cpu().numpy() / 255.0)
        # plt.show()

        # Extract green plane from image
        green_plane = image[1, :, :]
        # print(f'Green Shape: {green_plane.shape}')
        # print("Green min/max:", green_plane.min(), green_plane.max())
        # print(f'Green Type: {type(green_plane)}')

        # plt.figure(figsize=(10, 10))
        # plt.title('Green Plane')
        # plt.axis('off')
        # plt.imshow(green_plane.detach().cpu().numpy())
        # plt.show()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=CLIP_LIMIT,
            tileGridSize=TILE_GRID_SIZE
        )
        # Ensure image is rounded np.uint8 on CPU
        green_np = green_plane.cpu().numpy().round().astype('uint8')
        enhanced_image = clahe.apply(green_np)

        # plt.figure(figsize=(10, 10))
        # plt.title('CLAHE Image')
        # plt.axis('off')
        # plt.imshow(enhanced_image)
        # plt.show()

        # Mask Preprocessing:
        # Convert to single channel
        mask = mask.mean(dim=0)
        # Binarize to [0, 1]
        # Per-mask normalization (handles min > 0 safely)
        min_val = mask.min()
        max_val = mask.max()

        if max_val > min_val:
            mask = (mask - min_val) / (max_val - min_val)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros_like(mask, dtype=torch.float32)

        # Add channel dimension
        mask = mask.unsqueeze(0) # [Batch, Channel, Height, Width]

        # plt.figure(figsize=(10, 10))
        # plt.subplot(1, 2, 1)
        # plt.title('Image After Preprocessing')
        # plt.axis('off')
        # plt.imshow(enhanced_image)
        # plt.subplot(1, 2, 2)
        # plt.title('Mask After Preprocessing')
        # plt.axis('off')
        # plt.imshow(mask.detach().cpu().numpy().squeeze(0) * 255)
        # plt.show()

        # Normalize to [0, 1]
        enhanced_image = enhanced_image / 255.0
        # Convert to tensor
        enhanced_image = torch.tensor(enhanced_image, dtype=torch.float32).unsqueeze(0)

        # print(f'Enhanced After Shape: {enhanced_image.shape}')
        # print('Enhanced After min/max:', enhanced_image.min(), enhanced_image.max())
        # print(f'Enhanced After Type: {type(enhanced_image)}')
        # print(f'Mask After Shape: {mask.shape}')
        # print(f'Mask After Type: {mask.dtype}')
        # print("Mask After min/max:", mask.min(), mask.max())
        # print('\n')
        return enhanced_image, mask

        # https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1470941/full
