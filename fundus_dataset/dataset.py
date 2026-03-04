from src.config import IMAGE_KEY, MASK_KEY, METADATA_KEY, CLIP_LIMIT, TILE_GRID_SIZE
from torch.utils.data import Dataset
import torch
import cv2
import matplotlib.pyplot as plt
import sys

class FundusDataset(Dataset):
    def __init__(self, data_dict, transform=None, testing=False):
        self.data = data_dict
        self.transform = transform
        self.testing = testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        batch = self.data[index]
        if self.transform:
            batch = self.transform(batch)

        images = batch[IMAGE_KEY]
        masks = batch[MASK_KEY]
        metadata = batch[METADATA_KEY]

        def preprocess_image(image):
            green_plane = image[1, :, :]
            clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
            green_np = green_plane.cpu().numpy().round().astype('uint8')
            enhanced = clahe.apply(green_np)
            enhanced = enhanced / 255.0
            enhanced = torch.tensor(enhanced, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
            return enhanced

        def preprocess_mask(mask):
            mask = mask.mean(dim=0)
            min_val, max_val = mask.min(), mask.max()
            if max_val > min_val:
                mask = (mask - min_val) / (max_val - min_val)
                mask = (mask > 0.5).float()
            else:
                mask = torch.zeros_like(mask, dtype=torch.float32)
            return mask.unsqueeze(0)  # [1, H, W]

        # Split [6, H, W] -> two [3, H, W]
        image_min, image_max = images[:3], images[3:]
        # Split [6, H, W] -> two [3, H, W]  (adjust to [:1], [1:] if masks are single channel)
        mask_min, mask_max = masks[:3], masks[3:]

        # Preprocess
        enhanced_min = preprocess_image(image_min)
        enhanced_max = preprocess_image(image_max)
        processed_mask_min = preprocess_mask(mask_min)
        processed_mask_max = preprocess_mask(mask_max)

        # Stack into [2, 1, H, W]
        enhanced_images = torch.stack([enhanced_min, enhanced_max])
        processed_masks = torch.stack([processed_mask_min, processed_mask_max])

        if self.testing:
            # fig, axes = plt.subplots(3, 2, figsize=(12, 15))
            # for i, (orig, enhanced, pmask, label) in enumerate(zip(
            #         [image_min, image_max],
            #         [enhanced_min, enhanced_max],
            #         [processed_mask_min, processed_mask_max],
            #         ['Min', 'Max']
            # )):
            #     axes[0, i].set_title(f'Original {label}')
            #     axes[0, i].axis('off')
            #     axes[0, i].imshow(orig.permute(1, 2, 0).detach().cpu().numpy() / 255)
            #
            #     axes[1, i].set_title(f'Preprocessed {label}')
            #     axes[1, i].axis('off')
            #     axes[1, i].imshow(enhanced.squeeze(0).detach().cpu().numpy())
            #
            #     axes[2, i].set_title(f'Ground-Truth Mask {label}')
            #     axes[2, i].axis('off')
            #     axes[2, i].imshow(pmask.squeeze(0).detach().cpu().numpy())
            #
            # plt.tight_layout()
            # plt.show()
            #
            # for i, label in enumerate(['Min', 'Max']):
            #     print(f'--- {label} ---')
            #     print(f'Enhanced Shape: {enhanced_images[i].shape} | dtype: {enhanced_images[i].dtype}')
            #     print(f'Enhanced min/max: {enhanced_images[i].min():.3f} / {enhanced_images[i].max():.3f}')
            #     print(f'Mask Shape: {processed_masks[i].shape} | dtype: {processed_masks[i].dtype}')
            #     print(f'Mask min/max: {processed_masks[i].min():.3f} / {processed_masks[i].max():.3f}')
            #     print(f'Metadata: {metadata[i]}\n')

            return enhanced_images, processed_masks, torch.stack([image_min, image_max]), metadata

        return enhanced_images, processed_masks
        # enhanced_images: [2, 1, H, W]  — index 0 = min, index 1 = max
        # processed_masks: [2, 1, H, W]  — index 0 = min, index 1 = max

    # def __getitem__(self, index):
    #     # print('Data Type')
    #     # print(type(self.data[index]))
    #     batch = self.data[index]
    #     if self.transform:
    #         batch = self.transform(batch)
    #     image = batch[IMAGE_KEY]
    #     mask = batch[MASK_KEY]
    #     metadata = batch[METADATA_KEY]
    #
    #     # plt.figure(figsize=(10, 10))
    #     # plt.title('Original Image')
    #     # plt.axis('off')
    #     # plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy() / 255)
    #     # plt.show()
    #     print(f'Image Shape: {image.shape}')
    #     print(f'Image Type: {image.dtype}')
    #     print("Image min/max:", image.min(), image.max())
    #     print(f'Mask Shape: {mask.shape}')
    #     print(f'Mask Type: {mask.dtype}')
    #     print("Mask min/max:", mask.min(), mask.max())
    #     print('\n')
    #
    #     # Extract green plane from image
    #     green_plane = image[1, :, :]
    #     # print(f'Green Shape: {green_plane.shape}')
    #     # print("Green min/max:", green_plane.min(), green_plane.max())
    #     # print(f'Green Type: {type(green_plane)}')
    #
    #     # plt.figure(figsize=(10, 10))
    #     # plt.title('Image Green Plane Only')
    #     # plt.axis('off')
    #     # plt.imshow(green_plane.detach().cpu().numpy())
    #     # plt.show()
    #
    #     # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #     clahe = cv2.createCLAHE(
    #         clipLimit=CLIP_LIMIT,
    #         tileGridSize=TILE_GRID_SIZE
    #     )
    #     # Ensure image is rounded np.uint8 on CPU
    #     green_np = green_plane.cpu().numpy().round().astype('uint8')
    #     enhanced_image = clahe.apply(green_np)
    #
    #     # plt.figure(figsize=(10, 10))
    #     # plt.title('Image After CLAHE')
    #     # plt.axis('off')
    #     # plt.imshow(enhanced_image)
    #     # plt.show()
    #
    #     # Mask Preprocessing:
    #     # Convert to single channel
    #     mask = mask.mean(dim=0)
    #     # Binarize to [0, 1]
    #     # Per-mask normalization (handles min > 0 safely)
    #     min_val = mask.min()
    #     max_val = mask.max()
    #
    #     if max_val > min_val:
    #         mask = (mask - min_val) / (max_val - min_val)
    #         mask = (mask > 0.5).float()
    #     else:
    #         mask = torch.zeros_like(mask, dtype=torch.float32)
    #
    #     # Add channel dimension
    #     mask = mask.unsqueeze(0) # [Batch, Channel, Height, Width]
    #
    #     # plt.figure(figsize=(10, 10))
    #     # plt.subplot(1, 2, 1)
    #     # plt.title('Image After Preprocessing')
    #     # plt.axis('off')
    #     # plt.imshow(enhanced_image)
    #     # plt.subplot(1, 2, 2)
    #     # plt.title('Mask After Preprocessing')
    #     # plt.axis('off')
    #     # plt.imshow(mask.detach().cpu().numpy().squeeze(0) * 255)
    #     # plt.show()
    #
    #     # Normalize to [0, 1]
    #     enhanced_image = enhanced_image / 255.0
    #     # Convert to tensor
    #     enhanced_image = torch.tensor(enhanced_image, dtype=torch.float32).unsqueeze(0)
    #
    #     # print(f'Enhanced After Shape: {enhanced_image.shape}')
    #     # print('Enhanced After min/max:', enhanced_image.min(), enhanced_image.max())
    #     # print(f'Enhanced After Type: {type(enhanced_image)}')
    #     # print(f'Mask After Shape: {mask.shape}')
    #     # print(f'Mask After Type: {mask.dtype}')
    #     # print("Mask After min/max:", mask.min(), mask.max())
    #     # print('\n')
    #     if self.testing:
    #         return enhanced_image, mask, image, metadata
    #     return enhanced_image, mask, image
    #
    #     # https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1470941/full
