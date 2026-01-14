from src.config import IMAGE_KEY, MASK_KEY, TENSORBOARD_DIR
from monai.visualize import plot_2d_or_3d_image, img2tensorboard
from monai.visualize.utils import blend_images
from monai.handlers import TensorBoardHandler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import shutil
import sys

class Logger:
    def __init__(self):
        self.writer = SummaryWriter(TENSORBOARD_DIR)

    def clear_tensorboard(self):
        if os.path.exists(TENSORBOARD_DIR):
            print('Clearing Tensorboard Cache...')
            shutil.rmtree(TENSORBOARD_DIR)

    def visualize_batch(self, batch):
        image_batch = batch[IMAGE_KEY]
        mask_batch = batch[MASK_KEY]
        print(f'Image Shape: {image_batch.shape}')
        print(f'Image Dtype: {image_batch.dtype}')
        print(f'Mask Shape: {mask_batch.shape}')
        print(f'Mask Dtype: {mask_batch.dtype}')

        # Get first items in batch
        image = image_batch[0]
        mask = mask_batch[0]

        # stretch contrast for display
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        print("Image min/max:", image.min(), image.max())
        print("Mask min/max:", mask.min(), mask.max())

        # Convert mask to grayscale
        mask = mask.mean(dim=0, keepdim=True)  # [1, H, W]
        # Overlay Image
        blended = blend_images(
            image=image,
            label=mask,
            alpha=0.8
        )

        # Convert each image from [C, H, W] to [H, W, C] and to numpy
        mask = mask.permute(1, 2, 0)
        mask = mask.detach().cpu().numpy()
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        blended = blended.permute(1, 2, 0)
        blended = blended.detach().cpu().numpy()

        # Plot image, mask, and overlay
        # plot_2d_or_3d_image(
        #     data=[image, mask, blended],
        #     step=0,
        #     writer=self.writer
        # )

        # Plot Image, Mask, Overlay
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(mask.squeeze(-1), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Blended")
        plt.imshow(blended)
        plt.axis("off")
        plt.show()
