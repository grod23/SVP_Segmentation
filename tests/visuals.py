from src.config import IMAGE_KEY, MASK_KEY
import matplotlib.pyplot as plt
from monai.visualize.utils import blend_images

class Visualizer:
    def __init__(self, logger):
        self.logger = logger


    def display_training_loss(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.logger.training_loss_logs, c='b', label='Training Loss')
        plt.plot(self.logger.validation_loss_logs, c='r', label='Validation Loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.show()

    def prep_sample(self, sample):
        sample = sample.permute(1, 2, 0)
        sample = sample.detach().cpu().numpy()
        return sample

    def plot_single_sample(self, sample):
        # print(f'Sample Shape: {sample.shape}')
        sample_prepped = self.prep_sample(sample)
        plt.figure(figsize=(10, 10))
        plt.title("Image")
        plt.imshow(sample_prepped)
        plt.axis("off")
        plt.show()

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
            alpha=0.4
        )

        # Convert each image from [C, H, W] to [H, W, C] and to numpy
        mask = self.prep_sample(mask)
        image = self.prep_sample(image)
        blended = self.prep_sample(blended)

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

    def plot_image_results(self, original_image, mask, predicted_mask):
        # Prep each image
        mask = self.prep_sample(mask)
        predicted_mask = self.prep_sample(predicted_mask
                                          )
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image.permute(1, 2, 0).detach().cpu().numpy() / 255)
        plt.title('Original Image')
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.title('Ground Truth')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask)
        plt.title('Predicted Mask')
        plt.axis("off")
        plt.show()


