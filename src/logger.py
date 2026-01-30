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
    def __init__(self, training_loader_length, validation_loader_length):
        self.train_length = training_loader_length
        self.validation_length = validation_loader_length
        self.writer = SummaryWriter(TENSORBOARD_DIR)
        # Total Loss logs
        self.training_loss_logs = []
        self.validation_loss_logs = []
        # Epoch Loss Logs
        self.train_epoch_loss = 0
        self.validation_epoch_loss = 0
        # Accuracy Logs
        self.train_accuracy = []
        self.validation_accuracy = []
        self.current_epoch = 1

    def log_epoch_loss(self, epoch_loss_fn, train=True):
        epoch_loss = epoch_loss_fn.item()
        if train:
            # print(f'Logging Epoch {self.current_epoch} Train Loss: {epoch_loss}')
            self.train_epoch_loss += epoch_loss
        else:
            # print(f'Logging Epoch {self.current_epoch} Validation Loss: {epoch_loss}')
            self.validation_epoch_loss += epoch_loss

    def clear_epoch_loss(self):
        print('New Epoch: Resetting Epoch Loss')
        self.train_epoch_loss = 0
        self.validation_epoch_loss = 0
        self.current_epoch += 1


    def get_average_loss(self):
        avg_train_loss = self.train_epoch_loss / self.train_length
        avg_validation_loss = self.validation_epoch_loss / self.validation_length
        # Log Average Loss
        self.training_loss_logs.append(avg_train_loss)
        self.validation_loss_logs.append(avg_validation_loss)
        # Clear epoch loss for next epoch
        self.clear_epoch_loss()
        return avg_train_loss, avg_validation_loss


    def display_loss(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.training_loss_logs, c='b', label='Training Loss')
        plt.plot(self.validation_loss_logs, c='r', label='Validation Loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.show()


    def clear_tensorboard(self):
        if os.path.exists(TENSORBOARD_DIR):
            print('Clearing Tensorboard Cache...')
            shutil.rmtree(TENSORBOARD_DIR)

    def prep_sample(self, sample):
        sample = sample.permute(1, 2, 0)
        sample = sample.detach().cpu().numpy()
        return sample

    def plot_sample(self, sample):
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
