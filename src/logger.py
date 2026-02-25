from src.config import TENSORBOARD_DIR
from monai.visualize import plot_2d_or_3d_image, img2tensorboard
from monai.handlers import TensorBoardHandler
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import sys

class Logger:
    def __init__(self, training_loader_length, validation_loader_length):
        self.train_length = training_loader_length
        self.validation_length = validation_loader_length
        # self.writer = SummaryWriter(TENSORBOARD_DIR)
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


    def clear_tensorboard(self):
        if os.path.exists(TENSORBOARD_DIR):
            print('Clearing Tensorboard Cache...')
            shutil.rmtree(TENSORBOARD_DIR)

