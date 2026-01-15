from src.config import DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, IMAGE_KEY, MASK_KEY
from src.logger import Logger
from src.model import Segmentation_Model
from src import DataUtils
import torch
import cv2
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.engines import SupervisedTrainer, SupervisedEvaluator


print(f'Device Available: {torch.cuda.is_available()}')


class Train:
    def __init__(self):
        # Init Dataloaders
        self.datautils = DataUtils()
        (
            self.training_loader,
            self.validation_loader,
            self.testing_loader,
        ) = self.datautils.create_dataloaders()
        self.logger = Logger()
        self.model = Segmentation_Model(backbone_name='unet').to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False
                                  )

    def visualize_sample(self, batches=5):
        for _ in range(batches):
            batch = next(iter(self.training_loader))
            self.logger.visualize_batch(batch)
            self.logger.clear_tensorboard()


    def run_epoch(self):
        self.model.train()
        for batch in self.training_loader:
            self.optimizer.zero_grad()
            X_image_batch = batch[IMAGE_KEY]
            y_mask_batch = batch[MASK_KEY]
            print(f'X Shape: {X_image_batch.shape}')
            print(f'Y Shape: {y_mask_batch.shape}')
            # Conver to GPU
            X_image_batch = X_image_batch.to(DEVICE)
            y_mask_batch = y_mask_batch.to(DEVICE)
            # Get predicted mask
            y_predicted = self.model(X_image_batch)
            print(f'Y Pred Sha[e: {y_predicted.shape}')
            loss = self.loss_fn(y_predicted, y_mask_batch)
            loss.backward()
            self.optimizer.step()




