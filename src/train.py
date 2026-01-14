from src import EPOCHS, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, DataUtils, Logger
import torch
import cv2
from monai.losses import DiceCELoss
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

        # self.optimizer = torch.optim.AdamW(
        #     params=self.model.parameters(),
        #     lr=LEARNING_RATE,
        #     weight_decay=WEIGHT_DECAY
        # )

    def visualize_sample(self, batches=5):
        for _ in range(batches):
            batch = next(iter(self.training_loader))
            self.logger.visualize_batch(batch)
            self.logger.clear_tensorboard()


