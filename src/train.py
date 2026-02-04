from src.config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from src.logger import Logger
from src.model import Segmentation_Model
from src import DataUtils, Test, Visualizer
import torch
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, TverskyLoss
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
        self.logger = Logger(len(self.training_loader), len(self.validation_loader))
        self.model = Segmentation_Model(backbone_name='unet').to(DEVICE)
        self.visuals = Visualizer(self.logger)
        self.tester = Test(self.model, self.testing_loader, self.logger, self.visuals)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=3)
        # self.loss_fn = DiceCELoss(
        #     include_background=True,
        #     to_onehot_y=False,
        #     sigmoid=True,
        #     softmax=False
        #                          )
        self.loss_fn = TverskyLoss(
            sigmoid=True,
            alpha=0.8,  # penalize FP
            beta=0.2,  # allow small FN
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            batch=True
        )
        # bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3]).to(DEVICE))
        # dice = DiceLoss(sigmoid=True)
        # loss = dice(y_predicted, y_mask) + 0.7 * bce(y_predicted, y_mask)
        # self.loss_fn = DiceFocalLoss(
        #     # include_background=True,
        #     to_onehot_y=False,
        #     sigmoid=True,
        #     smooth_nr=1e-5,
        #     smooth_dr=1e-5,
        #     batch=True,
        #     alpha=0.5,
        #     gamma=2.0
        # )

    def visualize_sample(self, batches=5):
        loader_iter = iter(self.training_loader)
        for _ in range(batches):
            batch = next(loader_iter)
            self.visuals.visualize_batch(batch)
            # self.logger.clear_tensorboard()


    def run_epoch(self):
        self.model.train()
        for train_batch in self.training_loader:
            self.optimizer.zero_grad()
            X_image, y_mask = train_batch
            # Convert to GPU
            X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
            y_mask = y_mask.to(DEVICE, non_blocking=torch.cuda.is_available())
            # print(f'X Shape: {X_image.shape}')
            # print(f'Y Shape: {y_mask.shape}')
            # Get predicted mask
            y_predicted = self.model(X_image)
            # print(f'Y Pred Shape: {y_predicted.shape}')
            loss = self.loss_fn(y_predicted, y_mask)
            loss.backward()
            self.optimizer.step()
            self.logger.log_epoch_loss(loss, train=True)

        # Validation
        self.model.eval()
        with torch.no_grad():
            for val_batch in self.validation_loader:
                X_image, y_mask = val_batch
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask = y_mask.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_predicted = self.model(X_image)
                loss = self.loss_fn(y_predicted, y_mask)
                self.logger.log_epoch_loss(loss, train=False)

        avg_train_loss, avg_val_loss = self.logger.get_average_loss()
        print(f'Epoch: {self.logger.current_epoch} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        # Update the scheduler
        # self.scheduler.step(avg_val_loss)


    def train(self):
        for epoch in range(EPOCHS):
            self.run_epoch()

