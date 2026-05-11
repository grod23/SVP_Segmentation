from src.config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from src.logger import Logger
from src.model import SimpleUNet, SegmentationLoss
from src import DataUtils, Test, Visualizer
import torch
from pathlib import Path

print(f'Device Available: {torch.cuda.is_available()}')


class Train:
    def __init__(self):
        self.datautils = DataUtils()
        (
            self.training_loader,
            self.validation_loader,
            self.testing_loader,
        ) = self.datautils.create_dataloaders()

        self.logger  = Logger(len(self.training_loader), len(self.validation_loader))
        self.model   = SimpleUNet(in_channels=1).to(DEVICE)
        self.visuals = Visualizer(self.logger)
        self.tester  = Test(self.model, self.testing_loader, self.logger, self.visuals)

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.loss_fn = SegmentationLoss(pos_weight=65.0)

    def run_epoch(self):
        self.model.train()
        for batch in self.training_loader:
            self.optimizer.zero_grad()
            X_image, y_mask = batch
            X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
            y_mask  = y_mask.to(DEVICE,  non_blocking=torch.cuda.is_available())
            loss, components = self.loss_fn(self.model(X_image), y_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.logger.log_epoch_loss(loss, train=True)

        self.model.eval()
        with torch.no_grad():
            for batch in self.validation_loader:
                X_image, y_mask = batch
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask  = y_mask.to(DEVICE,  non_blocking=torch.cuda.is_available())
                loss, _ = self.loss_fn(self.model(X_image), y_mask)
                self.logger.log_epoch_loss(loss, train=False)

        avg_train, avg_val = self.logger.get_average_loss()
        print(f'Epoch {self.logger.current_epoch} | '
              f'Train: {avg_train:.4f} | Val: {avg_val:.4f} | '
              f'Seg: {components["seg"]:.4f}')
        self.scheduler.step(avg_val)

    def train(self):
        for epoch in range(EPOCHS):
            self.run_epoch()

    def save_model(self):
        torch.save(self.model.state_dict(), 'SVP_Seg.pth')

    def test_model(self, load_model=True):
        if load_model:
            self.load_model()
        self.model.eval()
        self.tester.test_model()
        self.tester.test_pulsation_mask()

    def classify_svp(self, load_model=True, target_recall=0.90):
        from tests.classifier import SVPClassifier
        if load_model:
            self.load_model()   
        self.model.eval()
        clf = SVPClassifier(target_recall=target_recall)
        clf.fit(self.training_loader, self.model)
        return clf.evaluate(self.testing_loader, self.model)

    def load_model(self):
        ROOT       = Path(__file__).resolve().parents[1]
        MODEL_PATH = ROOT / 'results' / 'SVP_Seg.pth'
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded from: {MODEL_PATH}")
        self.model.eval()