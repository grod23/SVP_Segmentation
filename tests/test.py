from src.config import DEVICE
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric, ROCAUCMetric
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from monai.transforms import Activations, AsDiscrete
import torch


class Test:
    def __init__(self, model, testing_loader, logger, visuals):
        self.model = model
        self.testing_loader = testing_loader
        self.logger = logger
        self.visuals = visuals
        # Metrics
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean"
        )
        self.iou_metric = MeanIoU(
            include_background=True
        )
        self.confusion_metric = ConfusionMatrixMetric(
            metric_name=["sensitivity", "specificity", "precision"],
            include_background=True
        )
        self.roc_auc_metric = ROCAUCMetric()
        # Post-processing
        self.sigmoid = Activations(sigmoid=True)
        self.threshold = AsDiscrete(threshold=0.7)


    def test_model(self):
        # Reset metrics at start of validation
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.confusion_metric.reset()
        self.roc_auc_metric.reset()
        self.model.eval()
        with torch.no_grad():
            for batch in self.testing_loader:
                X_image, y_mask, original_image, metadata = batch
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask = y_mask.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_predicted = self.model(X_image)
                ##########Converting predicted mask to binary##########
                min_val = y_predicted.min()
                max_val = y_predicted.max()
                if max_val > min_val:
                    y_predicted = (y_predicted - min_val) / (max_val - min_val)
                    y_predicted = (y_predicted > 0.5).float()
                else:
                    y_predicted = torch.zeros_like(y_predicted, dtype=torch.float32)
                self.visuals.plot_image_results(original_image[0], y_mask[0], y_predicted[0])
                ##########################################################
                # ---- Post-processing ----
                probs = self.sigmoid(y_predicted)
                preds = self.threshold(probs)
                # ---- Metrics ----
                self.dice_metric(preds, y_mask)
                self.iou_metric(preds, y_mask)
                self.confusion_metric(preds, y_mask)

                # Move tensors to CPU and convert to NumPy
                y_mask_np = y_mask.detach().cpu().numpy().flatten()
                y_predicted_np = preds.detach().cpu().numpy().flatten()
                # SKLearn Metrics
                score_jaccard = jaccard_score(y_mask_np, y_predicted_np, average='binary')
                score_f1 = f1_score(y_mask_np, y_predicted_np, average='binary')
                score_recall = recall_score(y_mask_np, y_predicted_np, average='binary')
                score_precision = precision_score(y_mask_np, y_predicted_np, average='binary')
                score_accuracy = accuracy_score(y_mask_np, y_predicted_np)
                print(
                    f"Epoch {self.logger.current_epoch} | "
                    f"Jaccard: {score_jaccard:.4f} | "
                    f"F1 Score: {score_f1:.4f} | "
                    f"Recall: {score_recall:.4f} | "
                    f"Precision: {score_precision:.4f} | "
                    f"Accuracy: {score_accuracy:.4f} | "
                )


                # ROC-AUC uses probabilities (no threshold)
                # self.roc_auc_metric(probs, y_mask)

        dice = self.dice_metric.aggregate().item()
        iou = self.iou_metric.aggregate().item()

        sensitivity, specificity, precision = self.confusion_metric.aggregate()
        sensitivity = sensitivity.item()
        specificity = specificity.item()
        precision = precision.item()

        # roc_auc = self.roc_auc_metric.aggregate().item()

        self.dice_metric.reset()
        self.iou_metric.reset()
        self.confusion_metric.reset()
        # self.roc_auc_metric.reset()

        print(
            f"Epoch {self.logger.current_epoch} | "
            f"Dice: {dice:.4f} | "
            f"IoU: {iou:.4f} | "
            f"Sensitivity: {sensitivity:.4f} | "
            f"Specificity: {specificity:.4f} | "
            f"Precision: {precision:.4f} | "
            # f"AUC: {roc_auc:.4f}"
        )


    def test_pulsation_mask(self):
        with torch.no_grad():
            for batch in self.testing_loader:
                X_image, y_mask, original_image, metadata = batch
                video_title = metadata[]
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask = y_mask.to(DEVICE, non_blocking=torch.cuda.is_available())
