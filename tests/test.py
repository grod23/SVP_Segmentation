from src.config import DEVICE, ViDEO_TITLE_KEY, FRAME_KEY, MIN_MAX_KEY, DISEASE_KEY, SVP_CLASS_KEY
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric, ROCAUCMetric
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from monai.transforms import Activations, AsDiscrete
import torch
from pathlib import Path


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
        # self.load_model()

        pairs = {}
        with torch.no_grad():
            for batch in self.testing_loader:
                X_image, y_mask, original_image, metadata = batch

                video_titles = metadata[ViDEO_TITLE_KEY]
                frames = metadata[FRAME_KEY]
                min_max_values = metadata[MIN_MAX_KEY]
                disease = metadata[DISEASE_KEY]
                svp_class = metadata[SVP_CLASS_KEY]

                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask = y_mask.to(DEVICE, non_blocking=torch.cuda.is_available())

                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask = y_mask.to(DEVICE, non_blocking=torch.cuda.is_available())

                batch_size = len(video_titles)

                for i in range(batch_size):
                    title = video_titles[i]
                    min_max = min_max_values[i]

                    if title not in pairs:
                        pairs[title] = {}

                    pairs[title][min_max] = {
                        "image": X_image[i],
                        "mask": y_mask[i],
                        "original": original_image[i],
                        "frame": frames[i],
                        'SVP': svp_class[i],
                        'Disease': disease[i]
                    }

                    # When both min and max exist
                    if "min" in pairs[title] and "max" in pairs[title]:

                        min_data = pairs[title]["min"]
                        max_data = pairs[title]["max"]
                        svp = min_data['SVP']
                        disease_present = min_data['Disease']
                        print(f'SVP Present: {svp}')
                        print(f'Disease: {disease_present}')
                        # Add batch dimension
                        min_img = min_data["image"].unsqueeze(0)
                        max_img = max_data["image"].unsqueeze(0)

                        min_mask = min_data["mask"].unsqueeze(0)
                        max_mask = max_data["mask"].unsqueeze(0)

                        print(f'Min Shape: {min_img.shape}')
                        print(f'Max Shape: {max_img.shape}')

                        # Forward pass
                        y_min = self.model(min_img)
                        y_max = self.model(max_img)

                        # Normalize + threshold
                        def binarize(pred):
                            min_val = pred.min()
                            max_val = pred.max()
                            if max_val > min_val:
                                pred = (pred - min_val) / (max_val - min_val)
                                return (pred > 0.5).float()
                            else:
                                return torch.zeros_like(pred)

                        y_min = binarize(y_min)
                        y_max = binarize(y_max)

                        # Example pulsation difference
                        pulsation_pred = torch.abs(y_max - y_min)
                        pulsation_gt = torch.abs(max_mask - min_mask)

                        # Visualize
                        self.visuals.plot_pulsation_masks(
                            max_mask.squeeze(0),
                            min_mask.squeeze(0),
                            y_max.squeeze(0),
                            y_min.squeeze(0)
                        )

                        # Remove processed pair (prevents memory growth)
                        del pairs[title]



    def load_model(self):
        # Portable Root
        ROOT = Path(__file__).resolve().parents[1]
        MODEL_PATH = ROOT / 'results' / 'SVP_Seg.pth'
        # Load Model Weights
        self.model.load_state_dict(torch.load(MODEL_PATH))
        print(f'Loading Model from... {MODEL_PATH}')
        self.model.eval()  # Set to evaluation mode