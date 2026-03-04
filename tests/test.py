from src.config import DEVICE, VIDEO_TITLE_KEY, FRAME_KEY, MIN_MAX_KEY, DISEASE_KEY, SVP_CLASS_KEY
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
from monai.transforms import Activations, AsDiscrete
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import torch
import os


class Test:
    def __init__(self, model, testing_loader, logger, visuals):
        self.model          = model
        self.testing_loader = testing_loader
        self.logger         = logger
        self.visuals        = visuals

        # MONAI metrics
        self.dice_metric      = DiceMetric(include_background=True, reduction="mean")
        self.iou_metric       = MeanIoU(include_background=True)
        self.confusion_metric = ConfusionMatrixMetric(
            metric_name=["sensitivity", "specificity", "precision"],
            include_background=True
        )

        # Post-processing: raw logits → sigmoid → threshold
        self.sigmoid   = Activations(sigmoid=True)
        self.threshold = AsDiscrete(threshold=0.5)

    def _postprocess(self, raw_logits):
        """Convert raw model output to binary predictions."""
        probs = self.sigmoid(raw_logits)
        return self.threshold(probs), probs

    def _reset_metrics(self):
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.confusion_metric.reset()

    # ─────────────────────────────────────────────────────────────────
    # Standard evaluation
    # ─────────────────────────────────────────────────────────────────

    def test_model(self):
        self._reset_metrics()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in self.testing_loader:
                X_image, y_mask, original_images, metadata = batch
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask  = y_mask.to(DEVICE,  non_blocking=torch.cuda.is_available())

                y_predicted = self.model(X_image)

                B, H, W     = X_image.shape[0], X_image.shape[-2], X_image.shape[-1]
                y_predicted_flat = y_predicted.view(B * 2, 1, H, W)
                y_mask_flat      = y_mask.view(B * 2, 1, H, W)

                preds, _ = self._postprocess(y_predicted_flat)

                self.dice_metric(preds, y_mask_flat)
                self.iou_metric(preds, y_mask_flat)
                self.confusion_metric(preds, y_mask_flat)

                all_preds.append(preds.detach().cpu().flatten())
                all_targets.append(y_mask_flat.detach().cpu().flatten())

        dice = self.dice_metric.aggregate().item()
        iou  = self.iou_metric.aggregate().item()
        sensitivity, specificity, precision = self.confusion_metric.aggregate()
        self._reset_metrics()

        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()

        sklearn_scores = {
            "Jaccard":   jaccard_score(y_true, y_pred, average='binary'),
            "F1":        f1_score(y_true, y_pred, average='binary'),
            "Recall":    recall_score(y_true, y_pred, average='binary'),
            "Precision": precision_score(y_true, y_pred, average='binary'),
            "Accuracy":  accuracy_score(y_true, y_pred),
        }

        epoch = self.logger.current_epoch
        print(
            f"Epoch {epoch} | "
            f"Dice: {dice:.4f} | IoU: {iou:.4f} | "
            f"Sensitivity: {sensitivity.item():.4f} | "
            f"Specificity: {specificity.item():.4f} | "
            f"Precision: {precision.item():.4f}"
        )
        print(
            f"Epoch {epoch} [sklearn] | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in sklearn_scores.items())
        )

    # ─────────────────────────────────────────────────────────────────
    # Pulsation mask creation
    # ─────────────────────────────────────────────────────────────────

    def create_pulsation_mask(self, trough_mask, peak_mask,
                              save_path: str = "pulsation.gif",
                              amplify: float = 2.0):
        """
        Thin wrapper so test.py can call visuals.create_pulsation_mask
        with consistent defaults suited to retinal vessel pulsation.

        Args:
            trough_mask : tensor [1, H, W]  – diastole / min frame prediction
            peak_mask   : tensor [1, H, W]  – systole  / max frame prediction
            save_path   : output GIF filepath
            amplify     : morphological dilation iterations on diff zones.
                          Default 2.0 — widens 1-2px vessel boundary changes
                          so they are visible in the animation.
                          Set to 1.0 to show true pixel-level difference only.
        """
        return self.visuals.create_pulsation_mask(
            trough_mask=trough_mask,
            peak_mask=peak_mask,
            save_path=save_path,
            n_frames=30,
            fps=15,
            amplify=amplify,
        )

    # ─────────────────────────────────────────────────────────────────
    # Pulsation test loop
    # ─────────────────────────────────────────────────────────────────

    def test_pulsation_mask(self, output_dir: str = "outputs/pulsation"):
        """
        For each sample in the test set:
          1. Run backbone on trough and peak frames independently.
          2. Postprocess both predictions to binary masks.
          3. Save a pulsation GIF + static comparison figure.

        Args:
            output_dir : directory where GIFs and figures are saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.testing_loader):
                X_image, y_mask, original_images, metadata = batch

                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask  = y_mask.to(DEVICE,  non_blocking=torch.cuda.is_available())

                # Split paired frames: each [B, 1, H, W]
                img_min,  img_max  = X_image[:, 0], X_image[:, 1]
                mask_min, mask_max = y_mask[:, 0],  y_mask[:, 1]

                # Independent backbone predictions for trough and peak
                y_trough_raw = self.model.backbone(img_min)   # [B, 1, H, W]  raw logits
                y_peak_raw   = self.model.backbone(img_max)   # [B, 1, H, W]  raw logits

                # ── Diagnostic: raw backbone output ──────────────────
                print("\n── Backbone raw output (trough) ──")
                print(f"  shape  : {y_trough_raw.shape}")
                print(f"  dtype  : {y_trough_raw.dtype}")
                print(f"  min    : {y_trough_raw.min():.4f}")
                print(f"  max    : {y_trough_raw.max():.4f}")
                print(f"  mean   : {y_trough_raw.mean():.4f}")
                print(f"  unique : {y_trough_raw.unique()[:10].tolist()}")

                print("\n── Backbone raw output (peak) ──")
                print(f"  shape  : {y_peak_raw.shape}")
                print(f"  dtype  : {y_peak_raw.dtype}")
                print(f"  min    : {y_peak_raw.min():.4f}")
                print(f"  max    : {y_peak_raw.max():.4f}")
                print(f"  mean   : {y_peak_raw.mean():.4f}")
                print(f"  unique : {y_peak_raw.unique()[:10].tolist()}")

                # ── Diagnostic: ground truth masks ───────────────────
                print("\n── GT mask (trough / min) ──")
                print(f"  shape  : {mask_min.shape}")
                print(f"  dtype  : {mask_min.dtype}")
                print(f"  min    : {mask_min.min():.4f}")
                print(f"  max    : {mask_min.max():.4f}")
                print(f"  unique : {mask_min.unique()[:10].tolist()}")
                print(f"  #fg px : {mask_min.sum().item():.0f}")

                print("\n── GT mask (peak / max) ──")
                print(f"  shape  : {mask_max.shape}")
                print(f"  dtype  : {mask_max.dtype}")
                print(f"  min    : {mask_max.min():.4f}")
                print(f"  max    : {mask_max.max():.4f}")
                print(f"  unique : {mask_max.unique()[:10].tolist()}")
                print(f"  #fg px : {mask_max.sum().item():.0f}")

                # Soft probabilities (sigmoid only — no threshold yet)
                y_trough_prob = self.sigmoid(y_trough_raw)   # [B, 1, H, W]  in (0, 1)
                y_peak_prob   = self.sigmoid(y_peak_raw)     # [B, 1, H, W]  in (0, 1)

                # ── Diagnostic: soft probs ────────────────────────────
                print("\n── Soft probs (trough) ──")
                print(f"  prob min/max/mean: {y_trough_prob.min():.4f} / {y_trough_prob.max():.4f} / {y_trough_prob.mean():.4f}")
                print("\n── Soft probs (peak) ──")
                print(f"  prob min/max/mean: {y_peak_prob.min():.4f} / {y_peak_prob.max():.4f} / {y_peak_prob.mean():.4f}")

                # ── Strategy: normalise then diff ─────────────────────
                # The backbone was trained as part of a paired model and never
                # produces negative logits in isolation — all pixels exceed the
                # 0.5 threshold. Fix: normalise each prob map to [0,1] relative
                # to itself (removes global offset bias), then threshold the
                # signed difference to isolate genuine dilation / contraction.
                def normalise(t):
                    mn, mx = t.min(), t.max()
                    return (t - mn) / (mx - mn + 1e-8)

                y_trough_norm = normalise(y_trough_prob)   # [B, 1, H, W]
                y_peak_norm   = normalise(y_peak_prob)     # [B, 1, H, W]

                prob_diff = y_peak_norm - y_trough_norm    # +ve = dilated, -ve = contracted

                # Tune DIFF_THRESHOLD down if diff pixels still reads 0
                DIFF_THRESHOLD = 0.05
                pred_dilation    = (prob_diff >  DIFF_THRESHOLD).float()
                pred_contraction = (prob_diff < -DIFF_THRESHOLD).float()

                print(f"\n── Prob-diff map (peak_norm - trough_norm) ──")
                print(f"  min / max / mean : {prob_diff.min():.4f} / {prob_diff.max():.4f} / {prob_diff.mean():.4f}")
                print(f"  |diff| > {DIFF_THRESHOLD} px  : {(prob_diff.abs() > DIFF_THRESHOLD).sum().item():.0f}")
                print(f"  dilation px      : {pred_dilation.sum().item():.0f}")
                print(f"  contraction px   : {pred_contraction.sum().item():.0f}")
                print(f"  GT diff px       : {(mask_max != mask_min).float().sum().item():.0f}")

                # Synthesise binary trough / peak masks for the GIF
                pred_core    = (y_trough_norm > 0.5).float()
                y_trough_bin = torch.clamp(pred_core + pred_contraction, 0, 1)
                y_peak_bin   = torch.clamp(pred_core + pred_dilation,    0, 1)

                print(f"\n── Synthesised binary masks ──")
                print(f"  y_trough_bin #fg px : {y_trough_bin.sum().item():.0f}")
                print(f"  y_peak_bin   #fg px : {y_peak_bin.sum().item():.0f}")
                print(f"  diff px             : {(y_peak_bin != y_trough_bin).float().sum().item():.0f}")
                print(f"  GT diff px          : {(mask_max != mask_min).float().sum().item():.0f}")

                for i in range(X_image.shape[0]):
                    meta_min  = {k: v[i] for k, v in metadata[0].items()}
                    svp_label = meta_min[SVP_CLASS_KEY]
                    disease   = meta_min[DISEASE_KEY]

                    sample_id = f"batch{batch_idx:03d}_sample{i:02d}"
                    print(f"\n[{sample_id}] SVP: {svp_label} | Disease: {disease}")

                    # ── Per-sample diagnostics ────────────────────────
                    print(f"  y_trough_bin[i] #fg px : {y_trough_bin[i].sum().item():.0f}")
                    print(f"  y_peak_bin[i]   #fg px : {y_peak_bin[i].sum().item():.0f}")
                    print(f"  mask_min[i]     #fg px : {mask_min[i].sum().item():.0f}")
                    print(f"  mask_max[i]     #fg px : {mask_max[i].sum().item():.0f}")
                    sample_diff_pred = (y_peak_bin[i] != y_trough_bin[i]).float().sum().item()
                    sample_diff_gt   = (mask_max[i]   != mask_min[i]).float().sum().item()
                    print(f"  Pred diff px (peak vs trough) : {sample_diff_pred:.0f}")
                    print(f"  GT   diff px (max  vs min   ) : {sample_diff_gt:.0f}")
                    if sample_diff_pred == 0:
                        print("  ⚠️  WARNING: still 0 diff — try lowering DIFF_THRESHOLD below 0.05.")

                    # ── 1. Predicted pulsation GIF ────────────────────
                    gif_path = os.path.join(output_dir, f"{sample_id}_pulsation.gif")
                    self.create_pulsation_mask(
                        trough_mask=y_trough_bin[i],
                        peak_mask=y_peak_bin[i],
                        save_path=gif_path,
                        amplify=2.0,
                    )

                    # ── 2. GT pulsation GIF ──────────────────────────
                    gt_gif_path = os.path.join(output_dir, f"{sample_id}_gt_pulsation.gif")
                    self.create_pulsation_mask(
                        trough_mask=mask_min[i],
                        peak_mask=mask_max[i],
                        save_path=gt_gif_path,
                        amplify=2.0,
                    )

                    # ── 3. Static 6-panel comparison figure ─────────
                    fig_path = os.path.join(output_dir, f"{sample_id}_comparison.png")
                    self.visuals.plot_pulsation_masks(
                        mask_max=mask_max[i],
                        mask_min=mask_min[i],
                        predicted_max=y_peak_bin[i],
                        predicted_min=y_trough_bin[i],
                        save_path=fig_path,
                    )