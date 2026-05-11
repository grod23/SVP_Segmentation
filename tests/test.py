from src.config import DEVICE, DISEASE_KEY, SVP_CLASS_KEY
from monai.transforms import Activations
import torch
import os
import numpy as np


class Test:
    def __init__(self, model, testing_loader, logger, visuals):
        self.model          = model
        self.testing_loader = testing_loader
        self.logger         = logger
        self.visuals        = visuals
        self.sigmoid        = Activations(sigmoid=True)

    # ─────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────

    def test_model(self):
        from sklearn.metrics import roc_auc_score, average_precision_score
        from scipy.stats import pearsonr

        pulse_probs, pulse_targets = [], []
        seg_probs,   seg_targets   = [], []

        with torch.no_grad():
            for batch in self.testing_loader:
                X_image, y_mask, original_images, metadata = batch
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask  = y_mask.to(DEVICE,  non_blocking=torch.cuda.is_available())

                pulse_logits, seg_logits = self.model(X_image)

                pulse_gt = (y_mask[:, 1] != y_mask[:, 0]).float()
                seg_gt   = y_mask[:, 0].float()

                pulse_probs.append(torch.sigmoid(pulse_logits).detach().cpu().flatten())
                pulse_targets.append(pulse_gt.detach().cpu().flatten())
                seg_probs.append(torch.sigmoid(seg_logits).detach().cpu().flatten())
                seg_targets.append(seg_gt.detach().cpu().flatten())

        def eval_soft(probs_list, targets_list, label):
            y_p = torch.cat(probs_list).numpy()
            y_t = torch.cat(targets_list).numpy()
            sd  = (2.0*(y_p*y_t).sum()) / (y_p.sum()+y_t.sum()+1e-8)
            if len(y_t) > 5_000_000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(y_t), 5_000_000, replace=False)
                auc = roc_auc_score(y_t[idx], y_p[idx])
                ap  = average_precision_score(y_t[idx], y_p[idx])
            else:
                auc = roc_auc_score(y_t, y_p)
                ap  = average_precision_score(y_t, y_p)
            r, _ = pearsonr(y_p, y_t)
            fg   = y_p[y_t == 1]
            bg   = y_p[y_t == 0]
            sep  = fg.mean() - bg.mean()
            print(f"  {label}:")
            print(f"    Soft Dice={sd:.4f}  AUROC={auc:.4f}  AP={ap:.4f}  r={r:.4f}")
            print(f"    fg mean={fg.mean():.3f}  bg mean={bg.mean():.3f}  gap={sep:.3f}", end="")
            if sep > 0.3:
                print(f"  ✓  threshold ~{bg.mean()+sep/2:.2f}")
            elif sep > 0.1:
                print("  △  learning")
            else:
                print("  ⚠️  weak")

        epoch = self.logger.current_epoch
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} — Evaluation")
        print(f"{'='*60}")
        eval_soft(pulse_probs, pulse_targets, "PULSATION MAP")
        eval_soft(seg_probs,   seg_targets,   "SEGMENTATION  (trough)")
        print(f"{'='*60}\n")

    # ─────────────────────────────────────────────────────────────────
    # Visualisation loop
    # ─────────────────────────────────────────────────────────────────

    def create_pulsation_mask(self, trough_mask, peak_mask,
                              save_path="pulsation.gif", amplify=2.0):
        return self.visuals.create_pulsation_mask(
            trough_mask=trough_mask, peak_mask=peak_mask,
            save_path=save_path, n_frames=30, fps=15, amplify=amplify,
        )

    def test_pulsation_mask(self, output_dir: str = "outputs/pulsation",
                            pulse_threshold: float = 0.25):
        """
        Three outputs per sample:
          _comparison.png      — 3x3 figure with originals, GT, pred heatmaps, overlays
          _gt_pulsation.gif    — animated GT trough↔peak
          _pred_soft.gif       — soft GIF: vessel opacity from seg prob,
                                 pulsation zone glows from pulse prob (NO threshold)
          _pred_binary.gif     — binary GIF at pulse_threshold=0.25
        """
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.testing_loader):
                X_image, y_mask, original_images, metadata = batch
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())
                y_mask  = y_mask.to(DEVICE,  non_blocking=torch.cuda.is_available())

                mask_min,    mask_max    = y_mask[:, 0], y_mask[:, 1]
                orig_trough, orig_peak   = original_images[:, 0], original_images[:, 1]

                pulse_logits, seg_logits = self.model(X_image)
                pulse_probs = torch.sigmoid(pulse_logits)   # [B,1,H,W]
                seg_probs   = torch.sigmoid(seg_logits)     # [B,1,H,W]
                pulse_gt    = (mask_max != mask_min).float()

                for i in range(X_image.shape[0]):
                    meta = {k: v[i] for k, v in metadata[0].items()}
                    sid  = f"batch{batch_idx:03d}_sample{i:02d}"
                    print(f"\n[{sid}] SVP: {meta[SVP_CLASS_KEY]} | Disease: {meta[DISEASE_KEY]}")

                    gt_px   = pulse_gt[i].sum().item()
                    pred_px = (pulse_probs[i] > pulse_threshold).float().sum().item()
                    seg_px  = (seg_probs[i] > 0.5).float().sum().item()
                    print(f"  GT pulsation px : {gt_px:.0f}")
                    print(f"  Pred pulse px   : {pred_px:.0f}  (thresh={pulse_threshold})")
                    print(f"  Pred vessel px  : {seg_px:.0f}  (thresh=0.5)")
                    print(f"  GT vessel px    : {mask_min[i].sum().item():.0f}")

                    # ── 1. Comparison figure ──────────────────────────────
                    self.visuals.plot_pulsation_map_full(
                        img_trough=orig_trough[i],
                        img_peak=orig_peak[i],
                        mask_trough=mask_min[i],
                        mask_peak=mask_max[i],
                        pulse_gt=pulse_gt[i],
                        pulse_prob=pulse_probs[i],
                        seg_prob=seg_probs[i],
                        pulse_threshold=pulse_threshold,
                        save_path=os.path.join(output_dir, f"{sid}_comparison.png"),
                    )

                    # ── 2. GT presentation figure (clean, labelled) ───────
                    self.visuals.plot_gt_pulsation_figure(
                        img_trough=orig_trough[i],
                        img_peak=orig_peak[i],
                        mask_trough=mask_min[i],
                        mask_peak=mask_max[i],
                        save_path=os.path.join(output_dir, f"{sid}_gt_figure.png"),
                        title_fontsize=16,
                    )

                    # ── 3. GT animated GIF ────────────────────────────────
                    self.create_pulsation_mask(
                        trough_mask=mask_min[i],
                        peak_mask=mask_max[i],
                        save_path=os.path.join(output_dir, f"{sid}_gt_pulsation.gif"),
                        amplify=2.0,
                    )

                    # ── 3. Soft GIF — no threshold, smooth vessel edges ───
                    self.visuals.create_soft_pulsation_gif(
                        seg_prob=seg_probs[i],
                        pulse_prob=pulse_probs[i],
                        save_path=os.path.join(output_dir, f"{sid}_pred_soft.gif"),
                    )

                    # ── 4. Binary GIF — thresholded at pulse_threshold ────
                    stable        = (seg_probs[i] > 0.5).float()
                    pred_pulse_bin = (pulse_probs[i] > pulse_threshold).float()
                    pred_peak_bin  = torch.clamp(stable + pred_pulse_bin, 0, 1)
                    self.create_pulsation_mask(
                        trough_mask=stable,
                        peak_mask=pred_peak_bin,
                        save_path=os.path.join(output_dir, f"{sid}_pred_binary.gif"),
                        amplify=2.0,
                    )