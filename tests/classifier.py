"""
SVP Classifier
==============
Uses pulsation map statistics as features to classify whether a patient
has SVP (Superficial Vascular Plexus involvement) or not.

The key clinical insight:
  - SVP patients show REDUCED or ABSENT pulsatility
  - Normal vessels show clear pulsation (meaningful dilation/contraction)
  - Features derived from the pulsation map capture this difference

We optimise for RECALL (sensitivity) — catching all true SVP cases
is more important than avoiding false positives in a screening context.

Usage:
    from svp_classifier import SVPClassifier
    clf = SVPClassifier()
    clf.fit(train_loader, model)
    results = clf.evaluate(test_loader, model)
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score,
    RocCurveDisplay
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.config import DEVICE, SVP_CLASS_KEY, DISEASE_KEY


# ─────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_pulsation_features(pulse_prob: np.ndarray,
                                seg_prob:   np.ndarray) -> dict:
    """
    Extract scalar features from one sample's pulsation + segmentation maps.

    Args:
        pulse_prob : [H, W] numpy array — predicted pulsation probabilities
        seg_prob   : [H, W] numpy array — predicted vessel probabilities

    Returns:
        dict of scalar features
    """
    vessel_mask = seg_prob > 0.5
    vessel_px   = vessel_mask.sum()

    if vessel_px == 0:
        # No vessel detected — return zeros
        return {f: 0.0 for f in _feature_names()}

    # Restrict pulsation analysis to vessel region
    pulse_in_vessel = pulse_prob[vessel_mask]

    # ── Pulsation magnitude features ─────────────────────────────────
    pulse_mean   = float(pulse_in_vessel.mean())
    pulse_max    = float(pulse_in_vessel.max())
    pulse_std    = float(pulse_in_vessel.std())
    pulse_median = float(np.median(pulse_in_vessel))

    # Fraction of vessel pixels with meaningful pulsation (>0.3 prob)
    pulse_frac_03 = float((pulse_in_vessel > 0.30).mean())
    pulse_frac_05 = float((pulse_in_vessel > 0.50).mean())

    # Total predicted pulsation area normalised by vessel area
    pulse_norm_area = float((pulse_prob > 0.25).sum() / max(vessel_px, 1))

    # ── Vessel size features ──────────────────────────────────────────
    vessel_area_norm = float(vessel_px / (pulse_prob.shape[0] * pulse_prob.shape[1]))

    # ── Spatial consistency ───────────────────────────────────────────
    # High pulsation should be concentrated (low entropy = focused signal)
    p = pulse_in_vessel.clip(1e-6, 1 - 1e-6)
    entropy = float(-((p * np.log(p)) + ((1-p) * np.log(1-p))).mean())

    return {
        'pulse_mean':       pulse_mean,
        'pulse_max':        pulse_max,
        'pulse_std':        pulse_std,
        'pulse_median':     pulse_median,
        'pulse_frac_03':    pulse_frac_03,
        'pulse_frac_05':    pulse_frac_05,
        'pulse_norm_area':  pulse_norm_area,
        'vessel_area_norm': vessel_area_norm,
        'pulse_entropy':    entropy,
    }


def _feature_names():
    return ['pulse_mean', 'pulse_max', 'pulse_std', 'pulse_median',
            'pulse_frac_03', 'pulse_frac_05', 'pulse_norm_area',
            'vessel_area_norm', 'pulse_entropy']


# ─────────────────────────────────────────────────────────────────────────
#  Classifier
# ─────────────────────────────────────────────────────────────────────────

class SVPClassifier:
    """
    Logistic regression classifier on pulsation map features.

    Optimised for maximum recall (sensitivity) — in a screening context
    it is more costly to miss an SVP case than to flag a false positive.
    The decision threshold is tuned on the training set to achieve
    recall >= target_recall before being applied to the test set.
    """

    def __init__(self, target_recall: float = 0.90):
        """
        Args:
            target_recall : minimum recall to achieve on training data.
                            0.90 = catch at least 90% of SVP cases.
                            Increase toward 1.0 for stricter screening.
        """
        self.target_recall = target_recall
        self.pipeline      = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(
                class_weight='balanced',   # handles class imbalance
                max_iter=1000,
                C=0.1,                     # regularisation — prevents overfit
                random_state=42,
            )),
        ])
        self.threshold = 0.5   # updated by fit()
        self.feature_names = _feature_names()

    def _collect_features(self, loader, model):
        """Run model on all batches and extract per-sample features + labels."""
        all_features = []
        all_labels   = []
        all_meta     = []

        model.eval()
        with torch.no_grad():
            for batch in loader:
                X_image, y_mask, original_images, metadata = batch
                X_image = X_image.to(DEVICE, non_blocking=torch.cuda.is_available())

                pulse_logits, seg_logits = model(X_image)
                pulse_probs = torch.sigmoid(pulse_logits).detach().cpu().numpy()
                seg_probs   = torch.sigmoid(seg_logits).detach().cpu().numpy()

                for i in range(X_image.shape[0]):
                    meta    = {k: v[i] for k, v in metadata[0].items()}
                    svp_val = meta[SVP_CLASS_KEY]

                    # SVP label: 1 = has SVP, 0 = normal
                    # Adjust this logic to match your SVP_CLASS_KEY encoding
                    if isinstance(svp_val, torch.Tensor):
                        svp_val = svp_val.item()
                    label = int(svp_val)

                    pp = pulse_probs[i, 0]   # [H, W]
                    sp = seg_probs[i, 0]     # [H, W]

                    feats = extract_pulsation_features(pp, sp)
                    all_features.append([feats[k] for k in self.feature_names])
                    all_labels.append(label)
                    all_meta.append({
                        'disease': meta.get(DISEASE_KEY, 'Unknown'),
                        'svp':     label,
                    })

        return np.array(all_features), np.array(all_labels), all_meta

    def fit(self, train_loader, model):
        """
        Train classifier on training loader.
        Tunes decision threshold to achieve target_recall.
        """
        print(f"\nCollecting training features...")
        X_train, y_train, meta = self._collect_features(train_loader, model)

        print(f"  Samples: {len(y_train)}  |  "
              f"SVP: {y_train.sum()}  |  "
              f"Normal: {(y_train==0).sum()}")

        self.pipeline.fit(X_train, y_train)

        # Tune threshold on training set to hit target_recall
        probs = self.pipeline.predict_proba(X_train)[:, 1]
        self.threshold = self._find_threshold(probs, y_train)

        train_preds = (probs >= self.threshold).astype(int)
        rec = recall_score(y_train, train_preds, zero_division=0)
        print(f"  Threshold tuned to {self.threshold:.3f} → "
              f"train recall={rec:.3f}  (target={self.target_recall})")

        # Feature importances
        coef = self.pipeline.named_steps['clf'].coef_[0]
        print(f"\n  Feature importances (logistic regression coefficients):")
        for name, c in sorted(zip(self.feature_names, coef),
                               key=lambda x: abs(x[1]), reverse=True):
            direction = "↑ SVP" if c > 0 else "↓ SVP"
            print(f"    {name:<22}: {c:+.3f}  {direction}")

        return self

    def evaluate(self, test_loader, model, save_dir: str = "outputs/classifier"):
        """
        Evaluate on test loader. Prints full report and saves figures.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nCollecting test features...")
        X_test, y_test, meta = self._collect_features(test_loader, model)

        probs  = self.pipeline.predict_proba(X_test)[:, 1]
        preds  = (probs >= self.threshold).astype(int)

        recall    = recall_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        f1        = f1_score(y_test, preds, zero_division=0)
        auroc     = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        print(f"\n{'='*55}")
        print(f"SVP Classification Results")
        print(f"{'='*55}")
        print(f"  Decision threshold : {self.threshold:.3f}")
        print(f"  Test samples       : {len(y_test)}")
        print(f"  SVP / Normal       : {y_test.sum()} / {(y_test==0).sum()}")
        print(f"{'='*55}")
        print(f"  Recall (sensitivity): {recall:.4f}  ← KEY — missed SVP cases")
        print(f"  Precision           : {precision:.4f}")
        print(f"  F1 score            : {f1:.4f}")
        print(f"  AUROC               : {auroc:.4f}")
        print(f"{'='*55}")
        print(f"  True  Positives  (SVP correctly detected)   : {tp}")
        print(f"  False Negatives  (SVP missed — bad!)        : {fn}")
        print(f"  True  Negatives  (Normal correctly cleared) : {tn}")
        print(f"  False Positives  (Normal flagged as SVP)    : {fp}")
        if fn > 0:
            print(f"\n  ⚠️  {fn} SVP case(s) missed. Lower threshold to reduce.")
        else:
            print(f"\n  ✓  All SVP cases detected.")
        print(f"{'='*55}\n")

        # ── Figures ──────────────────────────────────────────────────
        self._plot_confusion_matrix(cm, y_test, preds,
            save_path=f"{save_dir}/confusion_matrix.png")
        self._plot_roc(y_test, probs, auroc, recall,
            save_path=f"{save_dir}/roc_curve.png")
        self._plot_feature_distributions(X_test, y_test,
            save_path=f"{save_dir}/feature_distributions.png")

        return {
            'recall': recall, 'precision': precision,
            'f1': f1, 'auroc': auroc,
            'tp': int(tp), 'fp': int(fp),
            'tn': int(tn), 'fn': int(fn),
            'threshold': self.threshold,
        }

    def _find_threshold(self, probs, y_true):
        """Find lowest threshold that achieves target_recall."""
        best_thresh = 0.5
        for t in np.linspace(0.05, 0.95, 180):
            preds = (probs >= t).astype(int)
            if recall_score(y_true, preds, zero_division=0) >= self.target_recall:
                best_thresh = t
                break
        return float(best_thresh)

    def _plot_confusion_matrix(self, cm, y_true, y_pred, save_path):
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        total = len(y_true)

        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        im = ax.imshow(cm, cmap='Blues', vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        labels = [['True Negative\n(Normal → Normal)',
                   'False Positive\n(Normal → SVP)'],
                  ['False Negative\n(SVP → Normal)\n⚠️ Missed',
                   'True Positive\n(SVP → SVP)']]

        for i in range(2):
            for j in range(2):
                val  = cm[i, j]
                pct  = val / total * 100
                text = f"{val}\n({pct:.1f}%)\n{labels[i][j]}"
                color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted Normal', 'Predicted SVP'], fontsize=12)
        ax.set_yticklabels(['Actual Normal', 'Actual SVP'], fontsize=12)
        ax.set_title('SVP Classification — Confusion Matrix',
                     fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved confusion matrix -> {save_path}")
        plt.show()
        plt.close()

    def _plot_roc(self, y_true, probs, auroc, recall, save_path):
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, probs)

        # Find operating point
        op_idx = np.argmin(np.abs(thresholds - self.threshold))

        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.plot(fpr, tpr, color='#0078ff', lw=2.5,
                label=f'ROC curve (AUC = {auroc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        ax.scatter(fpr[op_idx], tpr[op_idx], color='#ff8c00', s=120, zorder=5,
                   label=f'Operating point\n(threshold={self.threshold:.2f}, recall={recall:.2f})')
        ax.axhline(y=recall, color='#ff8c00', linestyle=':', alpha=0.6)

        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
        ax.set_title('SVP Classification — ROC Curve',
                     fontsize=14, fontweight='bold', pad=12)
        ax.legend(fontsize=10, loc='lower right')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved ROC curve -> {save_path}")
        plt.show()
        plt.close()

    def _plot_feature_distributions(self, X, y, save_path):
        """Box plots of each feature split by SVP vs Normal."""
        n_feat = len(self.feature_names)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor='white')
        axes = axes.flat

        for i, (ax, name) in enumerate(zip(axes, self.feature_names)):
            normal_vals = X[y == 0, i]
            svp_vals    = X[y == 1, i]
            bp = ax.boxplot([normal_vals, svp_vals],
                            patch_artist=True,
                            medianprops=dict(color='black', linewidth=2))
            bp['boxes'][0].set_facecolor('#aaccff')   # light blue — normal
            bp['boxes'][1].set_facecolor('#ffcc88')   # light orange — SVP
            ax.set_xticklabels(['Normal', 'SVP'], fontsize=11)
            ax.set_title(name.replace('_', ' ').title(), fontsize=11,
                         fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_facecolor('white')

        legend_patches = [
            mpatches.Patch(facecolor='#aaccff', label='Normal'),
            mpatches.Patch(facecolor='#ffcc88', label='SVP'),
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=2, fontsize=12, framealpha=0.9)
        fig.suptitle('Pulsation Feature Distributions by SVP Status',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved feature distributions -> {save_path}")
        plt.show()
        plt.close()