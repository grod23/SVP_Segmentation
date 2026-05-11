from monai.networks.nets import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  SimpleUNet  —  Baseline model
#
#  Takes paired input [B, 2, 1, H, W], segments each frame independently
#  through a shared UNet backbone, returns both segmentations.
#
#  This is the clean baseline before adding optic disc localisation.
# ═══════════════════════════════════════════════════════════════════════════

class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.backbone = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, X_image):
        """
        Args:
            X_image : [B, 2, 1, H, W]  paired frames (trough=0, peak=1)
        Returns:
            [B, 2, 1, H, W]  segmentation logits for both frames
        """
        img_trough = X_image[:, 0]                          # [B, 1, H, W]
        img_peak   = X_image[:, 1]                          # [B, 1, H, W]
        out_trough = self.backbone(img_trough)              # [B, 1, H, W]
        out_peak   = self.backbone(img_peak)                # [B, 1, H, W]
        return torch.stack([out_trough, out_peak], dim=1)   # [B, 2, 1, H, W]


# Alias
Segmentation_Model = SimpleUNet


# ═══════════════════════════════════════════════════════════════════════════
#  SegmentationLoss  —  Simple Dice + BCE
#
#  Applied independently to each frame then averaged.
#  No pulsation loss, no size loss, no sparsity — just clean segmentation.
# ═══════════════════════════════════════════════════════════════════════════

class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight: float = 65.0):
        """
        Args:
            pos_weight : BCE positive class weight.
                         ~65 for 1.5% foreground (vessel pixels).
        """
        super().__init__()
        self.register_buffer('pw', torch.tensor([pos_weight]))

    def forward(self, y_pred, y_mask):
        """
        Args:
            y_pred  : [B, 2, 1, H, W]  raw logits
            y_mask  : [B, 2, 1, H, W]  binary GT masks
        Returns:
            loss (scalar), components dict
        """
        pred_trough = y_pred[:, 0]
        pred_peak   = y_pred[:, 1]
        mask_trough = y_mask[:, 0].float()
        mask_peak   = y_mask[:, 1].float()

        loss = (self._bce_dice(pred_trough, mask_trough) +
                self._bce_dice(pred_peak,   mask_peak)) / 2.0

        return loss, {'seg': loss.item(), 'total': loss.item()}

    def _bce_dice(self, logits, target):
        pw       = self.pw.to(logits.device)
        bce      = nn.BCEWithLogitsLoss(pos_weight=pw)(logits, target)
        prob     = torch.sigmoid(logits)
        dice     = 1.0 - (2.0*(prob*target).sum()) / (prob.sum()+target.sum()+1e-6)
        return (bce + dice) / 2.0