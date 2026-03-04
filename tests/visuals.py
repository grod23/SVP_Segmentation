from src.config import IMAGE_KEY, MASK_KEY
import matplotlib.pyplot as plt
from monai.visualize.utils import blend_images

# class Visualizer:
#     def __init__(self, logger):
#         self.logger = logger
#
#
#     def display_training_loss(self):
#         plt.figure(figsize=(10, 10))
#         plt.plot(self.logger.training_loss_logs, c='b', label='Training Loss')
#         plt.plot(self.logger.validation_loss_logs, c='r', label='Validation Loss')
#         plt.legend()
#         plt.grid()
#         plt.xlabel('Epochs', fontsize=20)
#         plt.ylabel('Loss', fontsize=20)
#         plt.show()
#
#     def prep_sample(self, sample):
#         sample = sample.permute(1, 2, 0)
#         sample = sample.detach().cpu().numpy()
#         return sample
#
#     def plot_single_sample(self, sample):
#         # print(f'Sample Shape: {sample.shape}')
#         sample_prepped = self.prep_sample(sample)
#         plt.figure(figsize=(10, 10))
#         plt.title("Image")
#         plt.imshow(sample_prepped)
#         plt.axis("off")
#         plt.show()
#
#     def visualize_batch(self, batch):
#         image_batch = batch[IMAGE_KEY]
#         mask_batch = batch[MASK_KEY]
#         print(f'Image Shape: {image_batch.shape}')
#         print(f'Image Dtype: {image_batch.dtype}')
#         print(f'Mask Shape: {mask_batch.shape}')
#         print(f'Mask Dtype: {mask_batch.dtype}')
#
#         # Get first items in batch
#         image = image_batch[0]
#         mask = mask_batch[0]
#
#         # stretch contrast for display
#         image = (image - image.min()) / (image.max() - image.min() + 1e-8)
#
#         print("Image min/max:", image.min(), image.max())
#         print("Mask min/max:", mask.min(), mask.max())
#
#         # Convert mask to grayscale
#         mask = mask.mean(dim=0, keepdim=True)  # [1, H, W]
#         # Overlay Image
#         blended = blend_images(
#             image=image,
#             label=mask,
#             alpha=0.4
#         )
#
#         # Convert each image from [C, H, W] to [H, W, C] and to numpy
#         mask = self.prep_sample(mask)
#         image = self.prep_sample(image)
#         blended = self.prep_sample(blended)
#
#         # Plot image, mask, and overlay
#         # plot_2d_or_3d_image(
#         #     data=[image, mask, blended],
#         #     step=0,
#         #     writer=self.writer
#         # )
#
#         # Plot Image, Mask, Overlay
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 3, 1)
#         plt.title("Image")
#         plt.imshow(image)
#         plt.axis("off")
#
#         plt.subplot(1, 3, 2)
#         plt.title("Mask")
#         plt.imshow(mask.squeeze(-1), cmap="gray")
#         plt.axis("off")
#
#         plt.subplot(1, 3, 3)
#         plt.title("Blended")
#         plt.imshow(blended)
#         plt.axis("off")
#         plt.show()
#
#     def plot_image_results(self, original_image, mask, predicted_mask):
#         # Prep each image
#         mask = self.prep_sample(mask)
#         predicted_mask = self.prep_sample(predicted_mask)
#         plt.figure(figsize=(10, 10))
#         plt.subplot(1, 3, 1)
#         plt.imshow(original_image.permute(1, 2, 0).detach().cpu().numpy() / 255)
#         plt.title('Original Image')
#         plt.axis("off")
#
#         plt.subplot(1, 3, 2)
#         plt.imshow(mask)
#         plt.title('Ground Truth')
#         plt.axis("off")
#
#         plt.subplot(1, 3, 3)
#         plt.imshow(predicted_mask)
#         plt.title('Predicted Mask')
#         plt.axis("off")
#         plt.show()
#
#
#     def plot_pulsation_masks(self, mask_max, mask_min, predicted_max, predicted_min):
#         """
#             Plots the original image along with ground truth and predicted masks (min and max),
#             without creating pulsation masks yet.
#         """
#         # Prepare each image/mask
#         mask_max = self.prep_sample(mask_max)
#         mask_min = self.prep_sample(mask_min)
#         predicted_max = self.prep_sample(predicted_max)
#         predicted_min = self.prep_sample(predicted_min)
#
#         plt.figure(figsize=(20, 5))  # Wide figure for 5 images side by side
#         # Ground truth min
#         plt.subplot(1, 4, 1)
#         plt.imshow(mask_min, cmap='gray')
#         plt.title('GT Min Mask')
#         plt.axis("off")
#
#         # Ground truth max
#         plt.subplot(1, 4, 2)
#         plt.imshow(mask_max, cmap='gray')
#         plt.title('GT Max Mask')
#         plt.axis("off")
#
#         # Predicted min
#         plt.subplot(1, 4, 3)
#         plt.imshow(predicted_min, cmap='gray')
#         plt.title('Pred Min Mask')
#         plt.axis("off")
#
#         # Predicted max
#         plt.subplot(1, 4, 4)
#         plt.imshow(predicted_max, cmap='gray')
#         plt.title('Pred Max Mask')
#         plt.axis("off")
#
#         plt.tight_layout()
#         plt.show()



from src.config import IMAGE_KEY, MASK_KEY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from monai.visualize.utils import blend_images
import numpy as np
from PIL import Image
import io
import os


class Visualizer:
    def __init__(self, logger):
        self.logger = logger

    # ─────────────────────────────────────────────────────────────────
    # Existing helpers (unchanged)
    # ─────────────────────────────────────────────────────────────────

    def display_training_loss(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.logger.training_loss_logs, c='b', label='Training Loss')
        plt.plot(self.logger.validation_loss_logs, c='r', label='Validation Loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.show()

    def prep_sample(self, sample):
        sample = sample.permute(1, 2, 0)
        sample = sample.detach().cpu().numpy()
        return sample

    def plot_single_sample(self, sample):
        sample_prepped = self.prep_sample(sample)
        plt.figure(figsize=(10, 10))
        plt.title("Image")
        plt.imshow(sample_prepped)
        plt.axis("off")
        plt.show()

    def visualize_batch(self, batch):
        image_batch = batch[IMAGE_KEY]
        mask_batch  = batch[MASK_KEY]
        print(f'Image Shape: {image_batch.shape}')
        print(f'Image Dtype: {image_batch.dtype}')
        print(f'Mask Shape:  {mask_batch.shape}')
        print(f'Mask Dtype:  {mask_batch.dtype}')

        image = image_batch[0]
        mask  = mask_batch[0]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask  = mask.mean(dim=0, keepdim=True)

        blended = blend_images(image=image, label=mask, alpha=0.4)

        mask    = self.prep_sample(mask)
        image   = self.prep_sample(image)
        blended = self.prep_sample(blended)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("Image");   plt.imshow(image);                  plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("Mask");    plt.imshow(mask.squeeze(-1), cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Blended"); plt.imshow(blended);                plt.axis("off")
        plt.show()

    def plot_image_results(self, original_image, mask, predicted_mask):
        mask           = self.prep_sample(mask)
        predicted_mask = self.prep_sample(predicted_mask)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1); plt.imshow(original_image.permute(1, 2, 0).detach().cpu().numpy() / 255); plt.title('Original Image'); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(mask);           plt.title('Ground Truth');   plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(predicted_mask); plt.title('Predicted Mask'); plt.axis("off")
        plt.show()

    # ─────────────────────────────────────────────────────────────────
    # Updated: static 4-panel comparison + pulsation overlay column
    # ─────────────────────────────────────────────────────────────────

    def plot_pulsation_masks(self, mask_max, mask_min, predicted_max, predicted_min,
                             save_path: str = None):
        """
        6-panel figure:
          GT Min | GT Max | GT Pulsation overlay
          Pred Min | Pred Max | Pred Pulsation overlay

        Pulsation overlay colours (single-channel binary retinal vessel):
          Cyan   – vessel present in BOTH frames (stable core)
          Green  – vessel only at peak/max  (dilation Δ)
          Red    – vessel only at trough/min (contraction Δ)
          Black  – background

        Args:
            mask_max / mask_min:         GT masks,        tensors [1, H, W]
            predicted_max / predicted_min: Pred masks,    tensors [1, H, W]
            save_path: optional filepath (.png) to save the figure
        """
        # To numpy [H, W] bool
        def to_np(t):
            return t.squeeze().detach().cpu().numpy().astype(bool)

        gt_min  = to_np(mask_min)
        gt_max  = to_np(mask_max)
        pr_min  = to_np(predicted_min)
        pr_max  = to_np(predicted_max)

        gt_overlay   = self._build_pulsation_overlay(gt_min,  gt_max)
        pred_overlay = self._build_pulsation_overlay(pr_min,  pr_max)

        # Legend patches
        legend_patches = [
            mpatches.Patch(color='cyan',  label='Stable core (both)'),
            mpatches.Patch(color='lime',  label='Dilation only (peak)'),
            mpatches.Patch(color='red',   label='Contraction only (trough)'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor('#0d0d0d')

        panels = [
            (axes[0, 0], gt_min,       'GT Trough',             'gray'),
            (axes[0, 1], gt_max,       'GT Peak',               'gray'),
            (axes[0, 2], gt_overlay,   'GT Pulsation Overlay',  None),
            (axes[1, 0], pr_min,       'Pred Trough',           'gray'),
            (axes[1, 1], pr_max,       'Pred Peak',             'gray'),
            (axes[1, 2], pred_overlay, 'Pred Pulsation Overlay',None),
        ]

        for ax, img, title, cmap in panels:
            ax.imshow(img, cmap=cmap, interpolation='nearest')
            ax.set_title(title, color='white', fontsize=11, pad=6)
            ax.axis('off')
            ax.set_facecolor('#0d0d0d')

        axes[0, 2].legend(handles=legend_patches, loc='lower right',
                          fontsize=7, framealpha=0.6, facecolor='#1a1a1a',
                          labelcolor='white')
        axes[1, 2].legend(handles=legend_patches, loc='lower right',
                          fontsize=7, framealpha=0.6, facecolor='#1a1a1a',
                          labelcolor='white')

        plt.tight_layout(pad=1.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Saved static figure → {save_path}")

        plt.show()
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────
    # create_pulsation_mask  (also called from test.py)
    # ─────────────────────────────────────────────────────────────────

    def create_pulsation_mask(self, trough_mask, peak_mask,
                              save_path: str = "pulsation.gif",
                              n_frames: int = 30,
                              fps: int = 15,
                              amplify: float = 1.0):
        """
        Build a looping GIF that pulsates between trough_mask and peak_mask.

        Designed for subtle single-channel binary retinal vessel masks where
        the diameter change may be only a few pixels.

        Three-zone colour encoding per frame:
          Cyan   – vessel in BOTH masks (stable core)   always visible
          Green  – peak-only pixels (dilation zone)     fades in/out
          Red    – trough-only pixels (contraction zone) fades in/out

        Args:
            trough_mask : tensor [1, H, W] or [H, W]  – diastole / min frame
            peak_mask   : tensor [1, H, W] or [H, W]  – systole  / max frame
            save_path   : filepath for the output GIF
            n_frames    : frames per half-cycle  (total = n_frames * 2)
            fps         : playback speed
            amplify     : >1.0 visually widens the difference zones for clarity
                          (morphological dilation on diff regions before blending)

        Returns:
            save_path (str)
        """
        import torch
        from scipy.ndimage import binary_dilation as morph_dilate

        def to_bool(t):
            return t.squeeze().detach().cpu().numpy().astype(bool)

        trough = to_bool(trough_mask)
        peak   = to_bool(peak_mask)

        # ── Three semantic zones ──────────────────────────────────────
        core        = trough & peak          # stable vessel core
        dilation    = peak   & ~trough       # appears at peak
        contraction = trough & ~peak         # disappears at peak

        # Optional: morphologically amplify the diff zones so they're visible
        # even when only 1-2 pixels wide (common in retinal vessel pulsation)
        if amplify > 1.0:
            iters = max(1, int(amplify))
            dilation    = morph_dilate(dilation,    iterations=iters) & ~core
            contraction = morph_dilate(contraction, iterations=iters) & ~core

        # ── Ease function ─────────────────────────────────────────────
        def ease(t):
            """Smooth cubic ease-in-out [0→1]."""
            return t * t * (3 - 2 * t)

        # ── Colour palette (RGBA float) ───────────────────────────────
        BLACK      = np.array([0.04, 0.04, 0.04, 1.0])
        CORE_COL   = np.array([0.00, 0.85, 0.95, 1.0])   # cyan
        DILAT_COL  = np.array([0.10, 0.95, 0.30, 1.0])   # green
        CONTR_COL  = np.array([0.95, 0.20, 0.15, 1.0])   # red
        BORDER_COL = np.array([1.00, 1.00, 1.00, 1.0])   # white glow

        H, W = trough.shape

        def render_frame(alpha: float) -> np.ndarray:
            """
            alpha : 0.0 = trough state, 1.0 = peak state
            Returns RGB uint8 [H, W, 3]
            """
            canvas = np.zeros((H, W, 4), dtype=float)
            canvas[:] = BLACK

            # Core always fully lit, slight luminance pulse
            lum = 0.80 + 0.20 * ease(alpha)
            canvas[core] = CORE_COL * lum

            # Dilation zone fades IN as alpha→1
            a_dil = ease(alpha)
            canvas[dilation] = (DILAT_COL * a_dil +
                                BLACK      * (1 - a_dil))

            # Contraction zone fades OUT as alpha→1
            a_con = ease(1 - alpha)
            canvas[contraction] = (CONTR_COL * a_con +
                                   BLACK      * (1 - a_con))

            # Thin white glow on outer border of active vessel
            active = canvas[..., 0] > BLACK[0] + 0.05
            border = morph_dilate(active, iterations=1) & ~active
            glow   = 0.35 + 0.65 * np.sin(alpha * np.pi)
            border_rgba        = BORDER_COL.copy()
            border_rgba[3]     = glow
            canvas[border]     = border_rgba

            rgb = (np.clip(canvas[..., :3], 0, 1) * 255).astype(np.uint8)
            return rgb

        # ── Build frame sequence  (trough→peak→trough = one full cycle) ──
        alphas = (list(np.linspace(0, 1, n_frames)) +
                  list(np.linspace(1, 0, n_frames)))

        pil_frames = []
        for a in alphas:
            rgb = render_frame(a)
            pil_frames.append(Image.fromarray(rgb))

        # ── Save GIF ─────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=int(1000 / fps),
            optimize=False,
        )
        print(f"  Pulsation GIF saved → {save_path}")

        # ── Print a brief zone summary ────────────────────────────────
        total = H * W
        print(f"  Zone areas | Core: {core.sum()} px "
              f"| Dilation: {dilation.sum()} px "
              f"| Contraction: {contraction.sum()} px "
              f"| ({dilation.sum() + contraction.sum()}/{core.sum() or 1:.1%} of core)")

        return save_path

    # ─────────────────────────────────────────────────────────────────
    # Internal helper
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_pulsation_overlay(trough: np.ndarray, peak: np.ndarray) -> np.ndarray:
        """
        Build an RGB overlay image from two boolean masks.
          Cyan  = core (both)
          Green = dilation (peak only)
          Red   = contraction (trough only)
          Black = background
        """
        H, W   = trough.shape
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        core        = trough & peak
        dilation    = peak   & ~trough
        contraction = trough & ~peak

        canvas[core]        = [0,   210, 240]   # cyan
        canvas[dilation]    = [30,  240,  80]   # green
        canvas[contraction] = [240,  50,  40]   # red

        return canvas