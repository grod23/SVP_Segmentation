from src.config import IMAGE_KEY, MASK_KEY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from monai.visualize.utils import blend_images
import numpy as np
from PIL import Image
import os


class Visualizer:
    def __init__(self, logger):
        self.logger = logger

    # ─────────────────────────────────────────────────────────────────
    # General helpers
    # ─────────────────────────────────────────────────────────────────

    def display_training_loss(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.logger.training_loss_logs,   c='b', label='Training Loss')
        plt.plot(self.logger.validation_loss_logs, c='r', label='Validation Loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss',   fontsize=20)

    def prep_sample(self, sample):
        """[C, H, W] tensor -> [H, W, C] numpy float."""
        return sample.permute(1, 2, 0).detach().cpu().numpy()

    def plot_single_sample(self, sample):
        plt.figure(figsize=(10, 10))
        plt.title("Image")
        plt.imshow(self.prep_sample(sample))
        plt.axis("off")

    def visualize_batch(self, batch):
        image_batch = batch[IMAGE_KEY]
        mask_batch  = batch[MASK_KEY]
        image = image_batch[0]
        mask  = mask_batch[0]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask  = mask.mean(dim=0, keepdim=True)
        blended = blend_images(image=image, label=mask, alpha=0.4)
        mask    = self.prep_sample(mask)
        image   = self.prep_sample(image)
        blended = self.prep_sample(blended)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("Image");   plt.imshow(image);                         plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("Mask");    plt.imshow(mask.squeeze(-1), cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Blended"); plt.imshow(blended);                       plt.axis("off")

    def plot_image_results(self, original_image, mask, predicted_mask):
        mask           = self.prep_sample(mask)
        predicted_mask = self.prep_sample(predicted_mask)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1); plt.imshow(original_image.permute(1, 2, 0).detach().cpu().numpy() / 255); plt.title('Original Image'); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(mask);           plt.title('Ground Truth');   plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(predicted_mask); plt.title('Predicted Mask'); plt.axis("off")

    # ─────────────────────────────────────────────────────────────────
    # Pulsation: static 6-panel comparison figure
    # ─────────────────────────────────────────────────────────────────


    def plot_gt_pulsation_figure(self, img_trough, img_peak,
                                  mask_trough, mask_peak,
                                  save_path: str = None,
                                  title_fontsize: int = 16):
        """
        Clean presentation figure for ground truth pulsation.

        Layout — 2 rows × 3 columns:

          Row 0 (Trough / Diastole):
            Col 0: Original retinal image at trough
            Col 1: Ground truth vessel mask at trough
            Col 2: (spans both rows) Pulsation overlay

          Row 1 (Peak / Systole):
            Col 0: Original retinal image at peak
            Col 1: Ground truth vessel mask at peak

        Overlay colour key (white background):
          Dark gray — Stable vessel core (present in both frames)
          Blue      — Vessel dilation (expanded at systole/peak)
          Red       — Vessel contraction (narrowed at systole/peak)

        Args:
            img_trough / img_peak   : [C, H, W] tensors — original images
            mask_trough / mask_peak : [1, H, W] tensors — GT binary masks
            save_path               : optional .png filepath
            title_fontsize          : font size for panel titles
        """
        import matplotlib.gridspec as gridspec

        def to_np(t):
            arr = t.squeeze().detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr.transpose(1, 2, 0)
            return arr

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        def to_bool(t):
            return to_np(t) > 0.5

        img_tr  = norm(to_np(img_trough))
        img_pk  = norm(to_np(img_peak))
        msk_tr  = to_bool(mask_trough)
        msk_pk  = to_bool(mask_peak)

        overlay = self._build_pulsation_overlay(msk_tr, msk_pk)

        # Stats for annotation
        core_px = int((msk_tr & msk_pk).sum())
        dil_px  = int((msk_pk & ~msk_tr).sum())
        con_px  = int((msk_tr & ~msk_pk).sum())

        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs  = gridspec.GridSpec(
            2, 3,
            figure=fig,
            width_ratios=[1, 1, 1.05],
            hspace=0.18,
            wspace=0.06,
        )

        ax_tr_img  = fig.add_subplot(gs[0, 0])
        ax_pk_img  = fig.add_subplot(gs[1, 0])
        ax_tr_mask = fig.add_subplot(gs[0, 1])
        ax_pk_mask = fig.add_subplot(gs[1, 1])
        ax_overlay = fig.add_subplot(gs[:, 2])   # spans both rows

        # ── Column 0: Original images ─────────────────────────────────
        ax_tr_img.imshow(img_tr, cmap='gray')
        ax_tr_img.set_title('Trough Frame(minimum diameter)',
                             fontsize=title_fontsize, fontweight='bold', pad=20)

        ax_pk_img.imshow(img_pk, cmap='gray')
        ax_pk_img.set_title('Peak Frame(maximum diameter)',
                             fontsize=title_fontsize, fontweight='bold', pad=5)

        # ── Column 1: Masks ───────────────────────────────────────────
        ax_tr_mask.imshow(msk_tr, cmap='gray', vmin=0, vmax=1)
        ax_tr_mask.set_title('',
                              fontsize=title_fontsize, fontweight='bold', pad=10)

        ax_pk_mask.imshow(msk_pk, cmap='gray', vmin=0, vmax=1)
        ax_pk_mask.set_title('',
                              fontsize=title_fontsize, fontweight='bold', pad=10)

        # ── Column 2: Pulsation overlay ───────────────────────────────
        ax_overlay.imshow(overlay, interpolation='nearest')
        ax_overlay.set_title(
            'Vessel Pulsation Map',
            fontsize=title_fontsize + 2, fontweight='bold', pad=14
        )

        # Annotation with pixel counts
        annotation = (
            f"Stable core:   {core_px:,} px\n"
            f"Dilation:        {dil_px:,} px\n"
            f"Contraction:  {con_px:,} px"
        )
        ax_overlay.text(
            0.03, 0.03, annotation,
            transform=ax_overlay.transAxes,
            fontsize=title_fontsize - 3,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9),
            family='monospace',
        )

        # Legend
        legend_patches = [
            mpatches.Patch(facecolor='#646464', edgecolor='#999',
                           label='Stable core (present in both frames)'),

            mpatches.Patch(facecolor='#ff8c00', edgecolor='#999',
                           label='Dilation (vessel expansion)'),

            mpatches.Patch(facecolor='#0078ff', edgecolor='#999',
                           label='Contraction (vessel narrowing)'),
        ]

        ax_overlay.legend(
            handles=legend_patches,
            loc='upper right',
            bbox_to_anchor=(1.0, -0.02),
            fontsize=title_fontsize - 3,
            framealpha=0.95,
            edgecolor='#cccccc',
            labelspacing=0.8,
            handlelength=1.5,
        )

        # Clean up axes
        for ax in [ax_tr_img, ax_pk_img, ax_tr_mask, ax_pk_mask, ax_overlay]:
            ax.axis('off')
            ax.set_facecolor('white')

        # Row labels on left margin
        fig.text(0.01, 0.73, 'TROUGH', fontsize=title_fontsize - 1,
                 fontweight='bold', va='center', rotation=90, color='#333333')
        fig.text(0.01, 0.27, 'PEAK', fontsize=title_fontsize - 1,
                 fontweight='bold', va='center', rotation=90, color='#333333')

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight',
                        facecolor='white')
            print(f"  Saved GT pulsation figure -> {save_path}")

        plt.show()
        plt.close(fig)

    def plot_pulsation_masks(self, mask_max, mask_min, predicted_max, predicted_min,
                             img_trough=None, img_peak=None,
                             save_path: str = None):
        """
        Layout — 3 rows × 4 columns:

          Row 0 (Trough): Original Image | GT Mask | Prediction | Pulsation Overlay
          Row 1 (Peak)  : Original Image | GT Mask | Prediction | Pulsation Overlay
          Row 2 (Diff)  : GT Overlay (trough vs peak) | Pred Overlay | [spacer x2]

        Overlay colour key:
          White  - stable core (vessel in both trough and peak)
          Green  - dilation  (peak only — vessel grew)
          Red    - contraction (trough only — vessel shrunk)
          Black  - background
        """
        def to_np_img(t):
            """Tensor [1,H,W] or [3,H,W] -> numpy [H,W] or [H,W,3] for display."""
            arr = t.squeeze().detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
            return arr

        def to_bool(t):
            return to_np_img(t) > 0.5

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        gt_tr   = to_bool(mask_min)
        gt_pk   = to_bool(mask_max)
        pr_tr   = to_bool(predicted_min)
        pr_pk   = to_bool(predicted_max)

        gt_overlay   = self._build_pulsation_overlay(gt_tr, gt_pk)
        pred_overlay = self._build_pulsation_overlay(pr_tr, pr_pk)

        legend_patches = [
            mpatches.Patch(color='white', label='Stable core'),
            mpatches.Patch(color='lime',  label='Dilation (peak only)'),
            mpatches.Patch(color='red',   label='Contraction (trough only)'),
        ]

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.patch.set_facecolor('#0d0d0d')

        # ── Row 0: Trough ─────────────────────────────────────────────
        if img_trough is not None:
            axes[0, 0].imshow(norm(to_np_img(img_trough)), cmap='gray')
            axes[0, 0].set_title('Trough — Original', color='white', fontsize=11, pad=6)
        else:
            axes[0, 0].set_visible(False)

        axes[0, 1].imshow(gt_tr,  cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Trough — GT Mask',       color='white', fontsize=11, pad=6)
        axes[0, 2].imshow(pr_tr,  cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('Trough — Prediction',    color='white', fontsize=11, pad=6)
        axes[0, 3].set_visible(False)   # spacer — overlay spans row 2

        # ── Row 1: Peak ───────────────────────────────────────────────
        if img_peak is not None:
            axes[1, 0].imshow(norm(to_np_img(img_peak)), cmap='gray')
            axes[1, 0].set_title('Peak — Original',    color='white', fontsize=11, pad=6)
        else:
            axes[1, 0].set_visible(False)

        axes[1, 1].imshow(gt_pk,  cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Peak — GT Mask',         color='white', fontsize=11, pad=6)
        axes[1, 2].imshow(pr_pk,  cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('Peak — Prediction',      color='white', fontsize=11, pad=6)
        axes[1, 3].set_visible(False)   # spacer

        # ── Row 2: Pulsation overlays ─────────────────────────────────
        axes[2, 0].imshow(gt_overlay)
        axes[2, 0].set_title('GT Pulsation Overlay',   color='white', fontsize=11, pad=6)
        axes[2, 0].legend(handles=legend_patches, loc='lower right',
                          fontsize=7, framealpha=0.6,
                          facecolor='#1a1a1a', labelcolor='white')

        axes[2, 1].imshow(pred_overlay)
        axes[2, 1].set_title('Pred Pulsation Overlay', color='white', fontsize=11, pad=6)
        axes[2, 1].legend(handles=legend_patches, loc='lower right',
                          fontsize=7, framealpha=0.6,
                          facecolor='#1a1a1a', labelcolor='white')

        axes[2, 2].set_visible(False)
        axes[2, 3].set_visible(False)

        # ── Row labels ────────────────────────────────────────────────
        for row, label in enumerate(['Trough', 'Peak', 'Pulsation']):
            axes[row, 0].set_ylabel(label, color='white', fontsize=12,
                                    rotation=90, labelpad=10)

        for ax in axes.flat:
            if ax.get_visible():
                ax.axis('off')
                ax.set_facecolor('#0d0d0d')

        # Re-enable row labels (axis('off') hides them)
        for row in range(3):
            axes[row, 0].yaxis.set_visible(True)

        plt.tight_layout(pad=1.5)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Saved static figure -> {save_path}")
        plt.show()
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────
    # Pulsation: animated GIF
    # ─────────────────────────────────────────────────────────────────

    def create_pulsation_mask(self, trough_mask, peak_mask,
                              save_path: str = "pulsation.gif",
                              n_frames: int = 40,
                              fps: int = 12,
                              amplify: float = 1.0):
        """
        Build and save a looping pulsation GIF: trough -> peak -> trough.

        FIX: GIF animates correctly by ensuring frames differ visually and
             using convert('P') palette mode which PIL GIF encoder handles
             reliably.

        FIX: Colour scheme corrected —
          White  - stable vessel core
          Green  - dilation zone (vessel only at peak)   fades in
          Red    - contraction zone (vessel only at trough) fades out
          Black  - background

        FIX: trough is shown FIRST (alpha=0 = trough state).
        """
        from scipy.ndimage import binary_dilation as morph_dilate

        def to_bool(t):
            arr = t.squeeze().detach().cpu().numpy()
            return arr > 0.5

        trough = to_bool(trough_mask)
        peak   = to_bool(peak_mask)

        # ── Semantic zones ────────────────────────────────────────────
        core        = trough & peak       # stable vessel in both
        dilation    = peak   & ~trough    # vessel gained at peak
        contraction = trough & ~peak      # vessel lost at peak

        if amplify > 1.0:
            iters = max(1, int(amplify))
            dilation    = morph_dilate(dilation,    iterations=iters) & ~core
            contraction = morph_dilate(contraction, iterations=iters) & ~core

        H, W = trough.shape

        # ── Colour palette RGB uint8 ──────────────────────────────────
        # Using explicit uint8 arrays avoids any float rounding issues
        # that can make PIL think frames are identical.
        BG_R, BG_G, BG_B         = 255, 255, 255  # white background
        CORE_R, CORE_G, CORE_B   = 80,  80,  80   # dark gray — stable core
        DIL_R,  DIL_G,  DIL_B    = 30,  80,  200  # blue  — dilation
        CON_R,  CON_G,  CON_B    = 200, 30,  30   # red   — contraction
        GLO_R,  GLO_G,  GLO_B    = 40,  40,  40   # dark border glow

        def ease(t):
            return t * t * (3 - 2 * t)

        def lerp_val(fg, bg, alpha):
            """Scalar -> scalar interpolation, returns a single uint8 value."""
            return np.uint8(np.clip(round(fg * alpha + bg * (1 - alpha)), 0, 255))

        def render_frame(alpha: float) -> np.ndarray:
            """
            alpha=0.0  ->  trough state (shown first)
            alpha=1.0  ->  peak state
            Returns RGB uint8 [H, W, 3].
            """
            R = np.full((H, W), BG_R, dtype=np.uint8)
            G = np.full((H, W), BG_G, dtype=np.uint8)
            B = np.full((H, W), BG_B, dtype=np.uint8)

            # Core: white, brightness pulses slightly with ease curve
            lum = 0.75 + 0.25 * ease(alpha)
            R[core] = np.uint8(CORE_R * lum)
            G[core] = np.uint8(CORE_G * lum)
            B[core] = np.uint8(CORE_B * lum)

            # Dilation: green fades IN as alpha -> 1
            a_dil = ease(alpha)
            R[dilation] = lerp_val(DIL_R, BG_R, a_dil)
            G[dilation] = lerp_val(DIL_G, BG_G, a_dil)
            B[dilation] = lerp_val(DIL_B, BG_B, a_dil)

            # Contraction: red fades OUT as alpha -> 1
            a_con = ease(1.0 - alpha)
            R[contraction] = lerp_val(CON_R, BG_R, a_con)
            G[contraction] = lerp_val(CON_G, BG_G, a_con)
            B[contraction] = lerp_val(CON_B, BG_B, a_con)

            # Subtle dark border glow around entire active vessel
            active = (R < BG_R - 10) | (G < BG_G - 10) | (B < BG_B - 10)
            border = morph_dilate(active, iterations=1) & ~active
            glow   = 0.25 + 0.75 * np.sin(alpha * np.pi)  # peaks mid-cycle
            R[border] = np.uint8(GLO_R * glow)
            G[border] = np.uint8(GLO_G * glow)
            B[border] = np.uint8(GLO_B * glow)

            return np.stack([R, G, B], axis=-1)

        # ── Build frames: trough(0) -> peak(1) -> trough(0) ──────────
        # Add a brief hold at trough and peak so the eye can register each state
        hold = max(2, n_frames // 8)
        alphas = (
            [0.0] * hold +
            list(np.linspace(0, 1, n_frames)) +
            [1.0] * hold +
            list(np.linspace(1, 0, n_frames))
        )

        frames = []
        for a in alphas:
            rgb = render_frame(a)
            # Convert to palette mode — PIL GIF encoder is most reliable with 'P'
            pil = Image.fromarray(rgb, mode='RGB').convert('P', dither=Image.NONE)
            frames.append(pil)

        # ── Save GIF ──────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        frame_duration_ms = int(1000 / fps)
        frames[0].save(
            save_path,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            loop=0,                       # loop forever
            duration=frame_duration_ms,
            optimize=False,               # must be False — optimizer can strip frames
            disposal=2,                   # clear each frame before drawing next
        )

        # ── Zone summary ──────────────────────────────────────────────
        core_px = int(core.sum())
        dil_px  = int(dilation.sum())
        con_px  = int(contraction.sum())
        diff_px = dil_px + con_px
        pct     = (diff_px / core_px * 100) if core_px > 0 else 0.0
        print(f"  Pulsation GIF saved -> {save_path}  ({len(frames)} frames @ {fps} fps)")
        print(f"  Zone areas | Core: {core_px:,} px "
              f"| Dilation: {dil_px:,} px "
              f"| Contraction: {con_px:,} px "
              f"| Delta: {diff_px:,} px ({pct:.2f}% of core)")

        return save_path


    def plot_raw_vs_thresholded(self, mask_trough, mask_peak,
                                raw_trough, raw_peak,
                                pred_trough, pred_peak,
                                save_path: str = None,
):
        """
        3-row comparison figure for one sample:

          Row 0 — Ground Truth         : Trough mask | Peak mask | Pulsation overlay
          Row 1 — Raw probability map  : Trough prob | Peak prob | Prob-diff map
          Row 2 — Thresholded (>0.5)   : Trough bin  | Peak bin  | Pulsation overlay

        Row 1 uses a heatmap (hot colormap) so you can see where the model is
        uncertain (prob ~0.5, dark) vs confident foreground (prob ~1.0, bright).
        The prob-diff map shows signed(peak-trough) so you can see which pixels
        actually shifted between the two frames before any thresholding.

        Args:
            mask_trough / mask_peak   : GT binary masks,         tensor [1, H, W]
            raw_trough  / raw_peak    : raw sigmoid probs (0-1), tensor [1, H, W]
                                        Pass sigmoid(logits) — NOT the logits.
            pred_trough / pred_peak   : thresholded binary preds, tensor [1, H, W]
            save_path                 : optional .png filepath
        """
        def to_np(t):
            return t.squeeze().detach().cpu().numpy()

        def to_bool(t):
            return to_np(t) > 0.5

        # ── GT ───────────────────────────────────────────────────────
        gt_tr   = to_bool(mask_trough)
        gt_pk   = to_bool(mask_peak)
        gt_ov   = self._build_pulsation_overlay(gt_tr, gt_pk)

        # ── Raw probs (float 0-1) ─────────────────────────────────────
        prob_tr = to_np(raw_trough)    # [H, W]  float
        prob_pk = to_np(raw_peak)      # [H, W]  float

        # Normalise each map to [0,1] so subtle differences are visible
        def norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)

        prob_tr_n = norm(prob_tr)
        prob_pk_n = norm(prob_pk)

        # Signed diff: +ve = grew at peak, -ve = shrunk at peak
        prob_diff = prob_pk_n - prob_tr_n   # range roughly [-1, 1]

        # ── Thresholded preds ─────────────────────────────────────────
        pr_tr = to_bool(pred_trough)
        pr_pk = to_bool(pred_peak)
        pr_ov = self._build_pulsation_overlay(pr_tr, pr_pk)

        # ── Stats printout ────────────────────────────────────────────
        gt_diff_px   = int((gt_pk != gt_tr).sum())
        pred_diff_px = int((pr_pk != pr_tr).sum())
        prob_diff_px = int((np.abs(prob_diff) > 0.05).sum())
        print(f"  GT   diff px           : {gt_diff_px:,}")
        print(f"  Prob diff px (|d|>0.05): {prob_diff_px:,}")
        print(f"  Pred diff px (thresh)  : {pred_diff_px:,}")
        print(f"  Raw prob trough — min: {prob_tr.min():.3f}  max: {prob_tr.max():.3f}  "
              f"mean: {prob_tr.mean():.3f}")
        print(f"  Raw prob peak   — min: {prob_pk.min():.3f}  max: {prob_pk.max():.3f}  "
              f"mean: {prob_pk.mean():.3f}")

        # ── Figure ────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.patch.set_facecolor('#0d0d0d')

        # Legend for binary overlay rows
        bin_legend = [
            mpatches.Patch(color='white', label='Stable core'),
            mpatches.Patch(color='lime',  label='Dilation (peak only)'),
            mpatches.Patch(color='red',   label='Contraction (trough only)'),
        ]

        # ── Row 0: GT ─────────────────────────────────────────────────
        axes[0, 0].imshow(gt_tr,  cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('GT Trough',           color='white', fontsize=10, pad=5)
        axes[0, 1].imshow(gt_pk,  cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('GT Peak',             color='white', fontsize=10, pad=5)
        axes[0, 2].imshow(gt_ov)
        axes[0, 2].set_title('GT Pulsation Overlay',color='white', fontsize=10, pad=5)
        axes[0, 2].legend(handles=bin_legend, loc='lower right',
                          fontsize=6, framealpha=0.6, facecolor='#1a1a1a', labelcolor='white')

        # ── Row 1: Raw prob maps ──────────────────────────────────────
        im_tr = axes[1, 0].imshow(prob_tr_n, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Raw Prob Trough(normalised, hot=high conf)',
                              color='white', fontsize=10, pad=5)
        plt.colorbar(im_tr, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im_pk = axes[1, 1].imshow(prob_pk_n, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Raw Prob Peak(normalised, hot=high conf)',
                              color='white', fontsize=10, pad=5)
        plt.colorbar(im_pk, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Signed diff: use a diverging colormap — red=grew, blue=shrunk
        im_diff = axes[1, 2].imshow(prob_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[1, 2].set_title('Prob Diff (peak - trough)Red=grew  Blue=shrunk',
                              color='white', fontsize=10, pad=5)
        plt.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # ── Row 2: Thresholded preds ──────────────────────────────────
        axes[2, 0].imshow(pr_tr, cmap='gray', vmin=0, vmax=1)
        axes[2, 0].set_title('Pred Trough (thresh=0.30)', color='white', fontsize=10, pad=5)
        axes[2, 1].imshow(pr_pk, cmap='gray', vmin=0, vmax=1)
        axes[2, 1].set_title('Pred Peak (thresh=0.30)',   color='white', fontsize=10, pad=5)
        axes[2, 2].imshow(pr_ov)
        axes[2, 2].set_title('Pred Pulsation Overlay',   color='white', fontsize=10, pad=5)
        axes[2, 2].legend(handles=bin_legend, loc='lower right',
                          fontsize=6, framealpha=0.6, facecolor='#1a1a1a', labelcolor='white')

        # Row labels on left edge
        for row_idx, label in enumerate(['Ground Truth', 'Raw Probs', 'Thresholded']):
            axes[row_idx, 0].set_ylabel(label, color='white', fontsize=11,
                                        rotation=90, labelpad=8)

        for ax in axes.flat:
            ax.axis('off')
            ax.set_facecolor('#0d0d0d')
        # Re-enable ylabel after axis('off') strips it
        for row_idx in range(3):
            axes[row_idx, 0].set_ylabel(
                ['Ground Truth', 'Raw Probs', 'Thresholded'][row_idx],
                color='white', fontsize=11, rotation=90, labelpad=8
            )
            axes[row_idx, 0].yaxis.set_visible(True)

        plt.suptitle('Raw Probability vs Thresholded Prediction Comparison',
                     color='white', fontsize=13, y=1.01)
        plt.tight_layout(pad=1.5)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Saved comparison figure -> {save_path}")
        plt.show()
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────
    # Internal helper
    # ─────────────────────────────────────────────────────────────────



    def plot_diameter_change(self, img_trough, img_peak,
                             mask_trough, mask_peak,
                             diameter_change,
                             save_path: str = None):
        """
        Diameter change heatmap — shows WHERE and HOW MUCH the vessel
        changed between trough and peak frames.

        Layout — 2 rows x 3 cols:
          Row 0: Trough original | Peak original | GT pulsation overlay
          Row 1: Signed change map (RdBu) | Magnitude map (hot) | GT vs Pred diff

        diameter_change : [1, H, W] tensor — sigmoid(peak_logit) - sigmoid(trough_logit)
            Positive (red)  = vessel got larger at peak (dilation)
            Negative (blue) = vessel got smaller at peak (contraction)
        """
        def to_np(t):
            arr = t.squeeze().detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr.transpose(1, 2, 0)
            return arr

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        def to_bool(t):
            return to_np(t) > 0.5

        img_tr = norm(to_np(img_trough))
        img_pk = norm(to_np(img_peak))
        msk_tr = to_bool(mask_trough)
        msk_pk = to_bool(mask_peak)
        dc     = to_np(diameter_change)           # [H, W] signed float
        mag    = np.abs(dc)                       # magnitude

        gt_overlay = self._build_pulsation_overlay(msk_tr, msk_pk)

        # Pred overlay from the diameter change map
        dil_mask = dc >  0.05   # model thinks vessel grew
        con_mask = dc < -0.05   # model thinks vessel shrunk
        pred_overlay = np.zeros((*dc.shape, 3), dtype=np.uint8)
        pred_overlay[msk_tr & msk_pk]  = [220, 220, 220]   # stable core white
        pred_overlay[dil_mask]         = [30,  240,  80]   # green dilation
        pred_overlay[con_mask]         = [240,  50,  40]   # red contraction

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('#0d0d0d')

        # Row 0
        axes[0, 0].imshow(img_tr, cmap='gray')
        axes[0, 0].set_title('Trough — Original', color='white', fontsize=11, pad=5)

        axes[0, 1].imshow(img_pk, cmap='gray')
        axes[0, 1].set_title('Peak — Original', color='white', fontsize=11, pad=5)

        axes[0, 2].imshow(gt_overlay)
        axes[0, 2].set_title('GT Pulsation Overlay(white=stable, green=dilation, red=contraction)',
                              color='white', fontsize=10, pad=5)

        # Row 1
        im1 = axes[1, 0].imshow(dc, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[1, 0].set_title('Signed Diameter Change(red=dilation, blue=contraction)',
                              color='white', fontsize=10, pad=5)
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im2 = axes[1, 1].imshow(mag, cmap='hot', vmin=0, vmax=0.5)
        axes[1, 1].set_title('Change Magnitude(hot=large change)',
                              color='white', fontsize=10, pad=5)
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        axes[1, 2].imshow(pred_overlay)
        axes[1, 2].set_title('Pred Pulsation Overlay(from soft probability difference)',
                              color='white', fontsize=10, pad=5)

        for ax in axes.flat:
            ax.axis('off')
            ax.set_facecolor('#0d0d0d')

        plt.suptitle('Vessel Diameter Change Analysis',
                     color='white', fontsize=14, y=1.01)
        plt.tight_layout(pad=1.5)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Saved diameter change figure -> {save_path}")
        plt.show()
        plt.close(fig)

    def plot_pulsation_map(self, img_trough, img_peak,
                           mask_trough, mask_peak,
                           pulse_gt, pulse_pred_prob,
                           save_path: str = None):
        """
        Visualisation for Siamese single pulsation map output.

        Layout — 3 rows x 3 cols:
          Row 0: Trough original | Peak original  | GT pulsation map (binary)
          Row 1: Trough GT mask  | Peak GT mask   | Pred pulsation prob (heatmap)
          Row 2: GT overlay      | Pred overlay   | Prob diff colourmap

        Args:
            img_trough      : [C, H, W] original trough image
            img_peak        : [C, H, W] original peak image
            mask_trough     : [1, H, W] GT trough binary mask
            mask_peak       : [1, H, W] GT peak binary mask
            pulse_gt        : [1, H, W] GT pulsation map (binary)
            pulse_pred_prob : [1, H, W] predicted pulsation probabilities [0,1]
            save_path       : optional .png filepath
        """
        def to_np(t):
            arr = t.squeeze().detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr.transpose(1, 2, 0)
            return arr

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        def to_bool(t):
            return to_np(t) > 0.5

        img_tr  = norm(to_np(img_trough))
        img_pk  = norm(to_np(img_peak))
        msk_tr  = to_bool(mask_trough)
        msk_pk  = to_bool(mask_peak)
        pg      = to_np(pulse_gt)           # [H, W] binary
        pp      = to_np(pulse_pred_prob)    # [H, W] float [0,1]

        gt_overlay   = self._build_pulsation_overlay(msk_tr, msk_pk)
        pred_overlay = self._build_prob_heatmap_overlay(msk_tr, pp)

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.patch.set_facecolor('#0d0d0d')

        # ── Row 0: Original images + GT pulsation ─────────────────────
        axes[0, 0].imshow(img_tr, cmap='gray')
        axes[0, 0].set_title('Trough — Original', color='white', fontsize=11, pad=5)

        axes[0, 1].imshow(img_pk, cmap='gray')
        axes[0, 1].set_title('Peak — Original', color='white', fontsize=11, pad=5)

        axes[0, 2].imshow(pg, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('GT Pulsation Map(where vessel changed)',
                              color='white', fontsize=11, pad=5)

        # ── Row 1: GT masks + predicted prob map ──────────────────────
        axes[1, 0].imshow(msk_tr, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Trough — GT Mask', color='white', fontsize=11, pad=5)

        axes[1, 1].imshow(msk_pk, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Peak — GT Mask', color='white', fontsize=11, pad=5)

        im = axes[1, 2].imshow(pp, cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title('Pred Pulsation Prob(hot=high confidence)',
                              color='white', fontsize=11, pad=5)
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # ── Row 2: Overlays ───────────────────────────────────────────
        axes[2, 0].imshow(gt_overlay)
        axes[2, 0].set_title('GT Pulsation Overlay(white=core, green=dilation, red=contraction)',
                              color='white', fontsize=10, pad=5)

        axes[2, 1].imshow(pred_overlay)
        axes[2, 1].set_title('Pred Pulsation Overlay(on GT trough mask, hot=predicted change)',
                              color='white', fontsize=10, pad=5)

        # Diff: GT pulsation vs predicted pulsation
        diff = pp - pg.astype(float)
        im2 = axes[2, 2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2, 2].set_title('Pred - GT Diff(red=over-predicted, blue=missed)',
                              color='white', fontsize=10, pad=5)
        plt.colorbar(im2, ax=axes[2, 2], fraction=0.046, pad=0.04)

        for ax in axes.flat:
            ax.axis('off')
            ax.set_facecolor('#0d0d0d')

        plt.suptitle('Siamese Pulsation Map Comparison',
                     color='white', fontsize=14, y=1.01)
        plt.tight_layout(pad=1.5)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Saved pulsation map figure -> {save_path}")
        plt.show()
        plt.close(fig)

    @staticmethod
    def _build_prob_heatmap_overlay(base_mask: np.ndarray,
                                    prob_map: np.ndarray) -> np.ndarray:
        """
        Overlay predicted pulsation probability onto the base mask.
        base_mask pixels = dim white, high prob pixels = hot colour on top.
        """
        import matplotlib.cm as cm
        H, W   = base_mask.shape
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[base_mask] = [60, 60, 60]   # dim white for stable vessel

        # Map probability through 'hot' colormap
        cmap   = cm.get_cmap('hot')
        prob_rgb = (cmap(prob_map)[..., :3] * 255).astype(np.uint8)

        # Only show prob colour where prob > 0.1 (suppress background noise)
        active = prob_map > 0.1
        canvas[active] = prob_rgb[active]

        return canvas


    def plot_pulsation_map_full(self, img_trough, img_peak,
                                mask_trough, mask_peak,
                                pulse_gt, pulse_prob,
                                seg_prob=None,
                                pulse_threshold: float = 0.25,
                                save_path: str = None):
        """
        Comprehensive pulsation visualisation — 3 rows x 3 cols.

          Row 0 — Original images + GT pulsation map
                  Col 0: Trough original
                  Col 1: Peak original
                  Col 2: GT pulsation map (hot — where vessel changed)

          Row 1 — GT masks + predicted probability heatmap
                  Col 0: GT trough mask
                  Col 1: GT peak mask
                  Col 2: Pred pulsation prob (hot colourmap)

          Row 2 — Overlays comparing GT and prediction
                  Col 0: GT pulsation overlay (white=core, green=dilation, red=contraction)
                  Col 1: Pred pulsation overlay (same colour scheme, from thresholded prob)
                  Col 2: Error map — red=false positive, blue=false negative, white=correct

        Args:
            img_trough      : [C, H, W] original trough image tensor
            img_peak        : [C, H, W] original peak image tensor
            mask_trough     : [1, H, W] GT trough binary mask tensor
            mask_peak       : [1, H, W] GT peak binary mask tensor
            pulse_gt        : [1, H, W] GT pulsation map (binary) tensor
            pulse_prob      : [1, H, W] predicted pulsation probabilities tensor
            pulse_threshold : threshold to binarise pulse_prob for overlay/GIF
            save_path       : optional .png filepath
        """
        def to_np(t):
            arr = t.squeeze().detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr.transpose(1, 2, 0)
            return arr

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        def to_bool(t):
            return to_np(t) > 0.5

        img_tr   = norm(to_np(img_trough))
        img_pk   = norm(to_np(img_peak))
        msk_tr   = to_bool(mask_trough)
        msk_pk   = to_bool(mask_peak)
        pg       = to_np(pulse_gt)                              # [H,W] binary
        pp       = to_np(pulse_prob)                            # [H,W] float [0,1]
        pp_bin   = pp > pulse_threshold                         # [H,W] bool

        gt_overlay   = self._build_pulsation_overlay(msk_tr, msk_pk)

        # Pred overlay: stable vessel core from GT trough, pulsation from pred
        pred_core    = msk_tr & msk_pk
        pred_dil     = pp_bin & ~msk_tr    # pred pulse pixels outside trough
        pred_con     = msk_tr & ~pp_bin & ~msk_pk  # trough pixels model thinks contracted
        pred_overlay = np.zeros((*pg.shape, 3), dtype=np.uint8)
        pred_overlay[pred_core] = [220, 220, 220]
        pred_overlay[pred_dil]  = [30,  240,  80]
        pred_overlay[pred_con]  = [240,  50,  40]

        # Error map vs GT pulsation
        gt_bool  = pg > 0.5
        error    = np.zeros((*pg.shape, 3), dtype=np.uint8)
        error[gt_bool  &  pp_bin]  = [220, 220, 220]   # true positive — white
        error[gt_bool  & ~pp_bin]  = [50,  100, 240]   # false negative — blue (missed)
        error[~gt_bool &  pp_bin]  = [240,  80,  50]   # false positive — red (noise)

        legend_seg = [
            mpatches.Patch(color='white', label='Stable core'),
            mpatches.Patch(color='lime',  label='Dilation'),
            mpatches.Patch(color='red',   label='Contraction'),
        ]
        legend_err = [
            mpatches.Patch(color='white', label='True positive'),
            mpatches.Patch(color='blue',  label='False negative (missed)'),
            mpatches.Patch(color='red',   label='False positive (noise)'),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.patch.set_facecolor('#0d0d0d')

        # Row 0
        axes[0, 0].imshow(img_tr, cmap='gray')
        axes[0, 0].set_title('Trough — Original', color='white', fontsize=11, pad=5)
        axes[0, 1].imshow(img_pk, cmap='gray')
        axes[0, 1].set_title('Peak — Original', color='white', fontsize=11, pad=5)
        im0 = axes[0, 2].imshow(pg, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('GT Pulsation Map', color='white', fontsize=11, pad=5)
        plt.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Row 1
        if seg_prob is not None:
            sp = to_np(seg_prob)
            axes[1, 0].imshow(sp, cmap='hot', vmin=0, vmax=1)
            axes[1, 0].set_title('Pred Vessel Prob\n(hot=high confidence)', color='white', fontsize=11, pad=5)
        else:
            axes[1, 0].imshow(msk_tr, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title('GT Trough Mask', color='white', fontsize=11, pad=5)
        axes[1, 1].imshow(msk_pk, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('GT Peak Mask', color='white', fontsize=11, pad=5)
        im1 = axes[1, 2].imshow(pp, cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Pred Pulsation Prob(thresh={pulse_threshold:.2f})',
                              color='white', fontsize=11, pad=5)
        plt.colorbar(im1, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # Row 2
        axes[2, 0].imshow(gt_overlay)
        axes[2, 0].set_title('GT Pulsation Overlay', color='white', fontsize=11, pad=5)
        axes[2, 0].legend(handles=legend_seg, loc='lower right', fontsize=7,
                          framealpha=0.6, facecolor='#1a1a1a', labelcolor='white')
        axes[2, 1].imshow(pred_overlay)
        axes[2, 1].set_title('Pred Pulsation Overlay', color='white', fontsize=11, pad=5)
        axes[2, 1].legend(handles=legend_seg, loc='lower right', fontsize=7,
                          framealpha=0.6, facecolor='#1a1a1a', labelcolor='white')
        axes[2, 2].imshow(error)
        axes[2, 2].set_title('Error Map', color='white', fontsize=11, pad=5)
        axes[2, 2].legend(handles=legend_err, loc='lower right', fontsize=7,
                          framealpha=0.6, facecolor='#1a1a1a', labelcolor='white')

        for ax in axes.flat:
            ax.axis('off')
            ax.set_facecolor('#0d0d0d')

        plt.suptitle('Pulsation Analysis', color='white', fontsize=14, y=1.01)
        plt.tight_layout(pad=1.5)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Saved comparison figure -> {save_path}")
        plt.show()
        plt.close(fig)


    def create_soft_pulsation_gif(self, seg_prob, pulse_prob,
                                  save_path: str = "soft_pulsation.gif",
                                  n_frames: int = 30, fps: int = 12):
        """
        Soft pulsation GIF — no threshold, smooth edges.

        Uses raw probability maps directly:
          - Vessel core brightness ∝ seg_prob  (bright where model is confident)
          - Pulsation glow intensity ∝ pulse_prob × sin(alpha×π)
            (the pulsation zone glows and fades smoothly with the cardiac cycle)

        This is why the heatmap looks better than binary: the vessel boundary
        is gradient, not a hard edge. This GIF preserves that gradient.

        Args:
            seg_prob   : [1, H, W] vessel segmentation probabilities
            pulse_prob : [1, H, W] pulsation map probabilities
            save_path  : output .gif filepath
            n_frames   : frames per half-cycle
            fps        : playback speed
        """
        from scipy.ndimage import binary_dilation as morph_dilate

        seg   = seg_prob.squeeze().detach().cpu().numpy()    # [H,W] float [0,1]
        pulse = pulse_prob.squeeze().detach().cpu().numpy()  # [H,W] float [0,1]
        H, W  = seg.shape

        # Colour palette
        BG_R, BG_G, BG_B     = 10,  10,  10
        CORE_R, CORE_G, CORE_B = 180, 220, 255   # soft blue-white for vessel
        DIL_R,  DIL_G,  DIL_B  = 30,  240,  80   # green for pulsation

        def ease(t):
            return t * t * (3 - 2 * t)

        def render_soft_frame(alpha: float) -> np.ndarray:
            """
            alpha=0.0 → trough (dim), alpha=1.0 → peak (bright pulsation)
            """
            R = np.full((H, W), BG_R, dtype=np.float32)
            G = np.full((H, W), BG_G, dtype=np.float32)
            B = np.full((H, W), BG_B, dtype=np.float32)

            # Vessel core: brightness scales with seg_prob
            # At alpha=0 (trough) slightly dimmer, at alpha=1 (peak) full
            vessel_lum = seg * (0.7 + 0.3 * ease(alpha))
            R += (CORE_R - BG_R) * vessel_lum
            G += (CORE_G - BG_G) * vessel_lum
            B += (CORE_B - BG_B) * vessel_lum

            # Pulsation glow: pulse_prob × sine wave peaks at mid-cycle
            # Additive on top of vessel — pulsation zone brightens to green
            pulse_intensity = pulse * np.sin(alpha * np.pi)
            pulse_intensity = np.clip(pulse_intensity, 0, 1)
            R += (DIL_R - CORE_R) * pulse_intensity
            G += (DIL_G - CORE_G) * pulse_intensity
            B += (DIL_B - CORE_B) * pulse_intensity

            # Thin white border around the vessel boundary
            vessel_mask = seg > 0.3
            border = morph_dilate(vessel_mask, iterations=1) & ~vessel_mask
            glow   = 0.2 + 0.8 * np.sin(alpha * np.pi)
            R[border] = 255 * glow
            G[border] = 255 * glow
            B[border] = 255 * glow

            rgb = np.stack([
                np.clip(R, 0, 255).astype(np.uint8),
                np.clip(G, 0, 255).astype(np.uint8),
                np.clip(B, 0, 255).astype(np.uint8),
            ], axis=-1)
            return rgb

        hold   = max(2, n_frames // 8)
        alphas = (
            [0.0] * hold +
            list(np.linspace(0, 1, n_frames)) +
            [1.0] * hold +
            list(np.linspace(1, 0, n_frames))
        )

        frames = [
            Image.fromarray(render_soft_frame(a), mode='RGB').convert('P', dither=Image.NONE)
            for a in alphas
        ]

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        frames[0].save(
            save_path, format='GIF', save_all=True,
            append_images=frames[1:], loop=0,
            duration=int(1000/fps), optimize=False, disposal=2,
        )
        print(f"  Soft pulsation GIF saved -> {save_path}  ({len(frames)} frames @ {fps} fps)")
        return save_path

    @staticmethod
    def _build_pulsation_overlay(trough: np.ndarray, peak: np.ndarray) -> np.ndarray:
        """
        Static RGB overlay on white background.

          White background
          Dark gray = stable core
          Orange    = dilation (expanded at peak)
          Blue      = contraction (narrowed at peak)
        """
        H, W = trough.shape
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)

        # Stable vessel in both frames
        canvas[trough & peak] = [100, 100, 100]

        # Present only at peak = grew
        canvas[peak & ~trough] = [255, 140, 0]  # orange

        # Present only at trough = shrunk
        canvas[trough & ~peak] = [0, 120, 255]  # blue

        return canvas