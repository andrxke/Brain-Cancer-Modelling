"""Visualization script for prediction masks overlaid on MRI data.

Usage (from the KurtBraTS directory):
    python -m brats2023_updated.utils.visualize_preds \
        --data_dir <path_to_subject_data> \
        --preds_dir <path_to_predictions> \
        [--subject <subject_name>]    # If omitted, lists available subjects
        [--modality t1c]              # MRI modality to display (t1c, t1n, t2f, t2w)
        [--slices 5]                  # Number of evenly-spaced axial slices to show
        [--save <output_path.png>]    # Save figure instead of showing interactively
        [--gt]                        # Also show ground truth segmentation if available

Examples:
    # List all subjects in the predictions folder:
    python -m brats2023_updated.utils.visualize_preds \
        --data_dir /path/to/ValidationData \
        --preds_dir /path/to/preds2026-01-28

    # Visualize a specific subject (5 slices, T1c modality):
    python -m brats2023_updated.utils.visualize_preds \
        --data_dir /path/to/ValidationData \
        --preds_dir /path/to/preds2026-01-28 \
        --subject BraTS-GLI-00000-000

    # Compare prediction vs ground truth on T2-FLAIR:
    python -m brats2023_updated.utils.visualize_preds \
        --data_dir /path/to/TrainingData \
        --preds_dir /path/to/preds2026-01-28 \
        --subject BraTS-GLI-00000-000 \
        --modality t2f --gt
"""

import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ── Colour map for BraTS labels ──────────────────────────────────────────────
# 0 = background (transparent), 1 = NCR, 2 = ED, 3 = ET
LABEL_COLORS = [
    (0, 0, 0, 0),          # 0: background – fully transparent
    (0.12, 0.47, 0.71, 1), # 1: NCR  – blue
    (0.20, 0.63, 0.17, 1), # 2: ED   – green
    (0.89, 0.10, 0.11, 1), # 3: ET   – red
]
LABEL_CMAP = ListedColormap(LABEL_COLORS)
LEGEND_ELEMENTS = [
    Patch(facecolor=LABEL_COLORS[1], label='NCR (Necrotic Core)'),
    Patch(facecolor=LABEL_COLORS[2], label='ED (Edema)'),
    Patch(facecolor=LABEL_COLORS[3], label='ET (Enhancing Tumor)'),
]


def list_available_subjects(preds_dir):
    """List subjects that have prediction files in the given directory."""
    subjects = []
    for f in sorted(os.listdir(preds_dir)):
        if f.endswith('.nii.gz'):
            subjects.append(f.replace('.nii.gz', ''))
    return subjects


def load_volume(path):
    """Load a NIfTI file and return its data as a numpy array."""
    return nib.load(path).get_fdata()


def pick_slices(volume, n_slices):
    """Pick n evenly-spaced axial slice indices that pass through the volume center."""
    depth = volume.shape[2]
    return np.linspace(depth * 0.2, depth * 0.8, n_slices, dtype=int)


def find_tumor_slices(mask, n_slices):
    """Pick n slices that have the most tumor voxels (more interesting to look at)."""
    depth = mask.shape[2]
    counts = np.array([(mask[:, :, z] > 0).sum() for z in range(depth)])

    if counts.sum() == 0:
        # No tumor found – fall back to evenly spaced
        return np.linspace(depth * 0.2, depth * 0.8, n_slices, dtype=int)

    # Get indices sorted by tumor voxel count (descending), then spread them out
    top_indices = np.argsort(counts)[::-1]
    # Filter to slices with at least some tumor
    top_indices = top_indices[counts[top_indices] > 0]

    if len(top_indices) <= n_slices:
        selected = sorted(top_indices)
    else:
        # Spread the selection evenly across the range of tumor-containing slices
        selected = np.linspace(0, len(top_indices) - 1, n_slices, dtype=int)
        selected = sorted(top_indices[selected])

    return selected


def visualize_subject(data_dir, preds_dir, subject_name, modality='t1c',
                      n_slices=5, save_path=None, show_gt=False):
    """Visualize prediction mask overlaid on MRI for a single subject.

    Args:
        data_dir:     Path to the parent directory containing subject folders.
        preds_dir:    Path to the directory containing prediction NIfTI files.
        subject_name: Name of the subject (e.g. 'BraTS-GLI-00000-000').
        modality:     MRI modality to display ('t1c', 't1n', 't2f', 't2w').
        n_slices:     Number of axial slices to show.
        save_path:    If provided, save figure to this path instead of showing.
        show_gt:      If True, also display the ground truth segmentation.
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    mri_path = os.path.join(data_dir, subject_name, f'{subject_name}-{modality}.nii.gz')
    pred_path = os.path.join(preds_dir, f'{subject_name}.nii.gz')

    if not os.path.exists(mri_path):
        raise FileNotFoundError(f"MRI file not found: {mri_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    mri = load_volume(mri_path)
    pred = load_volume(pred_path)

    gt = None
    if show_gt:
        gt_path = os.path.join(data_dir, subject_name, f'{subject_name}-seg.nii.gz')
        if os.path.exists(gt_path):
            gt = load_volume(gt_path)
        else:
            print(f"⚠  Ground truth not found at {gt_path}, skipping GT row.")
            show_gt = False

    # ── Select slices ─────────────────────────────────────────────────────────
    slice_indices = find_tumor_slices(pred, n_slices)

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows = 3 if show_gt else 2  # MRI | MRI+Pred [| MRI+GT]
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(4 * n_slices, 4 * n_rows))

    if n_slices == 1:
        axes = axes[:, np.newaxis]  # ensure 2D indexing

    row_labels = ['MRI', 'Prediction Overlay']
    if show_gt:
        row_labels.append('Ground Truth Overlay')

    for col, z in enumerate(slice_indices):
        mri_slice = mri[:, :, z]
        pred_slice = pred[:, :, z]

        # Row 0: MRI only
        axes[0, col].imshow(np.rot90(mri_slice), cmap='gray')
        axes[0, col].set_title(f'Slice {z}', fontsize=11, fontweight='bold')
        axes[0, col].axis('off')

        # Row 1: MRI + prediction overlay
        axes[1, col].imshow(np.rot90(mri_slice), cmap='gray')
        pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[1, col].imshow(np.rot90(pred_masked), cmap=LABEL_CMAP,
                            vmin=0, vmax=3, alpha=0.55, interpolation='nearest')
        axes[1, col].axis('off')

        # Row 2 (optional): MRI + ground truth overlay
        if show_gt and gt is not None:
            gt_slice = gt[:, :, z]
            axes[2, col].imshow(np.rot90(mri_slice), cmap='gray')
            gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
            axes[2, col].imshow(np.rot90(gt_masked), cmap=LABEL_CMAP,
                                vmin=0, vmax=3, alpha=0.55, interpolation='nearest')
            axes[2, col].axis('off')

    # Row labels on the left
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=13, fontweight='bold', rotation=90,
                               labelpad=15)
        axes[i, 0].yaxis.set_label_position('left')
        # Re-enable the left axis line just for the label
        axes[i, 0].tick_params(left=False, labelleft=False)

    fig.suptitle(f'{subject_name}  –  {modality.upper()}', fontsize=15, fontweight='bold', y=0.98)
    fig.legend(handles=LEGEND_ELEMENTS, loc='lower center', ncol=3, fontsize=11,
               frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize BraTS prediction masks overlaid on MRI.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--data_dir', required=True,
                        help='Directory of subject data (containing subject sub-folders).')
    parser.add_argument('--preds_dir', required=True,
                        help='Directory containing prediction .nii.gz files.')
    parser.add_argument('--subject', default=None,
                        help='Subject name. If omitted, lists all available subjects.')
    parser.add_argument('--modality', default='t1c', choices=['t1c', 't1n', 't2f', 't2w'],
                        help='MRI modality to display (default: t1c).')
    parser.add_argument('--slices', type=int, default=5,
                        help='Number of axial slices to display (default: 5).')
    parser.add_argument('--save', default=None,
                        help='Save figure to this path instead of showing interactively.')
    parser.add_argument('--gt', action='store_true',
                        help='Also show ground truth segmentation overlay.')

    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.subject is None:
        subjects = list_available_subjects(args.preds_dir)
        print(f"\n{'─' * 50}")
        print(f"  Found {len(subjects)} subjects in {args.preds_dir}")
        print(f"{'─' * 50}")
        for s in subjects:
            print(f"  • {s}")
        print(f"\nRe-run with --subject <name> to visualize one.")
        return

    # ── Visualize mode ────────────────────────────────────────────────────────
    visualize_subject(
        data_dir=args.data_dir,
        preds_dir=args.preds_dir,
        subject_name=args.subject,
        modality=args.modality,
        n_slices=args.slices,
        save_path=args.save,
        show_gt=args.gt,
    )


if __name__ == '__main__':
    main()
