import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch

# Ensure segment-anything-2 is in the python path
SAM2_DIR = os.path.join(os.path.dirname(__file__), 'segment-anything-2')
sys.path.append(SAM2_DIR)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_nifti(path):
    return nib.load(path).get_fdata()

def get_bounding_box(mask):
    """Returns [x_min, y_min, x_max, y_max] from a 2D mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Optional: add a small margin
    margin = 5
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(mask.shape[1] - 1, x_max + margin)
    y_max = min(mask.shape[0] - 1, y_max + margin)
    
    return np.array([x_min, y_min, x_max, y_max])

def normalize_to_uint8(image):
    """Normalizes a 2D image array to 0-255 uint8."""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max == img_min:
         return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - img_min) / (img_max - img_min)
    return (normalized * 255).astype(np.uint8)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) # Dodger Blue
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def main():
    # 1. Define paths
    base_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    subject_id = 'BraTS-GLI-00000-000' # Change to any validation/train subject
    subject_dir = os.path.join(base_dir, subject_id)
    
    t1c_path = os.path.join(subject_dir, f'{subject_id}-t1c.nii.gz')
    t2f_path = os.path.join(subject_dir, f'{subject_id}-t2f.nii.gz')
    t2w_path = os.path.join(subject_dir, f'{subject_id}-t2w.nii.gz')
    seg_path = os.path.join(subject_dir, f'{subject_id}-seg.nii.gz')
    
    print(f"Loading data for {subject_id}...")
    t1c = load_nifti(t1c_path)
    t2f = load_nifti(t2f_path)
    t2w = load_nifti(t2w_path)
    seg = load_nifti(seg_path)
    
    # 2. Find a slice with a significant tumor
    tumor_areas = np.sum(seg > 0, axis=(0, 1))
    best_slice_idx = np.argmax(tumor_areas)
    print(f"Selected axial slice index {best_slice_idx} with max tumor area ({tumor_areas[best_slice_idx]} pixels).")
    
    slice_t1c = t1c[:, :, best_slice_idx]
    slice_t2f = t2f[:, :, best_slice_idx]
    slice_t2w = t2w[:, :, best_slice_idx]
    slice_seg = seg[:, :, best_slice_idx]
    
    # 3. Create RGB image array
    image_rgb = np.stack([
        normalize_to_uint8(slice_t1c),
        normalize_to_uint8(slice_t2f),
        normalize_to_uint8(slice_t2w)
    ], axis=-1)
    
    # Note: Nibabel loads in (X, Y, Z). Let's transpose to (Y, X) to match standard image plotting where X is width.
    image_rgb = np.transpose(image_rgb, (1, 0, 2))
    slice_seg = np.transpose(slice_seg, (1, 0))
    
    # 4. Get Bounding Box from Ground Truth
    input_box = get_bounding_box(slice_seg)
    if input_box is None:
        print("No tumor found in this slice.")
        return
        
    print(f"Tumor Bounding Box: {input_box}")
    
    # 5. Initialize SAM 2
    checkpoint = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # Relative to SAM2 repo root
    
    print("Loading SAM 2 model...")
    # Change current working directory to SAM2_DIR so it can find the config
    os.chdir(SAM2_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # 6. Predict
    print("Running inference...")
    predictor.set_image(image_rgb)
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False,
    )
    
    # The output mask is of shape (1, H, W). Let's take the first mask.
    predicted_mask = masks[0]
    score = scores[0]
    
    print(f"Prediction successful! Confidence Score: {score:.4f}")
    
    # Change dir back
    os.chdir(os.path.dirname(__file__))
    
    # 7. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # A) Original Image + Box + Ground Truth
    axes[0].imshow(image_rgb)
    show_box(input_box, axes[0])
    show_mask((slice_seg > 0).astype(int), axes[0], random_color=False)
    axes[0].set_title(f"Ground Truth (Slice {best_slice_idx})")
    axes[0].axis('off')
    
    # B) Original Image + Box + SAM Prediction
    axes[1].imshow(image_rgb)
    show_box(input_box, axes[1])
    show_mask(predicted_mask, axes[1], random_color=False)
    axes[1].set_title(f"SAM 2 Prediction (Score: {score:.3f})")
    axes[1].axis('off')
    
    # C) Comparison Overlay (True Positives, False Positives, False Negatives)
    axes[2].imshow(image_rgb)
    # create colored mask overlay
    overlay = np.zeros((*slice_seg.shape, 4))
    gt_bin = (slice_seg > 0).astype(bool)
    pred_bin = predicted_mask.astype(bool)
    
    # True Positive (White/Grayish) -> Actually let's use Yellow
    overlay[gt_bin & pred_bin] = [1, 1, 0, 0.6] 
    # False Positive (Red - predicted but wrong)
    overlay[~gt_bin & pred_bin] = [1, 0, 0, 0.6]
    # False Negative (Blue - missed by SAM)
    overlay[gt_bin & ~pred_bin] = [0, 0, 1, 0.6]
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Yellow=TP, Red=FP, Blue=FN)")
    axes[2].axis('off')
    
    output_path = "sam2_brats_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
