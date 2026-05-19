import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch

# Ensure sam3 is in the python path
SAM3_DIR = os.path.join(os.path.dirname(__file__), 'sam3')
sys.path.append(SAM3_DIR)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

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

def get_normalized_sam3_box(box, img_h, img_w):
    """Returns [center_x, center_y, width, height] normalized in [0, 1]."""
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2.0 / img_w
    center_y = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return [center_x, center_y, width, height]

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
    img_h, img_w = image_rgb.shape[:2]
    sam3_box = get_normalized_sam3_box(input_box, img_h, img_w)
    
    # 5. Initialize SAM 3
    print("Loading SAM 3 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build SAM 3
    sam3_model = build_sam3_image_model(device=device, load_from_HF=True)
    processor = Sam3Processor(sam3_model, device=device)
    
    # 6. Predict
    print("Running inference...")
    inference_state = processor.set_image(image_rgb)
    inference_state = processor.add_geometric_prompt(box=sam3_box, label=True, state=inference_state)
    
    # The output mask is of shape (1, 1, H, W).
    predicted_mask = inference_state["masks"][0, 0].cpu().numpy()
    score = inference_state["scores"][0].item()
    
    print(f"Prediction successful! Confidence Score: {score:.4f}")
    
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
    axes[1].set_title(f"SAM 3 Prediction (Score: {score:.3f})")
    axes[1].axis('off')
    
    # C) Comparison Overlay (True Positives, False Positives, False Negatives)
    axes[2].imshow(image_rgb)
    # create colored mask overlay
    overlay = np.zeros((*slice_seg.shape, 4))
    gt_bin = (slice_seg > 0).astype(bool)
    pred_bin = predicted_mask.astype(bool)
    
    # True Positive (Yellow)
    overlay[gt_bin & pred_bin] = [1, 1, 0, 0.6] 
    # False Positive (Red - predicted but wrong)
    overlay[~gt_bin & pred_bin] = [1, 0, 0, 0.6]
    # False Negative (Blue - missed by SAM)
    overlay[gt_bin & ~pred_bin] = [0, 0, 1, 0.6]
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Yellow=TP, Red=FP, Blue=FN)")
    axes[2].axis('off')
    
    output_path = "sam3_brats_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
