import os
import sys
import torch
import numpy as np
from datetime import date
from tqdm import tqdm

SAM2_DIR = os.path.join(os.path.dirname(__file__), 'segment-anything-2')
sys.path.append(SAM2_DIR)

# BraTS utilities
from ..utils.model_utils import make_dataloader
from ..utils.general_utils import probs_to_preds, save_pred_as_nifti
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def normalize_to_uint8(image):
    """Normalizes a 2D image array to 0-255 uint8."""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max == img_min:
         return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - img_min) / (img_max - img_min)
    return (normalized * 255).astype(np.uint8)

def get_bounding_box(mask):
    """Returns [x_min, y_min, x_max, y_max] from a 2D mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Add a minimal margin (like our training simulation jitter)
    margin = 5
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(mask.shape[1] - 1, x_max + margin)
    y_max = min(mask.shape[0] - 1, y_max + margin)
    return np.array([x_min, y_min, x_max, y_max])

def infer_sam2(data_dir, unet_ckpt_path, sam2_ckpt_path, out_dir=None, postprocess_function=None):
    if out_dir is None:
        out_dir = os.getcwd()
    preds_dir = os.path.join(out_dir, 'preds_sam2_' + str(date.today()))
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("---------------------------------------------------")
    print("LOADING UNET3D DETECTION MODEL")
    print("---------------------------------------------------")
    unet_checkpoint = torch.load(unet_ckpt_path, weights_only=False, map_location=device)
    unet_model = unet_checkpoint['model']
    unet_model.load_state_dict(unet_checkpoint['model_sd'])
    unet_model.to(device)
    unet_model.eval()
    training_regions = unet_checkpoint['training_regions']
    
    print("---------------------------------------------------")
    print("LOADING SAM 2 REFINEMENT MODEL")
    print("---------------------------------------------------")
    checkpoint_base = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    os.chdir(SAM2_DIR)
    sam2_model = build_sam2(model_cfg, checkpoint_base, device=device)
    os.chdir(os.path.dirname(__file__))
    
    sam2_finetuned_checkpoint = torch.load(sam2_ckpt_path, map_location=device)
    sam2_model.load_state_dict(sam2_finetuned_checkpoint['model_state_dict'])
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM 2 Model fully loaded.")
    
    # Dataloader (Batch size must be 1 for exact 3D volumes)
    test_loader = make_dataloader(data_dir, shuffle=False, mode='test', batch_size=1)
    
    print("---------------------------------------------------")
    print("INFERENCE STARTS")
    print("---------------------------------------------------")
    
    # We do bfloat16 precision for SAM2 if supported
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with torch.no_grad():
        for subject_names, imgs in tqdm(test_loader, desc="Processing Volumes"):
            subject_name = subject_names[0]
            
            # 1. UNet 3D Stage (Detection)
            imgs = [img.to(device) for img in imgs] # each img is B1HWD
            x_in = torch.cat(imgs, dim=1) # B4HWD
            
            unet_outputs = unet_model(x_in)
            if isinstance(unet_outputs, list):
                unet_output = unet_outputs[0]
            else:
                unet_output = unet_outputs
                
            unet_output = torch.sigmoid(unet_output.float())
            preds = probs_to_preds(unet_output, training_regions)
            # preds is shape (1, 3, H, W, D)
            preds_np = preds.squeeze(0).cpu().numpy() # (3, H, W, D)
            
            # Convert x_in to numpy for SAM RGB extraction (1, 4, H, W, D) -> (4, H, W, D)
            x_in_np = x_in.squeeze(0).cpu().numpy()
            
            depth = preds_np.shape[3]
            
            # 2. SAM 2 Stage (Refinement)
            sam2_refined_preds = np.copy(preds_np)
            
            for z in range(depth):
                # WT is at index 0 or derived from disjoint channels if disjoint was used
                if training_regions == 'overlapping':
                    # Indices: 0->WT, 1->TC, 2->ET
                    wt_mask = preds_np[0, :, :, z]
                else:
                    # 'disjoint' mapping: 0->NCR, 1->ED, 2->ET
                    wt_mask = np.sum(preds_np[:, :, :, z], axis=0) > 0
                
                box = get_bounding_box(wt_mask)
                if box is not None:
                    # Prepare RGB image: [T1c (ch0), T2f (ch2), T2w (ch3)]
                    slice_t1c = x_in_np[0, :, :, z]
                    slice_t2f = x_in_np[2, :, :, z]
                    slice_t2w = x_in_np[3, :, :, z]
                    
                    image_rgb = np.stack([
                        normalize_to_uint8(slice_t1c),
                        normalize_to_uint8(slice_t2f),
                        normalize_to_uint8(slice_t2w)
                    ], axis=-1)
                    
                    # Transpose to (Y, X, 3) format for PIL/SAM
                    image_rgb = np.transpose(image_rgb, (1, 0, 2)) 
                    
                    # Also the box coordinates must be mapped similarly if there's transposition
                    # Originally BraTS features are (X, Y) but arrays are usually accessed (X, Y).
                    # 'np.transpose(1, 0)' swaps X and Y.
                    # Bounding Box coordinates are [X_min, Y_min, X_max, Y_max].
                    # Let's ensure the bounding box correctly tracks the transposed image.
                    box_transposed = np.array([box[1], box[0], box[3], box[2]])
                    
                    # Apply SAM 2 Predictor
                    sam2_predictor.set_image(image_rgb)
                    
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        masks, scores, logits = sam2_predictor.predict(
                            box=box_transposed,
                            multimask_output=False,
                        )
                        
                    predicted_wt_mask = masks[0] > 0 # (Y, X)
                    
                    # Transpose back to (X, Y)
                    predicted_wt_mask = np.transpose(predicted_wt_mask, (1, 0))
                    
                    # 3. Class Refinement
                    # Overwrite the WT mask with the SAM 2 refined boundary
                    if training_regions == 'overlapping':
                        sam2_refined_preds[0, :, :, z] = predicted_wt_mask
                        # Cookie cutter: ensure TC and ET are STRICTLY inside the new WT mask
                        sam2_refined_preds[1, :, :, z] = sam2_refined_preds[1, :, :, z] * predicted_wt_mask
                        sam2_refined_preds[2, :, :, z] = sam2_refined_preds[2, :, :, z] * predicted_wt_mask
                    else:
                        # For disjoint, zero out anything outside the new WT
                        sam2_refined_preds[0, :, :, z] = sam2_refined_preds[0, :, :, z] * predicted_wt_mask
                        sam2_refined_preds[1, :, :, z] = sam2_refined_preds[1, :, :, z] * predicted_wt_mask
                        sam2_refined_preds[2, :, :, z] = sam2_refined_preds[2, :, :, z] * predicted_wt_mask

            # Convert back to tensor (keeping it on CPU for save_pred_as_nifti numpy conversion)
            final_pred_tensor = torch.from_numpy(sam2_refined_preds).unsqueeze(0).cpu()
            
            # Save using native logic
            save_pred_as_nifti(final_pred_tensor[0], preds_dir, data_dir, subject_name, postprocess_function)
            
    print(f"Inference complete! Results strictly saved targeting Submission logic inside {preds_dir}.")


if __name__ == '__main__':
    # Configuration
    data_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
    unet_ckpt_path = '/home/andrek/KurtBraTS/debug/train_with_vit/best_dice_ckpt.pth.tar'
    
    # Point to the last fine-tuned Checkpoint
    sam2_ckpt_path = os.path.join(os.path.dirname(__file__), 'sam2_finetuned_epoch3.pt')
    out_dir = '/home/andrek/KurtBraTS/data/dataset/'
    
    # Native imported script logic automatically un-dusts the arrays.
    from ..processing.postprocess import rm_dust_fh
    
    infer_sam2(data_dir, unet_ckpt_path, sam2_ckpt_path, out_dir=out_dir, postprocess_function=rm_dust_fh)
