import os
import sys
import torch
import numpy as np
from datetime import date
from tqdm import tqdm

SAM3_DIR = os.path.join(os.path.dirname(__file__), 'sam3')
sys.path.append(SAM3_DIR)

# BraTS utilities
from ..utils.model_utils import make_dataloader
from ..utils.general_utils import probs_to_preds, save_pred_as_nifti
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# ---------------------------------------------------------------------------
# Must match the text prompt used during training
# ---------------------------------------------------------------------------
TEXT_PROMPT = "brain tumor"


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

def get_normalized_sam3_box(box, img_h, img_w):
    """Returns [center_x, center_y, width, height] normalized in [0, 1]."""
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2.0 / img_w
    center_y = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return [center_x, center_y, width, height]


def get_sam3_model_with_lora(device, lora_ckpt_path=None):
    """Build SAM3 model and optionally load LoRA weights."""
    sam3_model = build_sam3_image_model(device=device, load_from_HF=True)

    if lora_ckpt_path and os.path.exists(lora_ckpt_path):
        print(f"Loading LoRA fine-tuned weights: {lora_ckpt_path}")
        # Apply the same LoRA config used during training
        lora_config = LoRAConfig(
            rank=16,
            alpha=32,
            dropout=0.0,  # No dropout at inference
            apply_to_vision_encoder=True,
            apply_to_text_encoder=True,
            apply_to_geometry_encoder=True,
            apply_to_detr_encoder=True,
            apply_to_detr_decoder=True,
            apply_to_mask_decoder=False,
        )
        sam3_model = apply_lora_to_model(sam3_model, lora_config)
        sam3_model = load_lora_weights(sam3_model, lora_ckpt_path, device=device)
        sam3_model = sam3_model.to(device)  # Ensure all LoRA params are on GPU

    return sam3_model


def infer_sam3(data_dir, unet_ckpt_path, sam3_ckpt_path=None, sam3_confidence_threshold=0.5, out_dir=None, postprocess_function=None):
    if out_dir is None:
        out_dir = os.getcwd()
    preds_dir = os.path.join(out_dir, 'preds_sam3_' + str(date.today()))
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
    training_regions = unet_checkpoint.get('training_regions', 'overlapping')

    print("---------------------------------------------------")
    print("LOADING SAM 3 REFINEMENT MODEL (with LoRA)")
    print("---------------------------------------------------")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam3_model = get_sam3_model_with_lora(device, lora_ckpt_path=sam3_ckpt_path)
    sam3_predictor = Sam3Processor(sam3_model, device=device, confidence_threshold=sam3_confidence_threshold)
    print("SAM 3 Model fully loaded.")

    # Dataloader (Batch size must be 1 for exact 3D volumes)
    test_loader = make_dataloader(data_dir, shuffle=False, mode='test', batch_size=1)

    print("---------------------------------------------------")
    print(f"INFERENCE STARTS (text_prompt='{TEXT_PROMPT}')")
    print("---------------------------------------------------")

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for subject_names, imgs in tqdm(test_loader, desc="Processing Volumes"):
                subject_name = subject_names[0]

                # 1. UNet 3D Stage (Detection)
                imgs = [img.to(device) for img in imgs]
                x_in = torch.cat(imgs, dim=1)

                unet_outputs = unet_model(x_in)
                if isinstance(unet_outputs, list):
                    unet_output = unet_outputs[0]
                else:
                    unet_output = unet_outputs

                unet_output = torch.sigmoid(unet_output.float())
                preds = probs_to_preds(unet_output, training_regions)
                preds_np = preds.squeeze(0).cpu().numpy()  # (3, H, W, D)

                x_in_np = x_in.squeeze(0).cpu().numpy()
                depth = preds_np.shape[3]

                # 2. SAM 3 Stage (Refinement)
                sam3_refined_preds = np.copy(preds_np)

                for z in range(depth):
                    if training_regions == 'overlapping':
                        wt_mask = preds_np[0, :, :, z]
                    else:
                        wt_mask = np.sum(preds_np[:, :, :, z], axis=0) > 0

                    box = get_bounding_box(wt_mask)
                    if box is not None:
                        slice_t1c = x_in_np[0, :, :, z]
                        slice_t2f = x_in_np[2, :, :, z]
                        slice_t2w = x_in_np[3, :, :, z]

                        image_rgb = np.stack([
                            normalize_to_uint8(slice_t1c),
                            normalize_to_uint8(slice_t2f),
                            normalize_to_uint8(slice_t2w)
                        ], axis=-1)

                        image_rgb = np.transpose(image_rgb, (1, 0, 2))

                        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)  # [3, Y, X]

                        box_transposed = [box[1], box[0], box[3], box[2]]
                        img_h, img_w = image_rgb.shape[:2]
                        sam3_box = get_normalized_sam3_box(box_transposed, img_h, img_w)

                        # Use text-prompted inference with Sam3Processor
                        inference_state = sam3_predictor.set_image(image_tensor)
                        inference_state = sam3_predictor.add_geometric_prompt(box=sam3_box, label=True, state=inference_state)

                        if "masks" in inference_state and inference_state["masks"].shape[0] > 0:
                            predicted_wt_mask = inference_state["masks"][0, 0].cpu().numpy()
                            predicted_wt_mask = np.transpose(predicted_wt_mask, (1, 0))

                            # 3. Class Refinement
                            if training_regions == 'overlapping':
                                sam3_refined_preds[0, :, :, z] = predicted_wt_mask
                                sam3_refined_preds[1, :, :, z] = sam3_refined_preds[1, :, :, z] * predicted_wt_mask
                                sam3_refined_preds[2, :, :, z] = sam3_refined_preds[2, :, :, z] * predicted_wt_mask
                            else:
                                sam3_refined_preds[0, :, :, z] = sam3_refined_preds[0, :, :, z] * predicted_wt_mask
                                sam3_refined_preds[1, :, :, z] = sam3_refined_preds[1, :, :, z] * predicted_wt_mask
                                sam3_refined_preds[2, :, :, z] = sam3_refined_preds[2, :, :, z] * predicted_wt_mask

                final_pred_tensor = torch.from_numpy(sam3_refined_preds).unsqueeze(0).cpu()
                save_pred_as_nifti(final_pred_tensor[0], preds_dir, data_dir, subject_name, postprocess_function)

    print(f"Inference complete! Results saved to {preds_dir}.")


if __name__ == '__main__':
    # Configuration
    data_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
    unet_ckpt_path = '/home/andrek/KurtBraTS/debug/train_with_vit/best_dice_ckpt.pth.tar'

    out_dir = '/home/andrek/KurtBraTS/data/dataset/'

    # Point to the best LoRA checkpoint
    sam3_ckpt_path = os.path.join(os.path.dirname(__file__), 'sam3_lora_best.pt')

    from ..processing.postprocess import rm_dust_fh

    infer_sam3(data_dir, unet_ckpt_path, sam3_ckpt_path=sam3_ckpt_path, sam3_confidence_threshold=0.5, out_dir=out_dir, postprocess_function=rm_dust_fh)
