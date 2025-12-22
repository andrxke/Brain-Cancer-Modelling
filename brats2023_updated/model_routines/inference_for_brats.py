#!/usr/bin/env python3
"""
Inference script for generating BraTS challenge submissions.
Loads trained model and generates predictions on official BraTS validation/test data.
"""

import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader

from ..datasets import brats_dataset
from ..utils.general_utils import probs_to_preds, overlapping_to_disjoint

def inference_for_brats(model_checkpoint_path, data_dir, output_dir, batch_size=1):
    """
    Generate predictions for BraTS challenge submission.
    
    Args:
        model_checkpoint_path: Path to trained model checkpoint
        data_dir: Directory containing official BraTS validation/test data
        output_dir: Directory to save prediction files
        batch_size: Batch size for inference
    """
    
    # Load trained model
    print(f"Loading model from: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location='cuda')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_sd'])
    model.cuda()
    model.eval()
    
    training_regions = checkpoint.get('training_regions', 'overlapping')
    print(f"Model was trained on: {training_regions} regions")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset for inference (mode='test' - no segmentation expected)
    dataset = brats_dataset.BratsDataset(data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    print(f"Found {len(dataset)} subjects for inference")
    print("Starting inference...")
    
    with torch.no_grad():
        for batch_idx, (subject_names, imgs, _) in enumerate(dataloader):
            
            # Move data to GPU
            imgs = [img.cuda() for img in imgs]  # List of B1HWD tensors
            
            # Concatenate input modalities
            x_in = torch.cat(imgs, dim=1)  # B4HWD
            
            # Forward pass
            output = model(x_in)  # B3HWD (3 channels for overlapping regions)
            output = output.float()
            
            # Convert probabilities to predictions
            preds = probs_to_preds(output, training_regions)  # B3HWD
            
            # Convert to disjoint regions (required for BraTS submission)
            if training_regions == 'overlapping':
                preds_disjoint = overlapping_to_disjoint(preds)  # B3HWD
            else:
                preds_disjoint = preds
            
            # Convert to 4-class format (background + 3 tumor classes)
            preds_4class = convert_to_brats_format(preds_disjoint)  # B1HWD
            
            # Save predictions for each subject in batch
            for i, subject_name in enumerate(subject_names):
                save_prediction(preds_4class[i], subject_name, data_dir, output_dir)
                print(f"Processed {batch_idx * batch_size + i + 1}/{len(dataset)}: {subject_name}")
    
    print(f"Inference completed! Predictions saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Compress the output directory: tar -czf predictions.tar.gz predictions/")
    print("2. Submit to BraTS evaluation server")
    print("3. Check submission format matches BraTS requirements")

def convert_to_brats_format(preds_disjoint):
    """
    Convert 3-channel disjoint predictions to 4-class BraTS format.
    
    Args:
        preds_disjoint: B3HWD tensor with disjoint region predictions
        
    Returns:
        B1HWD tensor with values [0, 1, 2, 4] for [background, NCR, ED, ET]
    """
    batch_size = preds_disjoint.shape[0]
    H, W, D = preds_disjoint.shape[2:]
    
    # Initialize output with background (0)
    output = torch.zeros(batch_size, 1, H, W, D, device=preds_disjoint.device)
    
    # BraTS labels: 0=background, 1=NCR, 2=ED, 4=ET
    output[preds_disjoint[:, 0:1] == 1] = 1  # NCR
    output[preds_disjoint[:, 1:2] == 1] = 2  # ED  
    output[preds_disjoint[:, 2:3] == 1] = 4  # ET
    
    return output

def save_prediction(prediction, subject_name, data_dir, output_dir):
    """
    Save prediction in NIfTI format matching original image geometry.
    
    Args:
        prediction: 1HWD tensor with prediction
        subject_name: Name of the subject
        data_dir: Original data directory (to get header info)
        output_dir: Directory to save prediction
    """
    
    # Load original image to get header/affine information
    original_img_path = os.path.join(data_dir, subject_name, f"{subject_name}-t1n.nii.gz")
    original_nii = nib.load(original_img_path)
    
    # Convert prediction to numpy and remove channel dimension
    pred_numpy = prediction.squeeze(0).cpu().numpy().astype(np.uint8)  # HWD
    
    # Create NIfTI image with same header as original
    pred_nii = nib.Nifti1Image(pred_numpy, original_nii.affine, original_nii.header)
    
    # Save prediction
    output_path = os.path.join(output_dir, f"{subject_name}.nii.gz")
    nib.save(pred_nii, output_path)

if __name__ == '__main__':
    
    # Configuration for BraTS submission
    model_checkpoint = '/home/andrek/KurtBraTS/debug/train_with_vit/best_dice_ckpt.pth.tar'
    
    # Official BraTS validation data (download from BraTS website)
    brats_val_data = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
    
    # Output directory for predictions
    output_dir = '/home/andrek/KurtBraTS/brats_predictions_vit'
    
    inference_for_brats(model_checkpoint, brats_val_data, output_dir)
