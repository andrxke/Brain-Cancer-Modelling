import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import random
from scipy.ndimage import rotate as nd_rotate

class BraTSSAMDataset(Dataset):
    def __init__(self, data_dir, image_size=1024, jitter=5, slices_per_subject=3, augment=True):
        """
        Args:
            data_dir: Path to directory containing BraTS subject folders.
            image_size: SAM input size (typically 1024).
            jitter: Max pixels to randomly jitter the bounding box during training.
            slices_per_subject: Number of slices to sample per subject per epoch.
            augment: Whether to apply random augmentations (flips, rotation).
        """
        self.data_dir = data_dir
        self.subject_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.image_size = image_size
        self.jitter = jitter
        self.slices_per_subject = slices_per_subject
        self.augment = augment
        
    def __len__(self):
        # Multiply dataset length by slices_per_subject to sample more slices per epoch
        return len(self.subject_ids) * self.slices_per_subject
    
    def _normalize_to_uint8(self, image):
        """Normalizes a 2D image array to 0-255 uint8."""
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max == img_min:
             return np.zeros_like(image, dtype=np.uint8)
        normalized = (image - img_min) / (img_max - img_min)
        return (normalized * 255).astype(np.uint8)

    def _get_bounding_box(self, mask):
        """Returns [x_min, y_min, x_max, y_max] from a 2D mask with optional jitter."""
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Add random jitter
        j_x1 = random.randint(-self.jitter, self.jitter)
        j_y1 = random.randint(-self.jitter, self.jitter)
        j_x2 = random.randint(-self.jitter, self.jitter)
        j_y2 = random.randint(-self.jitter, self.jitter)
        
        x_min = max(0, x_min + j_x1)
        y_min = max(0, y_min + j_y1)
        x_max = min(mask.shape[1] - 1, x_max + j_x2)
        y_max = min(mask.shape[0] - 1, y_max + j_y2)
        
        # Ensure coordinates are valid box
        if x_max <= x_min: x_max = x_min + 1
        if y_max <= y_min: y_max = y_min + 1
        
        return np.array([x_min, y_min, x_max, y_max])

    def _apply_augmentation(self, image_rgb, gt_mask):
        """Apply random augmentations to image and mask consistently."""
        # Random horizontal flip
        if random.random() > 0.5:
            image_rgb = np.flip(image_rgb, axis=1).copy()
            gt_mask = np.flip(gt_mask, axis=1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            image_rgb = np.flip(image_rgb, axis=0).copy()
            gt_mask = np.flip(gt_mask, axis=0).copy()
        
        # Random rotation (±15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image_rgb = nd_rotate(image_rgb, angle, axes=(0, 1), reshape=False, order=1)
            gt_mask = nd_rotate(gt_mask, angle, axes=(0, 1), reshape=False, order=0)
        
        return image_rgb, gt_mask
        
    def __getitem__(self, idx):
        # Map the expanded index back to a subject
        subject_idx = idx % len(self.subject_ids)
        subject_id = self.subject_ids[subject_idx]
        subject_dir = os.path.join(self.data_dir, subject_id)
        
        seg_path = os.path.join(subject_dir, f'{subject_id}-seg.nii.gz')
        seg = nib.load(seg_path).get_fdata()
        
        t1c_path = os.path.join(subject_dir, f'{subject_id}-t1c.nii.gz')
        t2f_path = os.path.join(subject_dir, f'{subject_id}-t2f.nii.gz')
        t2w_path = os.path.join(subject_dir, f'{subject_id}-t2w.nii.gz')
        
        # Load modalities lazily to save memory
        t1c = nib.load(t1c_path).get_fdata()
        t2f = nib.load(t2f_path).get_fdata()
        t2w = nib.load(t2w_path).get_fdata()

        # Find all valid slices (where tumor area > 50 pixels)
        tumor_areas = np.sum(seg > 0, axis=(0, 1))
        valid_slices = np.where(tumor_areas > 50)[0]
        
        if len(valid_slices) == 0:
            # If no tumor, pick middle slice and create a dummy box/mask to prevent crashing
            best_slice_idx = seg.shape[2] // 2
            input_box = np.array([0, 0, 10, 10])
            slice_seg = np.zeros_like(seg[:, :, best_slice_idx])
        else:
            # Randomly pick a valid slice for this epoch
            best_slice_idx = random.choice(valid_slices)
            slice_seg = seg[:, :, best_slice_idx]
            slice_seg = np.transpose(slice_seg, (1, 0)) # transpose to (Y, X)
            input_box = self._get_bounding_box(slice_seg)

        slice_t1c = t1c[:, :, best_slice_idx]
        slice_t2f = t2f[:, :, best_slice_idx]
        slice_t2w = t2w[:, :, best_slice_idx]
            
        # Combine to RGB
        image_rgb = np.stack([
            self._normalize_to_uint8(slice_t1c),
            self._normalize_to_uint8(slice_t2f),
            self._normalize_to_uint8(slice_t2w)
        ], axis=-1)
        
        # Transpose to (Y, X, C)
        image_rgb = np.transpose(image_rgb, (1, 0, 2))
        
        # Return Ground Truth Mask as Binary (1 where tumor, 0 elsewhere)
        gt_mask = (slice_seg > 0).astype(np.float32)
        
        # Apply augmentation
        if self.augment:
            image_rgb, gt_mask = self._apply_augmentation(image_rgb, gt_mask)
            # Recompute bounding box after augmentation
            if np.sum(gt_mask > 0) > 0:
                input_box = self._get_bounding_box(gt_mask)
            else:
                input_box = np.array([0, 0, 10, 10])
        
        return {
            "image": image_rgb,          # np array (H, W, 3) 0-255 uint8
            "gt_mask": gt_mask,          # np array (H, W) 0.0 or 1.0 float32
            "box": input_box,            # np array [x1, y1, x2, y2]
            "orig_hw": image_rgb.shape[:2] # tuple (H, W)
        }
