from torch.utils.data import Dataset
import os
import nibabel as nib
from ..processing.preprocess import znorm_rescale, center_crop
import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
)

class BratsDataset(Dataset):
    """Dataset class for loading BraTS training and test data.
    
    Args:
        data_dir: Directory of training or test data.
        mode: Either 'train', 'test', or 'val' specifying which data is being loaded.
              'val' mode tries to load segmentation but handles missing files gracefully.
    """
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.subject_list = os.listdir(data_dir)
        self.mode = mode

        # Define training augmentation transforms.
        # Uses MONAI dictionary-based transforms so the same spatial transform
        # is applied consistently to all image modalities AND the segmentation.
        if self.mode == 'train':
            image_keys = ['image']
            all_keys = ['image', 'seg']

            self.train_transforms = Compose([
                # Spatial augmentations (applied to both image and seg)
                RandFlipd(keys=all_keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=all_keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys=all_keys, spatial_axis=2, prob=0.5),

                # Intensity augmentations (applied to image only, not seg)
                # Reduced probabilities and magnitudes for better convergence
                RandScaleIntensityd(keys=image_keys, factors=0.05, prob=0.3),
                RandShiftIntensityd(keys=image_keys, offsets=0.05, prob=0.3),
                RandGaussianNoised(keys=image_keys, std=0.05, prob=0.1),
                RandGaussianSmoothd(keys=image_keys, prob=0.1,
                                    sigma_x=(0.5, 1.15),
                                    sigma_y=(0.5, 1.15),
                                    sigma_z=(0.5, 1.15)),
            ])

    def __len__(self):
        return len(self.subject_list)
    
    def load_nifti(self, subject_name, suffix):        
        """Loads nifti file for given subject and suffix."""

        nifti_filename = f'{subject_name}-{suffix}.nii.gz'
        nifti_path = os.path.join(self.data_dir, subject_name, nifti_filename)
        nifti = nib.load(nifti_path)
        return nifti
    
    def load_subject_data(self, subject_name):
        """Loads images (and segmentation if in train/val mode) and extra info for a subject."""

        modalities_data = []
        for suffix in ['t1c', 't1n', 't2f', 't2w']:
            modality_nifti = self.load_nifti(subject_name, suffix)
            modality_data = modality_nifti.get_fdata()
            modalities_data.append(modality_data)

        if self.mode == 'train':
            seg_nifti = self.load_nifti(subject_name, 'seg')
            seg_data = seg_nifti.get_fdata()
            return modalities_data, seg_data
        elif self.mode == 'val':
            # Try to load segmentation, but handle missing files gracefully
            try:
                seg_nifti = self.load_nifti(subject_name, 'seg')
                seg_data = seg_nifti.get_fdata()
                return modalities_data, seg_data
            except FileNotFoundError:
                # No segmentation file available
                return modalities_data, None
        elif self.mode == 'test':
            return modalities_data
    
    def __getitem__(self, idx):
        subject_name = self.subject_list[idx]

        # Load the data and extra info.
        if self.mode == 'train':
            imgs, seg = self.load_subject_data(subject_name)
        elif self.mode == 'val':
            result = self.load_subject_data(subject_name)
            if len(result) == 2:
                imgs, seg = result
            else:
                imgs = result
                seg = None
        elif self.mode == 'test':
            imgs = self.load_subject_data(subject_name)

        # Do Z-score norm and rescaling preprocessing.
        imgs = [znorm_rescale(img) for img in imgs]

        # Perform center crop.
        imgs = [center_crop(img) for img in imgs]

        if self.mode in ('train', 'val') and seg is not None:
            seg = center_crop(seg)

        # Stack all 4 modalities into a single (4, H, W, D) array for augmentation.
        imgs_stacked = np.stack(imgs, axis=0).astype(np.float32)  # (4, H, W, D)

        # Apply training augmentations.
        if self.mode == 'train':
            seg = seg[None, ...].astype(np.float32)  # (1, H, W, D)
            data_dict = {'image': imgs_stacked, 'seg': seg}
            data_dict = self.train_transforms(data_dict)
            imgs_stacked = data_dict['image']
            seg = data_dict['seg']

            # Convert to torch tensors.
            if not isinstance(imgs_stacked, torch.Tensor):
                imgs_stacked = torch.from_numpy(np.ascontiguousarray(imgs_stacked))
            if not isinstance(seg, torch.Tensor):
                seg = torch.from_numpy(np.ascontiguousarray(seg))

            # Split back into list of individual modality tensors (each 1HWD).
            imgs_out = [imgs_stacked[i:i+1] for i in range(4)]
            return subject_name, imgs_out, seg

        elif self.mode == 'val':
            imgs_out = [torch.from_numpy(np.ascontiguousarray(imgs_stacked[i:i+1])) for i in range(4)]
            if seg is not None:
                seg = seg[None, ...]
                seg = np.ascontiguousarray(seg)
                seg = torch.from_numpy(seg)
                return subject_name, imgs_out, seg
            else:
                return subject_name, imgs_out, None

        elif self.mode == 'test':
            imgs_out = [torch.from_numpy(np.ascontiguousarray(imgs_stacked[i:i+1])) for i in range(4)]
            return subject_name, imgs_out
