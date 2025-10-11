#!/usr/bin/env python3
"""
Script to create a validation split from training data.
Randomly selects 10% of training subjects and creates symlinks in a new validation directory.
"""

import os
import shutil
from sklearn.model_selection import train_test_split

def create_validation_split(train_dir, val_dir, test_size=0.1, random_state=42):
    """
    Create validation split by symlinking random subset of training data.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path where validation directory will be created
        test_size: Fraction of data to use for validation (default: 0.1 for 10%)
        random_state: Random seed for reproducible splits (default: 42)
    """
    
    # Get all subject directories from training data
    print(f"Reading subjects from: {train_dir}")
    all_subjects = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Found {len(all_subjects)} total subjects")
    
    # Randomly split into training and validation
    train_subjects, val_subjects = train_test_split(
        all_subjects, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Split: {len(train_subjects)} training, {len(val_subjects)} validation")
    
    # Create validation directory if it doesn't exist
    if os.path.exists(val_dir):
        print(f"Validation directory already exists: {val_dir}")
        response = input("Remove existing directory? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(val_dir)
        else:
            print("Aborted.")
            return
    
    os.makedirs(val_dir)
    print(f"Created validation directory: {val_dir}")
    
    # Create symlinks for validation subjects
    for subject in val_subjects:
        src_path = os.path.join(train_dir, subject)
        dst_path = os.path.join(val_dir, subject)
        os.symlink(src_path, dst_path)
        print(f"Created symlink: {subject}")
    
    print(f"\nValidation split created successfully!")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Validation directory: {val_dir}")
    
    # Save subject lists for reference
    train_list_file = os.path.join(os.path.dirname(val_dir), 'train_subjects.txt')
    val_list_file = os.path.join(os.path.dirname(val_dir), 'val_subjects.txt')
    
    with open(train_list_file, 'w') as f:
        f.write('\n'.join(sorted(train_subjects)))
    
    with open(val_list_file, 'w') as f:
        f.write('\n'.join(sorted(val_subjects)))
    
    print(f"Subject lists saved to: {train_list_file}, {val_list_file}")

if __name__ == '__main__':
    # Configuration
    train_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    val_dir = '/home/andrek/KurtBraTS/data/dataset/validation_split'
    
    create_validation_split(train_dir, val_dir)
