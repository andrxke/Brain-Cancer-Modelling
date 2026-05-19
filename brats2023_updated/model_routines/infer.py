import os
import torch
from datetime import date
from itertools import product

from ..utils.model_utils import make_dataloader
from ..utils.general_utils import probs_to_preds, save_pred_as_nifti


def _tta_flip_predict(model, x_in, training_regions):
    """Runs inference with test-time augmentation using all 8 flip combinations.
    
    Flips the input along each combination of spatial axes (0, 1, 2 of the HWD dims),
    runs inference, flips the output back, and averages all 8 sigmoid predictions.
    
    Args:
        model: Trained model.
        x_in: Input tensor of shape (B, 4, H, W, D).
        training_regions: Whether 'overlapping' or 'disjoint' regions were used for training.
    
    Returns:
        Averaged sigmoid probabilities of shape (B, 3, H, W, D).
    """
    # Spatial dims are 2, 3, 4 for a (B, C, H, W, D) tensor
    spatial_dims = [2, 3, 4]
    flip_combos = list(product([False, True], repeat=3))  # 8 combinations

    accumulated_probs = None

    for flip_flags in flip_combos:
        x_aug = x_in.clone()

        # Apply flips
        for dim_idx, should_flip in enumerate(flip_flags):
            if should_flip:
                x_aug = torch.flip(x_aug, dims=[spatial_dims[dim_idx]])

        # Run inference
        outputs = model(x_aug)
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs
        output = torch.sigmoid(output.float())

        # Un-flip the output
        for dim_idx, should_flip in enumerate(flip_flags):
            if should_flip:
                output = torch.flip(output, dims=[spatial_dims[dim_idx]])

        if accumulated_probs is None:
            accumulated_probs = output
        else:
            accumulated_probs += output

    # Average over all 8 combinations
    averaged_probs = accumulated_probs / len(flip_combos)
    return averaged_probs


def infer(data_dir, ckpt_path, out_dir=None, batch_size=1, postprocess_function=None, use_tta=True):
    """Uses trained model to make predictions on test data.
        To run this script from command line, use: 
        python -m brats2023_updated.model_routines.infer
        from the KurtBraTS directory.

    Args:
        data_dir: Directory of test data.
        ckpt_path: Path of trained model.
        out_dir: Directory in which to save predictions. Defaults to None.
        batch_size: Batch size of dataloader. Defaults to 1.
        postprocess_function: The postprocessing function to use. Defaults to None.
        use_tta: Whether to use test-time augmentation (8-flip averaging). Defaults to True.
    """

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    preds_dir = os.path.join(out_dir, 'preds' + str(date.today()))
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
        os.system(f'chmod a+rwx {preds_dir}')

    print(f"Loading model from {ckpt_path}...")
    # Modified weights_only to be false to load entire checkpoint dictionary due to PyTorch 2.6
    checkpoint = torch.load(ckpt_path, weights_only=False)

    model = checkpoint['model']
    loss_functions = checkpoint['loss_functions']
    loss_weights = checkpoint['loss_weights']
    training_regions = checkpoint['training_regions']

    epoch = checkpoint['epoch']
    model_sd = checkpoint['model_sd']

    model.load_state_dict(model_sd)

    print(f"Model loaded.")

    print("---------------------------------------------------")
    print(f"TRAINING SUMMARY")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Training regions: {training_regions}")
    print(f"Epochs trained: {epoch}")
    print("---------------------------------------------------")
    print("INFERENCE SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Trained model checkpoint path: {ckpt_path}")
    print(f"Out directory: {out_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Postprocess function: {postprocess_function}")
    print(f"Test-time augmentation: {use_tta}")
    print("---------------------------------------------------")

    test_loader = make_dataloader(data_dir, shuffle=False, mode='test', batch_size=batch_size)

    print('Inference starts.')
    with torch.no_grad():
        for subject_names, imgs in test_loader:

            model.eval()

            # Move data to GPU.
            imgs = [img.cuda() for img in imgs] # img is B1HWD

            x_in = torch.cat(imgs, dim=1) # x_in is B4HWD

            if use_tta:
                # TTA: average predictions across 8 flip combinations
                output = _tta_flip_predict(model, x_in, training_regions)
            else:
                outputs = model(x_in)
                if isinstance(outputs, list):
                    output = outputs[0]
                else:
                    output = outputs
                # Apply sigmoid manually as model now returns logits.
                output = torch.sigmoid(output.float())

            preds = probs_to_preds(output, training_regions)
            # preds is B3HWD - each channel is one-hot encoding of a disjoint region

            # Iterate over batch and save each prediction.
            for i, subject_name in enumerate(subject_names):
                save_pred_as_nifti(preds[i], preds_dir, data_dir, subject_name, postprocess_function)

    print(f'Inference completed. Predictions saved in {preds_dir}.')

if __name__ == '__main__':

    from ..processing.postprocess import rm_dust_fh

    # Directory of test data to run on
    data_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
    # Path of trained model
    ckpt_path = '/home/andrek/KurtBraTS/debug/train_with_vit/best_dice_ckpt.pth.tar'
    # Directory in which to save predictions
    out_dir = '/home/andrek/KurtBraTS/data/dataset/'

    postprocess_function = rm_dust_fh

    infer(data_dir, ckpt_path, out_dir=out_dir, postprocess_function=postprocess_function)