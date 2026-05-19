import os
import numpy as np
import torch 
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import csv
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.model_utils import load_or_initialize_training, make_dataloader, exp_decay_learning_rate, compute_loss, train_one_epoch
from ..utils.general_utils import seg_to_one_hot_channels, disjoint_to_overlapping, probs_to_preds

def train_with_val(train_data_dir, val_data_dir, model, loss_functions, loss_weights, init_lr, max_epoch, training_regions='overlapping', eval_regions='overlapping', out_dir=None, decay_rate=0.995, backup_interval=10, val_interval=10, batch_size=1, patience=20):
    """Runs training routine with validation on separate validation set.

    Args:
        train_data_dir: Directory of training data.
        val_data_dir: Directory of validation data.
        model: The PyTorch model to be trained.
        loss_functions: List of loss functions to be used for training.
        loss_weights: List of weights corresponding to each loss function.
        init_lr: Initial value of learning rate.
        max_epoch: Maximum number of epochs to train for.
        training_regions: Whether training on 'disjoint' or 'overlapping' regions. Defaults to 'overlapping'.
        eval_regions: Whether to evaluate on 'disjoint' or 'overlapping' regions. Defaults to 'overlapping'.
        out_dir: The directory to save model checkpoints and loss and metric values. Defaults to None.
        decay_rate: Rate at which to decay the learning rate. Defaults to 0.995.
        backup_interval: How often to save a backup checkpoint. Defaults to 10.
        val_interval: How often to perform validation. Defaults to 10.
        batch_size: Batch size of dataloader. Defaults to 1.
        patience: Number of validation intervals to wait for improvement before early stopping. Defaults to 20.
    """
    
    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    best_vloss_ckpt_path = os.path.join(out_dir, 'best_vloss_ckpt.pth.tar')
    best_dice_ckpt_path = os.path.join(out_dir, 'best_dice_ckpt.pth.tar')
    loss_and_metrics_path = os.path.join(out_dir, 'loss_and_metrics.csv')
    backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
    if not os.path.exists(backup_ckpts_dir):
        os.makedirs(backup_ckpts_dir)
        os.system(f'chmod a+rwx {backup_ckpts_dir}')

    # Write header of csv log file.
    eval_region_names = []
    if eval_regions == 'overlapping':
        eval_region_names = ['WT', 'TC', 'ET']
    elif eval_regions == 'disjoint':
        eval_region_names = ['NCR', 'ED', 'ET']
    dice_eval_region_names = [f'Dice {eval_region}' for eval_region in eval_region_names]
    
    # Check if we are resuming from a checkpoint and the log file exists
    if os.path.exists(latest_ckpt_path) and os.path.exists(loss_and_metrics_path):
        print(f"Resuming training. Appending to existing log file: {loss_and_metrics_path}")
    else:
        # If not resuming or log file missing, start fresh
        with open(loss_and_metrics_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Mean Dice'] + dice_eval_region_names)

    print("---------------------------------------------------")
    print(f"TRAINING WITH VALIDATION SUMMARY")
    print(f"Training data directory: {train_data_dir}")
    print(f"Validation data directory: {val_data_dir}")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Initial learning rate: {init_lr}")
    print(f"Max epochs: {max_epoch}")
    print(f"Training regions: {training_regions}")
    print(f"Evaluation regions: {eval_regions}")
    print(f"Out directory: {out_dir}")
    print(f"Decay rate: {decay_rate}")
    print(f"Backup interval: {backup_interval}")
    print(f"Validation interval: {val_interval}")
    print(f"Batch size: {batch_size}")
    print(f"Patience: {patience}")
    print("---------------------------------------------------")

    # Changed to AdamW for better regularization with ViT
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-5, amsgrad=True)
    
    # Cosine Annealing Scheduler
    # T_0: Number of iterations for the first restart.
    # T_mult: A factor increases T_i after a restart.
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Check if training for first time or continuing from a saved checkpoint.
    epoch_start, best_vloss, best_dice = load_or_initialize_training(model, optimizer, latest_ckpt_path, train_with_val=True)

    # Create dataloaders
    train_loader = make_dataloader(train_data_dir, batch_size=batch_size, shuffle=True, mode='train')
    val_loader = make_dataloader(val_data_dir, batch_size=batch_size, shuffle=False, mode='val')

    epochs_since_improvement = 0

    print('Training starts.')
    for epoch in range(epoch_start, max_epoch+1):
        print(f'Starting epoch {epoch}...')

        # Custom scheduler replaced by PyTorch scheduler
        # exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate)
        # Step the scheduler at the start of epoch (or end, depending on preference, but usually after optimizer step if it was per-batch, here it is per epoch)
        # CosineAnnealingWarmRestarts usually steps per batch or per epoch. Let's do per epoch.
        scheduler.step(epoch + epoch_start)

        average_epoch_loss = train_one_epoch(model, optimizer, train_loader, loss_functions, loss_weights, training_regions)

        # Report loss from the epoch.
        print(f'Epoch {epoch} completed. Average loss = {average_epoch_loss:.4f}.')

        update_vloss = False
        update_dice = False

        # Run validation loop.
        if epoch % val_interval == 0:
            print('Starting validation loop...')

            val_loss_vals = []

            # Recommend use MONAI metrics set-up for different metrics (Cumulative Iterative)
            dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

            # Validation block
            with torch.no_grad():
                for _, imgs, seg in val_loader:

                    model.eval()

                    # Move data to GPU.
                    imgs = [img.cuda() for img in imgs] # img is B1HWD
                    seg = seg.cuda()

                    # Split segmentation into 3 channels.
                    seg = seg_to_one_hot_channels(seg) # seg is B3HWD

                    if training_regions == 'overlapping':
                        seg_train = disjoint_to_overlapping(seg)
                        # seg_train is B3HWD - each channel is one-hot encoding of an overlapping region
                    elif training_regions == 'disjoint':
                        seg_train = seg
                        # seg_train is B3HWD - each channel is one-hot encoding of a disjoint region

                    x_in = torch.cat(imgs, dim=1) # x_in is B4HWD
                    outputs = model(x_in)
                    if isinstance(outputs, list):
                        output = outputs[0] # Take the final resolution output for validation
                    else:
                        output = outputs
                    
                    output = output.float()

                    # Compute weighted loss, summed across each training region.
                    val_loss = compute_loss(output, seg_train, loss_functions, loss_weights)
                    val_loss_vals.append(val_loss.detach().cpu())

                    # CHANGED: Apply sigmoid because model now returns logits.
                    # Conver the models' raw probabilities into hard predictions
                    preds = probs_to_preds(torch.sigmoid(output), training_regions)

                    if eval_regions == 'overlapping':
                        # eval_region_names = ['WT', 'TC', 'ET']
                        # Convert seg and pred to 3 channels corresponding to overlapping regions
                        seg_eval = disjoint_to_overlapping(seg)
                        preds_eval = disjoint_to_overlapping(preds)
                        
                    elif eval_regions == 'disjoint':
                        # eval_region_names = ['NCR', 'ED', 'ET']
                        # Convert seg and pred to 3 channels corresponding to disjoint regions
                        seg_eval = seg
                        preds_eval = preds

                    # seg_eval is B3HWD
                    # preds_eval is B3HWD

                    # Compute metrics between seg_eval and preds_eval.
                    dice_metric(y_pred = preds_eval, y=seg_eval)

            # Compute and report validation loss.
            average_val_loss = np.mean(val_loss_vals)
            print(f'Validation completed. Average validation loss = {average_val_loss}')

            # Aggregate and report the Dice scores.
            dice_metric_batch = dice_metric.aggregate()
            eval_region_dice_scores = []
            for i in range(3):
                eval_region_dice_scores.append(dice_metric_batch[i].item())
            mean_dice = np.mean(eval_region_dice_scores)

            if average_val_loss < best_vloss:
                best_vloss = average_val_loss
                update_vloss = True
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if mean_dice > best_dice:
                best_dice = mean_dice
                update_dice = True

            # Save training loss and validation loss and metrics.
            save_loss_and_metrics_csv(loss_and_metrics_path, epoch, average_epoch_loss, average_val_loss, mean_dice, eval_region_dice_scores)
            
            # Plot metrics
            plot_metrics(loss_and_metrics_path, out_dir)

            if epochs_since_improvement >= patience:
                print(f'Early stopping triggered. Validation loss has not improved for {patience} validation intervals.')
                break

        print('Saving model checkpoint...')
        checkpoint = {
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optim_sd': optimizer.state_dict(),
            'model': model,
            'loss_functions': loss_functions,
            'loss_weights': loss_weights,
            'init_lr': init_lr,
            'training_regions': training_regions,
            'decay_rate': decay_rate,
            'vloss': best_vloss,
            'dice': best_dice
        }
        torch.save(checkpoint, latest_ckpt_path)
        if epoch % backup_interval == 0:
            torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))
        if update_vloss:
            print('New best validation loss!')
            torch.save(checkpoint, best_vloss_ckpt_path)
        if update_dice:
            print('New best dice score!')
            torch.save(checkpoint, best_dice_ckpt_path)

        print('Checkpoint saved successfully.')

def save_loss_and_metrics_csv(pathname, epoch, tloss, vloss, mean_dice, eval_region_scores):
    with open(pathname, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, tloss, vloss, mean_dice] + eval_region_scores)

def plot_metrics(csv_path, out_dir):
    """Plots training metrics from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Losses
        ax1.plot(df['Epoch'], df['Training Loss'], label='Training Loss', marker='o')
        ax1.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Dice Scores
        # Get all columns that start with "Dice" or are "Mean Dice"
        dice_cols = [col for col in df.columns if 'Dice' in col]
        
        for col in dice_cols:
            ax2.plot(df['Epoch'], df[col], label=col, marker='o')
            
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Dice Scores')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_plots.png'))
        plt.close()
        print(f"Plots saved to {os.path.join(out_dir, 'training_plots.png')}")
    except Exception as e:
        print(f"Error plotting metrics: {e}")

if __name__ == '__main__':

    from ..models import unet3d
    import torch.nn as nn
    from monai.losses import HausdorffDTLoss

    train_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    val_dir = '/home/andrek/KurtBraTS/data/dataset/validation_split'  # Use the created validation split
    
    model = unet3d.U_Net3d()
    # CHANGED: Added HausdorffDTLoss as a third loss to improve boundary segmentation.
    loss_functions = [DiceLoss(include_background=True, sigmoid=True), nn.BCEWithLogitsLoss(), HausdorffDTLoss(sigmoid=True)]
    loss_weights = [1.0, 1.0, 0.5]
    # CHANGED: Lowered LR to 5e-5 for smoother convergence with data augmentation.
    lr = 5e-5
    max_epoch = 200
    val_interval = 5
    out_dir = '/home/andrek/KurtBraTS/debug/train_with_vit'

    # Train model and record time 
    start_time = time.time()
    train_with_val(train_dir, val_dir, model, loss_functions, loss_weights, lr, max_epoch, val_interval=val_interval, out_dir=out_dir)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")