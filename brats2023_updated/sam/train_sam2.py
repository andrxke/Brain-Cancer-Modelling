import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

SAM2_DIR = os.path.join(os.path.dirname(__file__), 'segment-anything-2')
sys.path.append(SAM2_DIR)

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from dataset import BraTSSAMDataset

# Custom mixed loss function
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # BCE
        loss_bce = self.bce(logits, targets)
        
        # Dice Loss on Sigmoid outputs
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(1,2))
        cardinality = probs.sum(dim=(1,2)) + targets.sum(dim=(1,2))
        dice_score = (2. * intersection + smooth) / (cardinality + smooth)
        loss_dice = 1.0 - dice_score.mean()
        
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice

def get_sam2_model(device, freeze_backbone=True):
    checkpoint = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    os.chdir(SAM2_DIR)
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    os.chdir(os.path.dirname(__file__))
    
    # Freeze the image encoder and prompt encoder
    if freeze_backbone:
        for param in sam2_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
            
    # Always keep mask_decoder trainable
    for param in sam2_model.sam_mask_decoder.parameters():
        param.requires_grad = True

    model_parameters = filter(lambda p: p.requires_grad, sam2_model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print(f"Trainable Parameters: {params:,}")
    return sam2_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optional TF32 optimizations
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("Initializing Model...")
    model = get_sam2_model(device, freeze_backbone=True)
    model.train()
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    loss_fn = CombinedLoss()
    
    # Dataset config
    base_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    dataset = BraTSSAMDataset(base_dir, jitter=10)
    # Using batch size of 1 because 3D files are heavy and image padding/scaling gets tricky with batching
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)  
    
    transforms = SAM2Transforms(resolution=1024, mask_threshold=0.0)

    epochs = 3
    print("Starting Training Loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            image = batch["image"][0].numpy() # [H, W, 3] np
            gt_mask = batch["gt_mask"].to(device) # [B, H, W]
            box = batch["box"].to(device) # [B, 4]
            orig_hw = batch["orig_hw"]
            
            orig_hw_tuple = (orig_hw[0].item(), orig_hw[1].item())
            
            # Use SAM2 transforms
            input_image = transforms(image).unsqueeze(0).to(device) # [1, 3, 1024, 1024]
            unnorm_box = transforms.transform_boxes(box, normalize=True, orig_hw=orig_hw_tuple) # [1, 2, 2]
            
            # Forward Image Encoding (No Grad for memory)
            with torch.no_grad():
                backbone_out = model.forward_image(input_image)
                _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
                if model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

                bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
                ][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]
            
            # Forward Prompts (No Grad)
            with torch.no_grad():
                box_coords = unnorm_box
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device).repeat(1, 1)
                concat_points = (box_coords, box_labels)

                sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                    points=concat_points,
                    boxes=None,
                    masks=None,
                )
            
            # Mask Decoder Forward (GRADIENTS ON)
            low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
            
            # Upscale mask to original resolution
            masks = transforms.postprocess_masks(low_res_masks, orig_hw_tuple) # [1, 1, H, W]
            
            # Calculate loss (resize ground truth to match predictions)
            masks = masks.squeeze(1) # [1, H, W]

            loss = loss_fn(masks, gt_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save placeholder checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"sam2_finetuned_epoch{epoch+1}.pt")
        
    print("Training Complete!")

if __name__ == "__main__":
    train()
