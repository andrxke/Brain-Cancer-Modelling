
import torch
import sys
import os

# Add parent directory to path to import models
sys.path.append('/home/andrek/KurtBraTS/brats2023_updated')

from models.unet3d import U_Net3d

def check_model():
    print("Initializing model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = U_Net3d(img_ch=4, output_ch=3).to(device)
    
    # Check if ViT exists and has parameters
    print(f"ViT params: {sum(p.numel() for p in model.ViT.parameters())}")
    
    # Create dummy input
    # Assuming input size of (128, 192, 128) based on comments
    x = torch.randn(1, 4, 128, 192, 128).to(device)
    
    print("Running forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Compute loss
    loss = output.mean()
    print(f"Loss: {loss.item()}")
    
    print("Running backward pass...")
    loss.backward()
    
    # Check gradients
    vit_grads = []
    for name, param in model.ViT.named_parameters():
        if param.grad is not None:
            vit_grads.append(param.grad.abs().mean().item())
        else:
            print(f"No grad for {name}")
            
    if len(vit_grads) > 0:
        print(f"ViT Average Gradient: {sum(vit_grads)/len(vit_grads)}")
        print("ViT is receiving gradients.")
    else:
        print("ViT is NOT receiving gradients.")

if __name__ == "__main__":
    check_model()
