import torch
from models.unet3d import U_Net3d
import sys

def verify():
    print("Initializing U_Net3d with ViT...")
    try:
        model = U_Net3d(img_ch=4, output_ch=3)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

    print("Model initialized successfully.")
    
    # Input shape: (Batch, Channel, D, H, W)
    # Based on preprocess.py: (128, 192, 128)
    # But unet3d usually expects (Batch, Channel, D, H, W) or (Batch, Channel, H, W, D)?
    # Let's assume (Batch, Channel, 128, 192, 128) matching the crop.
    # Note: The crop is (128, 192, 128) which is (X, Y, Z).
    # In torch, it's usually (D, H, W).
    # Let's try (1, 4, 128, 192, 128).
    
    input_tensor = torch.randn(1, 4, 128, 192, 128).cuda()
    print(f"Input shape: {input_tensor.shape}")
    
    try:
        print("Running forward pass...")
        outputs = model(input_tensor)
        if isinstance(outputs, list):
            print(f"Output is a list of length {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"Output {i} shape: {out.shape}")
            output = outputs[0]
        else:
            output = outputs
            print(f"Output shape: {output.shape}")
        
        expected_shape = (1, 3, 128, 192, 128)
        if output.shape == expected_shape:
            print("Verification SUCCESS: Main output shape matches expected shape.")
        else:
            print(f"Verification FAILED: Expected {expected_shape}, got {output.shape}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
