import torch
import torch.nn as nn
from brats2023_updated.sam.infer_sam3 import get_sam3_model_with_lora

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_sam3_model_with_lora(device, "brats2023_updated/sam/sam3_lora_best.pt")

cpu_count = 0
gpu_count = 0
for name, param in model.named_parameters():
    if param.device.type == "cpu":
        cpu_count += 1
    else:
        gpu_count += 1

print(f"Total params: {cpu_count + gpu_count}, CPU: {cpu_count}, GPU: {gpu_count}")
