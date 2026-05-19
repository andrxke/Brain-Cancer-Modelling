import torch
import torch.nn as nn
from brats2023_updated.sam.infer_sam3 import get_sam3_model_with_lora

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_sam3_model_with_lora(device, "brats2023_updated/sam/sam3_lora_best.pt")

cpu_params = []
for name, param in model.named_parameters():
    if param.device.type == "cpu":
        cpu_params.append(name)

print("CPU params:", len(cpu_params))
if cpu_params:
    print(cpu_params[:10])
