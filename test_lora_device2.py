import torch
import torch.nn as nn
from brats2023_updated.sam.lora_layers import LoRAConfig, apply_lora_to_model

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10)
        )
        self.q_proj = nn.Linear(10, 10)

model = DummyModel()
config = LoRAConfig(apply_to_vision_encoder=True)
model = apply_lora_to_model(model, config)
model.to("cuda")

for name, param in model.named_parameters():
    print(name, param.device)
