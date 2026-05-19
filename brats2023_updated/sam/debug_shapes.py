import torch
import sys
import os
import numpy as np

SAM3_DIR = os.path.join(os.getcwd(), 'sam3')
sys.path.append(SAM3_DIR)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cuda"
model = build_sam3_image_model(device=device, load_from_HF=False, checkpoint_path='sam3.pt') # using dummy? No, we have weights downloaded natively. Wait! I don't have sam3.pt. Let's just mock the shape.
