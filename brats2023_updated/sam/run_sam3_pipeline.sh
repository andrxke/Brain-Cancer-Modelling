#!/bin/bash

# Change the directory to the project root for module execution
cd /home/andrek/KurtBraTS

echo "Starting training in the background tmux session..."
python -m brats2023_updated.sam.train_sam3

echo "Training completed! Starting inference..."
python -m brats2023_updated.sam.infer_sam3

echo "Pipeline finished!"
