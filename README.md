## Brain Cancer Modelling URA 
- Code from https://github.com/KurtLabUW/brats2023_updated?tab=readme-ov-file
- Using BraTS 2023 Data from Synapse https://www.synapse.org/Synapse:syn51156910/files/

## Reproducing Results

This repository contains pipelines for 3D tumor segmentation using U-Net and boundary refinement using the Segment Anything Model 3 (SAM 3). Below are instructions to run the models from scratch.

### Prerequisites

Ensure you have installed the required dependencies:
```bash
pip install -r brats2023_updated/requirements.txt
```
*Note: Make sure your dataset and checkpoint paths in the `__main__` blocks of the respective training/inference scripts are pointing to the correct locations on your local machine.*

### U-Net Model

**1. Training**
To train the base 3D U-Net model on the BraTS dataset, execute the following from the root directory:
```bash
python -m brats2023_updated.model_routines.train_with_val
```

**2. Inference**
Once trained, to run inference (predicting on validation/test data) with the U-Net model, run:
```bash
python -m brats2023_updated.model_routines.infer
```

### SAM 3 Model (Fine-Tuning & Inference)

The SAM 3 pipeline utilizes a fine-tuned SAM 3 model for slice-by-slice boundary refinement.

**Option 1: Complete Pipeline Script**
You can run the full SAM 3 pipeline (training followed by inference) using the provided bash script:
```bash
./brats2023_updated/sam/run_sam3_pipeline.sh
```

**Option 2: Manual Execution**

**1. Training (LoRA Fine-tuning)**
To fine-tune the SAM 3 model using LoRA adapters:
```bash
python -m brats2023_updated.sam.train_sam3
```

**2. Inference**
To evaluate the fine-tuned SAM 3 model (this uses bounding boxes to prompt the model):
```bash
python -m brats2023_updated.sam.infer_sam3
```
