import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import v2

SAM3_DIR = os.path.join(os.path.dirname(__file__), 'sam3')
sys.path.append(SAM3_DIR)

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage, interpolate
from sam3.model.geometry_encoders import Prompt
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2

from .dataset import BraTSSAMDataset
from .lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters


# ---------------------------------------------------------------------------
# Text prompt for semantic grounding — SAM3 was designed for text-guided
# segmentation; passing a meaningful medical concept greatly improves
# query-to-object assignment versus the generic "visual" placeholder.
# ---------------------------------------------------------------------------
TEXT_PROMPT = "brain tumor"


def get_sam3_model(device):
    """Build SAM3 model with LoRA adapters for parameter-efficient fine-tuning."""
    model = build_sam3_image_model(device=str(device), load_from_HF=True, eval_mode=False)
    model = model.to(device)

    # Apply LoRA — freezes all base params and injects low-rank adapters
    lora_config = LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.1,
        apply_to_vision_encoder=True,
        apply_to_text_encoder=True,
        apply_to_geometry_encoder=True,
        apply_to_detr_encoder=True,
        apply_to_detr_decoder=True,
        apply_to_mask_decoder=False,
    )
    model = apply_lora_to_model(model, lora_config)
    model = model.to(device)  # Move LoRA parameters to GPU

    stats = count_parameters(model)
    print(f"Total Parameters:     {stats['total_parameters']:,}")
    print(f"Trainable Parameters: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")
    return model


def normalize_to_uint8(image):
    img_min = image.min()
    img_max = image.max()
    if img_max == img_min:
         return torch.zeros_like(image, dtype=torch.uint8)
    normalized = (image - img_min) / (img_max - img_min)
    return (normalized * 255.0).to(torch.uint8)


def build_loss():
    """Build the official SAM3 multi-component loss with Hungarian matching."""
    matcher = BinaryHungarianMatcherV2(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
    )

    loss_fns = [
        Boxes(weight_dict={
            "loss_bbox": 5.0,
            "loss_giou": 2.0
        }),
        IABCEMdetr(
            pos_weight=10.0,
            weight_dict={
                "loss_ce": 20.0,
                "presence_loss": 20.0
            },
            pos_focal=False,
            alpha=0.25,
            gamma=2,
            use_presence=True,
            pad_n_queries=200,
        ),
        Masks(
            weight_dict={
                "loss_mask": 200.0,
                "loss_dice": 10.0
            },
            focal_alpha=0.25,
            focal_gamma=2.0,
            compute_aux=False
        )
    ]

    loss_wrapper = Sam3LossWrapper(
        loss_fns_find=loss_fns,
        matcher=matcher,
        o2m_matcher=None,
        o2m_weight=0.0,
        normalization="local",
        normalize_by_valid_object_num=False,
    )
    return loss_wrapper, matcher


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Initializing Model...")
    model = get_sam3_model(device)
    model.train()

    # Hyperparameters — LoRA benefits from higher LR (adapters start at zero)
    epochs = 20
    warmup_steps = 200
    lr = 5e-5

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    loss_wrapper, matcher = build_loss()

    base_dir = '/home/andrek/KurtBraTS/data/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    dataset = BraTSSAMDataset(base_dir, jitter=20)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Cosine annealing LR scheduler with warmup
    total_steps = epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Transformations matching Sam3Processor
    resize = v2.Resize(size=(1008, 1008))
    norm = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    global_step = 0
    best_epoch_loss = float('inf')
    script_dir = os.path.dirname(__file__)

    print(f"Starting Training Loop — {epochs} epochs, lr={lr}, text_prompt='{TEXT_PROMPT}'")
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            image = batch["image"][0].to(device)
            gt_mask = batch["gt_mask"].to(device)    # [1, H, W]
            box = batch["box"].to(device)             # [1, 4]
            orig_hw = batch["orig_hw"]

            img_h, img_w = orig_hw[0].item(), orig_hw[1].item()

            # --- Image preprocessing ---
            image = image.permute(2, 0, 1)  # [3, H, W]
            image = normalize_to_uint8(image)
            image_f = image.float() / 255.0
            input_image = resize(image_f).unsqueeze(0).to(device)
            input_image = norm(input_image)

            # --- Box: convert [x1,y1,x2,y2] → normalized [cx,cy,w,h] ---
            x_min, y_min, x_max, y_max = box[0]
            center_x = (x_min + x_max) / 2.0 / img_w
            center_y = (y_min + y_max) / 2.0 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h
            sam3_box = torch.tensor(
                [[center_x, center_y, width, height]],
                device=device, dtype=torch.float32
            ).view(1, 1, 4)

            # --- Geometric prompt ---
            find_stage = FindStage(
                img_ids=torch.tensor([0], device=device, dtype=torch.long),
                text_ids=torch.tensor([0], device=device, dtype=torch.long),
                input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                input_points=None, input_points_mask=None,
            )

            geometric_prompt = Prompt(
                box_embeddings=sam3_box,
                box_mask=torch.zeros(1, 1, device=device, dtype=torch.bool),
                box_labels=torch.ones(1, 1, device=device, dtype=torch.long),
            )

            # --- Forward pass ---
            # Both backbone and decoder need consistent dtype under autocast.
            # LoRA adapters participate in the forward pass and receive gradients.
            # We call internal methods to avoid the automatic _compute_matching
            # guard in forward_grounding (which crashes when find_target=None).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    backbone_out = model.backbone.forward_image(input_image)
                    text_outputs = model.backbone.forward_text([TEXT_PROMPT], device=device)
                    backbone_out.update(text_outputs)

                # Encode prompt
                prompt, prompt_mask, backbone_out = model._encode_prompt(
                    backbone_out, find_stage, geometric_prompt
                )

                # Run encoder
                backbone_out, encoder_out, _ = model._run_encoder(
                    backbone_out, find_stage, prompt, prompt_mask
                )

                # Build intermediate output dict
                outputs = {
                    "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                    "prev_encoder_out": {
                        "encoder_out": encoder_out,
                        "backbone_out": backbone_out,
                    },
                }

                # Run decoder
                outputs, hs = model._run_decoder(
                    memory=outputs["encoder_hidden_states"],
                    pos_embed=encoder_out["pos_embed"],
                    src_mask=encoder_out["padding_mask"],
                    out=outputs,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    encoder_out=encoder_out,
                )

                # Run segmentation heads
                seg_img_ids = find_stage.img_ids
                if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                    seg_img_ids = backbone_out["id_mapping"][seg_img_ids]
                model._run_segmentation_heads(
                    out=outputs,
                    backbone_out=backbone_out,
                    img_ids=seg_img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=outputs["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hs,
                )

            # --- Prepare targets for the official SAM3 loss ---
            gt_box_norm = torch.tensor(
                [[center_x, center_y, width, height]],
                device=device, dtype=torch.float32
            )  # [1, 4]
            gt_box_xyxy = box_cxcywh_to_xyxy(gt_box_norm)  # [1, 4]

            # Resize GT mask to match pred_masks resolution
            pred_masks = outputs["pred_masks"]  # [B, num_queries, Hm, Wm]
            if len(pred_masks.shape) == 3:
                pred_masks = pred_masks.unsqueeze(0)

            mask_h, mask_w = pred_masks.shape[-2:]
            gt_mask_resized = torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0).float(),  # [1, 1, H, W]
                size=(mask_h, mask_w),
                mode="nearest"
            ).squeeze(0)  # [1, mask_h, mask_w]

            # Construct targets dict matching what back_convert() produces
            targets = {
                "boxes": gt_box_norm,                                    # [1, 4] CxCyWH
                "boxes_xyxy": gt_box_xyxy,                               # [1, 4] XYXY
                "boxes_padded": gt_box_norm.unsqueeze(0),                # [1, 1, 4]
                "num_boxes": torch.tensor([1], device=device),
                "is_exhaustive": torch.tensor([True], device=device),
                "masks": gt_mask_resized,                                # [1, Hm, Wm]
                "is_valid_mask": torch.tensor([True], device=device),
                "object_ids_packed": torch.tensor([0], device=device),
                "object_ids_padded": torch.tensor([[0]], device=device),  # [1, 1]
                "positive_map": torch.ones(1, 1, device=device),         # [1, 1]
            }

            # --- Compute matched indices via Hungarian matcher ---
            # The matcher returns (batch_idx, src_idx, tgt_idx)
            indices = matcher(outputs, targets)
            outputs["indices"] = indices

            # Also handle auxiliary outputs if present
            if "aux_outputs" in outputs:
                for aux_out in outputs["aux_outputs"]:
                    aux_out["indices"] = matcher(aux_out, targets)

            # --- Compute the full SAM3 loss ---
            num_boxes = torch.clamp(targets["num_boxes"].sum().float(), min=1)
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for loss_fn in loss_wrapper.loss_fns_find:
                l_dict = loss_fn(
                    outputs=outputs,
                    targets=targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    is_aux=False,
                )
                core = l_dict.pop(CORE_LOSS_KEY)
                total_loss = total_loss + core

            # Auxiliary outputs loss
            if "aux_outputs" in outputs:
                for i, aux_out in enumerate(outputs["aux_outputs"]):
                    aux_indices = aux_out["indices"]
                    for loss_fn in loss_wrapper.loss_fns_find:
                        l_dict = loss_fn(
                            outputs=aux_out,
                            targets=targets,
                            indices=aux_indices,
                            num_boxes=num_boxes,
                            is_aux=True,
                        )
                        core = l_dict.pop(CORE_LOSS_KEY)
                        total_loss = total_loss + core

            # --- Backward + optimize ---
            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += total_loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "LR": f"{current_lr:.2e}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save only best + last (disk space is limited)
        last_path = os.path.join(script_dir, "sam3_lora_last.pt")
        save_lora_weights(model, last_path)

        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            best_path = os.path.join(script_dir, "sam3_lora_best.pt")
            save_lora_weights(model, best_path)
            print(f"  ✓ New best model saved (avg_loss: {avg_loss:.4f})")

    print("Training Complete!")

if __name__ == "__main__":
    train()