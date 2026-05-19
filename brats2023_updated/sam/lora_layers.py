"""
LoRA (Low-Rank Adaptation) implementation for SAM3 model fine-tuning.
Adapted from MedSAM3 (https://github.com/Joey-S-Liu/MedSAM3).
Supports selective application to different transformer components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class MultiheadAttentionLoRA(nn.Module):
    """
    Custom MultiheadAttention with separate Q, K, V projections so LoRA
    can be applied to each projection individually. Replaces nn.MultiheadAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        in_proj_weight: Optional[torch.Tensor] = None,
        in_proj_bias: Optional[torch.Tensor] = None,
        out_proj_weight: Optional[torch.Tensor] = None,
        out_proj_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if in_proj_weight is not None:
            self.q_proj.weight.data = in_proj_weight[:embed_dim, :].clone()
            self.k_proj.weight.data = in_proj_weight[embed_dim:2*embed_dim, :].clone()
            self.v_proj.weight.data = in_proj_weight[2*embed_dim:, :].clone()

        if in_proj_bias is not None:
            self.q_proj.bias.data = in_proj_bias[:embed_dim].clone()
            self.k_proj.bias.data = in_proj_bias[embed_dim:2*embed_dim].clone()
            self.v_proj.bias.data = in_proj_bias[2*embed_dim:].clone()

        if out_proj_weight is not None:
            self.out_proj.weight.data = out_proj_weight.clone()

        if out_proj_bias is not None:
            self.out_proj.bias.data = out_proj_bias.clone()

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.batch_first:
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            tgt_len, batch_size, _ = query.shape
            src_len = key.shape[0]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape[0] == batch_size:
                    attn_mask = attn_mask.unsqueeze(1)
                elif attn_mask.shape[0] == batch_size * self.num_heads:
                    attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
                else:
                    attn_mask = attn_mask.unsqueeze(1)

            if attn_mask.shape != attn_weights.shape:
                attn_mask = attn_mask.expand_as(attn_weights)

            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


class LoRALayer(nn.Module):
    """Low-rank adaptation layer: x @ (A @ B) * scaling"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16, dropout: float = 0.0, device=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros((in_features, rank), device=device))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features), device=device))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation: original(x) + lora(x)"""

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        device = original_layer.weight.device if hasattr(original_layer, 'weight') else None

        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank, alpha=alpha, dropout=dropout,
            device=device,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.original_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.lora(x)


class LoRAConfig:
    """Configuration for LoRA application to SAM3 model."""

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        apply_to_vision_encoder: bool = True,
        apply_to_text_encoder: bool = True,
        apply_to_geometry_encoder: bool = False,
        apply_to_detr_encoder: bool = True,
        apply_to_detr_decoder: bool = True,
        apply_to_mask_decoder: bool = False,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",
                "qkv", "proj",
                "fc1", "fc2",
                "c_fc", "c_proj",
                "linear1", "linear2",
            ]
        self.target_modules = set(target_modules)

        self.apply_to_vision_encoder = apply_to_vision_encoder
        self.apply_to_text_encoder = apply_to_text_encoder
        self.apply_to_geometry_encoder = apply_to_geometry_encoder
        self.apply_to_detr_encoder = apply_to_detr_encoder
        self.apply_to_detr_decoder = apply_to_detr_decoder
        self.apply_to_mask_decoder = apply_to_mask_decoder


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Apply LoRA to specified modules in the SAM3 model."""

    # Freeze all base model parameters first
    for param in model.parameters():
        param.requires_grad = False

    def should_apply_lora_to_component(module_name: str) -> bool:
        if ("vision_encoder" in module_name or "vision_backbone" in module_name) and not config.apply_to_vision_encoder:
            return False
        if ("text_encoder" in module_name or "language_backbone" in module_name) and not config.apply_to_text_encoder:
            return False
        if "geometry_encoder" in module_name and not config.apply_to_geometry_encoder:
            return False
        if ("detr_encoder" in module_name or "transformer.encoder" in module_name) and not config.apply_to_detr_encoder:
            return False
        if ("detr_decoder" in module_name or "transformer.decoder" in module_name) and not config.apply_to_detr_decoder:
            return False
        if "mask_decoder" in module_name and not config.apply_to_mask_decoder:
            return False
        return True

    def should_apply_lora(module_name: str) -> bool:
        if not should_apply_lora_to_component(module_name):
            return False
        module_basename = module_name.split('.')[-1]
        if module_basename in config.target_modules:
            return True
        for target in config.target_modules:
            if target in module_basename:
                return True
        return False

    mha_replaced = []
    lora_modules_applied = []

    # Step 1: Replace nn.MultiheadAttention with MultiheadAttentionLoRA
    mha_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if should_apply_lora_to_component(name):
                mha_to_replace.append((name, module))

    for name, mha in mha_to_replace:
        *parent_path, attr_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        new_mha = MultiheadAttentionLoRA(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            batch_first=mha.batch_first,
            in_proj_weight=mha.in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            out_proj_weight=mha.out_proj.weight,
            out_proj_bias=mha.out_proj.bias if mha.out_proj.bias is not None else None,
        )
        for param in new_mha.parameters():
            param.requires_grad = False

        setattr(parent, attr_name, new_mha)
        mha_replaced.append(name)

    print(f"Replaced {len(mha_replaced)} nn.MultiheadAttention modules with MultiheadAttentionLoRA")

    # Step 2: Apply LoRA to all matching Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            *parent_path, attr_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)

            lora_linear = LoRALinear(module, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
            setattr(parent, attr_name, lora_linear)
            lora_modules_applied.append(name)

    print(f"Applied LoRA to {len(lora_modules_applied)} modules:")
    for module_name in lora_modules_applied[:15]:
        print(f"  - {module_name}")
    if len(lora_modules_applied) > 15:
        print(f"  ... and {len(lora_modules_applied) - 15} more")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from the model."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def save_lora_weights(model: nn.Module, save_path: str):
    """Save only LoRA weights (not the full model)."""
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights ({len(lora_state_dict)} tensors) to {save_path}")


def load_lora_weights(model: nn.Module, load_path: str, device="cpu"):
    """Load LoRA weights into a model that already has LoRA applied."""
    lora_state_dict = torch.load(load_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    loaded = set(lora_state_dict.keys()) - set(unexpected)
    print(f"Loaded {len(loaded)} LoRA weights from {load_path}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    return model
