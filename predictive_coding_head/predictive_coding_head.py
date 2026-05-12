"""Predictive coding head utilities for Qwen3-VL style training.

This module provides a small, self-contained implementation that can be
dropped into an existing HuggingFace training script. It assumes:

- The multimodal model exposes `get_image_features(...)`.
- The model config exposes `image_token_id`.
- Batch inputs contain:
  - `input_ids`
  - `pixel_values`
  - `image_grid_thw`
  - `num_images` (number of images per sample, in sequence order)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def resolve_multimodal_base_model(model: nn.Module) -> nn.Module:
    """Resolve the multimodal base model under plain model / PEFT wrappers."""
    candidates = [model]

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(get_base_model())
        except Exception:
            pass

    candidates.extend(
        [
            getattr(model, "base_model", None),
            getattr(getattr(model, "base_model", None), "model", None),
            getattr(model, "model", None),
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "get_image_features") and hasattr(candidate, "config"):
            return candidate

    raise TypeError(f"Unable to resolve multimodal base model from {type(model)!r}")


def init_predictive_coding_head(
    model: nn.Module,
    head_attr: str = "predictive_coding_head",
) -> None:
    """Attach a 2-layer MLP projection head to predict next image features."""
    base_model = resolve_multimodal_base_model(model)
    hidden_size = getattr(getattr(base_model.config, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(base_model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Unable to infer hidden size for predictive coding head")

    if getattr(base_model, head_attr, None) is not None:
        return

    param = next(base_model.parameters())
    setattr(
        base_model,
        head_attr,
        nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(device=param.device, dtype=param.dtype),
    )


def compute_predictive_coding_losses(
    model: nn.Module,
    inputs: dict,
    outputs,
    head_attr: str = "predictive_coding_head",
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute next-image prediction losses (MSE + cosine) over image tokens."""
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    num_images = inputs.get("num_images")
    input_ids = inputs.get("input_ids")

    if pixel_values is None or image_grid_thw is None or num_images is None or input_ids is None:
        return None, None

    if int(num_images.sum().item()) < 2:
        return None, None

    base_model = resolve_multimodal_base_model(model)
    projection_head = getattr(base_model, head_attr, None)
    if projection_head is None:
        raise ValueError(f"Model has no `{head_attr}`. Call init_predictive_coding_head first.")

    image_token_id = getattr(base_model.config, "image_token_id", None)
    if image_token_id is None:
        return None, None

    with torch.no_grad():
        image_outputs = base_model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        target_image_features = list(image_outputs.pooler_output)

    last_hidden = outputs.hidden_states[-1]
    image_mask = input_ids == image_token_id
    pred_image_tokens = last_hidden[image_mask]

    token_counts = [feat.shape[0] for feat in target_image_features]
    total_image_tokens = sum(token_counts)
    if pred_image_tokens.shape[0] != total_image_tokens:
        raise ValueError(
            f"Predictive coding token mismatch: predicted={pred_image_tokens.shape[0]} "
            f"target={total_image_tokens}"
        )

    pred_image_features = list(torch.split(pred_image_tokens, token_counts, dim=0))
    mse_terms = []
    cosine_terms = []
    image_offset = 0

    for sample_num_images in num_images.tolist():
        sample_pred_features = pred_image_features[image_offset : image_offset + sample_num_images]
        sample_target_features = target_image_features[image_offset : image_offset + sample_num_images]
        image_offset += sample_num_images

        for cur_pred_tokens, next_target_tokens in zip(sample_pred_features[:-1], sample_target_features[1:]):
            head_dtype = next(projection_head.parameters()).dtype
            predicted_next = projection_head(cur_pred_tokens.to(head_dtype))
            target_next = next_target_tokens.detach().to(predicted_next.device, predicted_next.dtype)

            if predicted_next.shape[0] != target_next.shape[0]:
                n = min(predicted_next.shape[0], target_next.shape[0])
                predicted_next = predicted_next[:n]
                target_next = target_next[:n]

            mse_terms.append(F.mse_loss(predicted_next.float(), target_next.float()))
            cosine_terms.append(
                1.0 - F.cosine_similarity(predicted_next.float(), target_next.float(), dim=-1).mean()
            )

    if not mse_terms:
        return None, None

    return torch.stack(mse_terms).mean(), torch.stack(cosine_terms).mean()


def combine_main_and_predictive_losses(
    ce_loss: torch.Tensor,
    mse_loss: Optional[torch.Tensor],
    cosine_loss: Optional[torch.Tensor],
    mse_weight: float = 0.1,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """Combine CE with predictive coding losses using scalar weights."""
    total_loss = ce_loss
    if mse_loss is not None and cosine_loss is not None:
        total_loss = total_loss + float(mse_weight) * mse_loss + float(cosine_weight) * cosine_loss
    return total_loss
