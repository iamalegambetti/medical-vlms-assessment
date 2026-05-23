"""
extractors.py — Image and text feature extraction from HuggingFace CLIP-like models.
"""

import torch
import torch.nn.functional as F

_IMAGE_KEYS = {"pixel_values"}
_TEXT_KEYS  = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}


@torch.no_grad()
def get_image_features(inputs_or_images, model, processor, device: torch.device) -> torch.Tensor:
    if isinstance(inputs_or_images, list):
        inputs = processor(images=inputs_or_images, return_tensors="pt").to(device)
    else:
        inputs = {k: v for k, v in inputs_or_images.items() if k in _IMAGE_KEYS}
    out   = model.get_image_features(**inputs)
    feats = out if isinstance(out, torch.Tensor) else out.pooler_output
    return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def get_text_features(inputs_or_texts, model, processor, device: torch.device) -> torch.Tensor:
    if isinstance(inputs_or_texts, list):
        inputs = processor(
            text=inputs_or_texts, return_tensors="pt",
            padding="max_length", max_length=64, truncation=True,
        ).to(device)
    else:
        inputs = {k: v for k, v in inputs_or_texts.items() if k in _TEXT_KEYS}
    out   = model.get_text_features(**inputs)
    feats = out if isinstance(out, torch.Tensor) else out.pooler_output
    return F.normalize(feats.float(), dim=-1)
