"""
utils.py — Model loading, data collation, and feature loading.
"""

import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


def _load_image(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    return Image.open(img).convert("RGB")


def load_model(model_id: str, dtype: torch.dtype, device: torch.device):
    model     = AutoModel.from_pretrained(model_id, dtype=dtype).to(device)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model.eval()
    return model, processor


def make_collate_fn(processor, device):
    """
    Returns a collate_fn for any HuggingFace CLIP-like processor.

    Each sample must be a dict with:
        "image"   — file path, raw bytes, or PIL Image
        "caption" — str or list[str]  (only the first caption is used)
    """
    # Cap max_length at 77 for models that leave the tokenizer's default at 1e9.
    _tok = getattr(processor, "tokenizer", processor)
    _max_length = getattr(_tok, "model_max_length", 77)
    if _max_length > 10_000:
        _max_length = 77

    def collate_fn(batch: list[dict]) -> dict:
        images, captions, label_ids = [], [], []
        for s in batch:
            if "caption" in s:
                caps = s["caption"] if isinstance(s["caption"], list) else [s["caption"]]
                label_id = "999"
            elif "label_name" in s:
                label = s["label_name"]
                labels = label if isinstance(label, list) else [label]
                caps = [f"a photo of a {l}" for l in labels]
                label_id = s.get("label_id")
            img = _load_image(s["image"])
            for cap in caps:
                images.append(img)
                captions.append(cap)
                label_ids.append(label_id)
        inputs = processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=_max_length,
        ).to(device)
        return {"inputs": inputs, "label_ids": label_ids}

    return collate_fn


def get_data(dataset: str, model: str, split: str, data_dir=None):
    """
    Load pre-extracted features saved by extract_features.py.

    By default, looks in a data/ subdirectory next to this file.
    Override with the DATA_DIR environment variable or the data_dir argument.
    """
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", str(Path(__file__).parent / "data"))
    path = Path(data_dir) / model / f"{dataset}_{split}_features.pt"
    return torch.load(path, weights_only=False)
