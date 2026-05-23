"""
config.py — Model registry and device helpers.

Set the DATA_PATH environment variable to point to your dataset root:
    export DATA_PATH=/path/to/your/data
"""

import os
import torch

# Root directory containing raw dataset folders (MSCOCO2014/, Flickr30k/, etc.)
DATA_PATH = os.environ.get("DATA_PATH", "/Volumes/server-data/data")

MODEL_CARDS = {
    "clip-base":        "openai/clip-vit-base-patch32",
    "clip-large":       "openai/clip-vit-large-patch14",

    "siglip-base":      "google/siglip-base-patch16-224",
    "siglip-large":     "google/siglip-so400m-patch14-384",

    "siglip2-base":     "google/siglip2-base-patch16-224",
    "siglip2-large":    "google/siglip2-so400m-patch14-384",

    "align":            "kakaobrain/align-base",

    "meta-clip-base":   "facebook/metaclip-b32-400m",
    "meta-clip-large":  "facebook/metaclip-h14-fullcc2.5b",

    "meta-clip2-base":  "facebook/metaclip-2-mt5-worldwide-b32",
    "meta-clip2-large": "facebook/metaclip-2-worldwide-huge-378",

    "laion-base":       "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion-large":      "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",

    "pubmed-clip-base": "flaviagiammarino/pubmed-clip-vit-base-patch32",
    "plip":             "vinid/plip",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on CUDA; float32 elsewhere (MPS does not support bfloat16)."""
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32
