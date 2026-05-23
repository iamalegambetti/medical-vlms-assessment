#!/usr/bin/env python3
"""
extract_features.py — Extract and save image/text embeddings for a given model and dataset.

Usage (from workspace root):
    python re-congruence/src/scripts/cikm/repository/extract_features.py \
        --dataset MSCOCO2014 --model clip-large --split test

Supported datasets: MSCOCO2014, Flickr30k, ROCO, MIMIC-CXR
"""

import sys
import os
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_CARDS, get_device, get_dtype, DATA_PATH
from utils import make_collate_fn, load_model
from extractors import get_image_features, get_text_features
from datasets import MSCOCO14KarpathyDataset, Flickr30kDataset, ROCODataset, MIMICDataset

DEVICE = get_device()
DTYPE  = get_dtype(DEVICE)

SUPPORTED_DATASETS = ["MSCOCO2014", "Flickr30k", "ROCO", "MIMIC-CXR"]

parser = argparse.ArgumentParser(
    description="Extract and save image/text features for a given dataset."
)
parser.add_argument("--dataset", type=str, required=True,
                    choices=SUPPORTED_DATASETS, help="Dataset to process.")
parser.add_argument("--model", type=str, choices=MODEL_CARDS.keys(),
                    default="clip-large", help="Model to use for feature extraction.")
parser.add_argument("--output_dir", type=str,
                    default=str(Path(__file__).parent / "data"),
                    help="Directory to save extracted features (default: data/ next to this script).")
parser.add_argument("--split", type=str, choices=["train", "val", "test"],
                    default="test", help="Dataset split to process.")
args = parser.parse_args()

DATASET    = args.dataset
SPLIT      = args.split
MODEL_ID   = MODEL_CARDS[args.model]
OUTPUT_DIR = Path(args.output_dir) / args.model
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if DATASET == "MSCOCO2014":
    dataset = MSCOCO14KarpathyDataset(os.path.join(DATA_PATH, DATASET), split=SPLIT)
elif DATASET == "Flickr30k":
    dataset = Flickr30kDataset(os.path.join(DATA_PATH, DATASET), split=SPLIT)
elif DATASET == "ROCO":
    dataset = ROCODataset(os.path.join(DATA_PATH, DATASET), split=SPLIT)
elif DATASET == "MIMIC-CXR":
    dataset = MIMICDataset(os.path.join(DATA_PATH, DATASET), split=SPLIT)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")

model, processor = load_model(MODEL_ID, DTYPE, DEVICE)
collate_fn = make_collate_fn(processor, DEVICE)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

output = []
for batch in tqdm(dataloader, desc=f"Extracting features [{args.model} / {DATASET}]"):
    image_embeds = get_image_features(batch["inputs"], model, processor, DEVICE)
    text_embeds  = get_text_features(batch["inputs"], model, processor, DEVICE)
    image_embeds = image_embeds[0].view(1, -1)
    output.append({
        "image":    image_embeds.cpu(),
        "text":     text_embeds.cpu(),
        "label_ids": int(batch["label_ids"][0]),
    })

out_path = OUTPUT_DIR / f"{DATASET}_{SPLIT}_features.pt"
torch.save(output, out_path)
print(f"Saved {len(output)} samples → {out_path}")
