"""
datasets.py — Dataset classes for the four retrieval benchmarks used in
the CIKM alignment experiments.

Datasets
  MSCOCO2014    Natural, 5 captions per image (Karpathy split)
  Flickr30k     Natural, 5 captions per image
  ROCO          Medical (radiology), 1 caption per image
  MIMIC-CXR     Medical (chest X-ray), 1 caption per image (findings + impression)
"""

import os
import json

import pandas as pd
from torch.utils.data import Dataset


class MSCOCO14KarpathyDataset(Dataset):
    """MSCOCO 2014 with Karpathy train/val/test splits. 5 captions per image."""

    def __init__(self, data_path, split="test"):
        self.data_path = data_path
        self.split = split

        annotations = os.path.join(data_path, "karpathy", "dataset.json")
        with open(annotations, "r") as f:
            raw = json.load(f)

        if split == "train":
            self.data = [img for img in raw["images"] if img["split"] == "train"]
            self.images_path = os.path.join(data_path, "train2014")
        elif split == "val":
            self.data = [img for img in raw["images"] if img["split"] == "val"]
            self.images_path = os.path.join(data_path, "val2014")
        elif split == "test":
            self.data = [img for img in raw["images"] if img["split"] == "test"]
            self.images_path = os.path.join(data_path, "val2014")
        else:
            raise ValueError(f"Invalid split '{split}'. Choose from: train, val, test.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.images_path, item["filename"])
        captions = [cap["raw"].strip() for cap in item["sentences"]]
        return {"image": image_path, "caption": captions}


class Flickr30kDataset(Dataset):
    """Flickr30k retrieval dataset. 5 captions per image."""

    def __init__(self, data_path, split="test"):
        self.data_path = data_path
        self.split = split
        self.data = pd.read_csv(os.path.join(data_path, "flickr_annotations_30k.csv"))
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.data_path, "flickr30k-images", str(row["filename"]))
        captions = json.loads(row["raw"])
        return {"image": image_path, "caption": captions}


class ROCODataset(Dataset):
    """
    Radiology Objects in COntext (ROCO) dataset. 1 caption per image.
    Images are extracted from the parquet file to an 'images/' subfolder on first load.
    """

    def __init__(self, data_path, split="test"):
        self.data_path = data_path
        self.split = split
        self.images_path = os.path.join(data_path, "images")
        self.data = pd.read_parquet(os.path.join(data_path, "test-00000-of-00001.parquet"))
        self._extract_images()

    def _extract_images(self):
        os.makedirs(self.images_path, exist_ok=True)
        for _, row in self.data.iterrows():
            image_path = os.path.join(self.images_path, f"{row['image_id']}.jpg")
            if not os.path.exists(image_path):
                with open(image_path, "wb") as f:
                    f.write(row["image"]["bytes"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path = os.path.join(self.images_path, f"{item['image_id']}.jpg")
        caption = item["caption"].strip()
        return {"image": image_path, "caption": caption}


class MIMICDataset(Dataset):
    """
    MIMIC-CXR dataset. 1 caption per image (findings + impression concatenated).
    Images are extracted from parquet files to an 'images/' subfolder on first load.
    """

    def __init__(self, data_path, split="test"):
        self.data_path = data_path
        self.split = split
        self.images_path = os.path.join(data_path, "images")
        parquet_files = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".parquet")
        )
        self.data = pd.concat(
            [pd.read_parquet(f) for f in parquet_files], ignore_index=True
        )
        self._extract_images()

    def _extract_images(self):
        os.makedirs(self.images_path, exist_ok=True)
        for idx, row in self.data.iterrows():
            image_path = os.path.join(self.images_path, f"{idx}.jpg")
            if not os.path.exists(image_path):
                with open(image_path, "wb") as f:
                    f.write(row["image"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path = os.path.join(self.images_path, f"{idx}.jpg")
        findings   = item["findings"]   if isinstance(item["findings"],   str) else ""
        impression = item["impression"] if isinstance(item["impression"], str) else ""
        caption = " ".join([findings.strip(), impression.strip()]).strip()
        return {"image": image_path, "caption": caption}
