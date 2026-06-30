import os
import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(mode="train"):
    transforms = [A.Resize(224, 224)]
    if mode == "train":
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=10, p=0.3),
        ]
    transforms += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def build_image_lookup(image_base_dir):
    """Scan all images_XXX/images/ subfolders and return filename -> full_path dict."""
    lookup = {}
    for folder in os.listdir(image_base_dir):
        if folder.startswith('images_'):
            # nested path: images_003/images/*.png
            nested = os.path.join(image_base_dir, folder, 'images')
            if os.path.exists(nested):
                for f in os.listdir(nested):
                    if f.endswith('.png'):
                        lookup[f] = os.path.join(nested, f)
    return lookup


class ChestXray14Dataset(Dataset):
    def __init__(self, csv_path, image_lookup, transform=None, mode="train"):
        self.mode = mode
        self.transform = transform if transform is not None else get_transforms(mode)
        self.image_lookup = image_lookup
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _encode_labels(self, finding_labels):
        vec = np.zeros(len(CLASSES), dtype=np.float32)
        if pd.isna(finding_labels):
            return vec
        for finding in finding_labels.split("|"):
            finding = finding.strip()
            if finding and finding != "No Finding" and finding in self.class_to_idx:
                vec[self.class_to_idx[finding]] = 1.0
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["Image Index"]
        full_path = self.image_lookup.get(filename)

        if full_path is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.imread(full_path)
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self._encode_labels(row["Finding Labels"])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.from_numpy(label)
