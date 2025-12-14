import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

# scripts/dataset.py

class UnlabeledDataset(Dataset):
    def __init__(self, img_dir, transform=None, csv_files=None):
        """
        Args:
            img_dir (str): Path to image folder.
            transform (callable, optional): Augmentations.
            csv_files (dict, optional): If provided, only use images listed in these CSVs.
                                        Format: {'label': 'path/to/file.csv'}
        """
        self.img_dir = img_dir
        self.transform = transform
        

        if csv_files:
            self.images = []
            print(f"Filtering SSL dataset based on provided CSVs...")
            for _, csv_file in csv_files.items():
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, header=None, names=["id"])

                    for img_id in df["id"].astype(str):
                        for ext in ['.jpg', '.png', '.bmp', '.jpeg']:
                            filename = f"{img_id}{ext}"
                            if os.path.exists(os.path.join(img_dir, filename)):
                                self.images.append(filename)
                                break
            print(f"SSL Dataset restricted to {len(self.images)} training images.")
        

        else:
            self.images = [f for f in os.listdir(img_dir) if f.endswith(('jpg','png','jpeg','bmp'))]
            print(f"SSL Dataset loaded all {len(self.images)} images found in folder.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.images[idx])
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:

            return self.__getitem__((idx + 1) % len(self))

class LabeledDataset(Dataset):
    def __init__(self, img_dir, csv_files, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        data = []
        for label, csv_file in csv_files.items():
            df = pd.read_csv(csv_file, header=None, names=["id"])
            for img_id in df["id"].astype(str):
                filename = f"{img_id}.jpg"
                if os.path.exists(os.path.join(img_dir, filename)):
                    data.append((filename, label))
        self.data = data
        self.label2idx = {l: i for i, l in enumerate({lbl for _, lbl in data})}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.label2idx[label])