import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image   
import os  
from torchvision import transforms


class LateXImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        unique_labels = sorted(self.labels_frame['label'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_map.items()}

        print(f"Label mapping: {self.label_map}")

    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, idx):
        row = self.labels_frame.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['label']]
        if self.transform:
            image = self.transform(image)
        return image, label