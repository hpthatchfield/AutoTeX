import matplotlib.pyplot as plt
import numpy as np
import os
from src.dataset import LateXImageDataset
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms


#### first test: check if dataset can be loaded
def test_dataset_loading():
    # Define paths
    csv_file = "data/labels.csv"
    image_dir = "data/images"
    
    # Check if files exist
    assert os.path.exists(csv_file), f"CSV file {csv_file} does not exist."
    assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."
    
    # Create a dataset instance
    dataset = LateXImageDataset(csv_file=csv_file, image_dir=image_dir)
    
    # Check length of dataset
    assert len(dataset) > 0, "Dataset is empty."
    
    # Check label mapping
    assert len(dataset.label_map) > 0, "Label mapping is empty."
    
    print("Dataset loaded successfully with the following label mapping:")
    print(dataset.label_map)

test_dataset_loading()