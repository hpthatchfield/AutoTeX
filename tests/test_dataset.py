import matplotlib.pyplot as plt
import numpy as np
import os
from src.dataset import LateXImageDataset
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from src.render import render_latex_to_image
import pytest
from PIL import Image


def test_dataset_loading():
    """Test if dataset can be loaded correctly."""
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


def test_image_generation():
    """Test image generation with different parameters."""
    test_expr = r"\frac{d}{dx} \sin(x^2)"
    output_path = "test_output.png"
    
    # Test basic rendering
    assert render_latex_to_image(test_expr, output_path), "Basic rendering failed"
    assert os.path.exists(output_path), "Output file not created"
    
    # Test with different parameters
    params = [
        {"font_size": 20, "add_noise": True},
        {"font_size": 16, "add_noise": False},
        {"background_color": "#f0f0f0", "rotation_range": (-1, 1)}
    ]
    
    for i, param in enumerate(params):
        test_path = f"test_output_{i}.png"
        assert render_latex_to_image(test_expr, test_path, **param), f"Rendering failed with params: {param}"
        assert os.path.exists(test_path), f"Output file not created for params: {param}"
        os.remove(test_path)
    
    # Cleanup
    os.remove(output_path)


def test_image_quality():
    """Test the quality of generated images."""
    test_expr = r"\frac{d}{dx} \sin(x^2)"
    output_path = "test_quality.png"
    
    # Generate image
    render_latex_to_image(test_expr, output_path)
    
    # Load and check image
    img = Image.open(output_path)
    
    # Check image properties
    assert img.mode == 'RGBA', "Image should be in RGBA mode"
    assert img.size[0] > 0 and img.size[1] > 0, "Image dimensions should be positive"
    
    # Check image content
    img_array = np.array(img)
    assert img_array.shape[2] == 4, "Image should have 4 channels (RGBA)"
    assert np.any(img_array[:, :, 3] > 0), "Image should have non-transparent pixels"
    
    # Cleanup
    os.remove(output_path)


def test_dataset_statistics():
    """Test dataset statistics and distributions."""
    csv_file = "data/labels.csv"
    image_dir = "data/images"
    
    # Load dataset
    dataset = LateXImageDataset(csv_file=csv_file, image_dir=image_dir)
    labels_df = pd.read_csv(csv_file)
    
    # Check basic statistics
    assert len(labels_df) == len(dataset), "Dataset length mismatch"
    assert all(cat in dataset.label_map for cat in labels_df['label'].unique()), "Missing categories in label map"
    
    # Check complexity distribution
    assert labels_df['complexity'].min() >= 0, "Negative complexity found"
    assert labels_df['complexity'].max() <= 3, "Complexity exceeds maximum level"
    
    # Check category distribution
    category_counts = labels_df['label'].value_counts()
    assert len(category_counts) == len(dataset.label_map), "Category count mismatch"
    assert all(count > 0 for count in category_counts), "Empty categories found"


if __name__ == "__main__":
    # Run all tests
    test_dataset_loading()
    test_image_generation()
    test_image_quality()
    test_dataset_statistics()
    print("All tests passed successfully!")