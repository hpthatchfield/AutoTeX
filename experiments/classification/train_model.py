#!/usr/bin/env python3
"""
Training script for mathematical expression classification.
This script demonstrates the complete ML pipeline from data loading to model training.
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import LateXImageDataset
from src.models import get_model
from src.trainer import Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size=224):
    """Get data transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(csv_file, image_dir, batch_size=32, train_split=0.8, 
                       image_size=224, num_workers=2):
    """Create train and validation data loaders."""
    logger.info("Creating data loaders...")
    
    # Load full dataset
    full_dataset = LateXImageDataset(csv_file, image_dir)
    logger.info(f"Total dataset size: {len(full_dataset)}")
    logger.info(f"Classes: {list(full_dataset.label_map.keys())}")
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.label_map


def main():
    parser = argparse.ArgumentParser(description='Train mathematical expression classifier')
    parser.add_argument('--model', type=str, default='cnn', 
                       choices=['cnn', 'resnet'], help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/val split')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    csv_file = "data/labels.csv"
    image_dir = "data/images"
    
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return
    
    train_loader, val_loader, label_map = create_data_loaders(
        csv_file, image_dir, args.batch_size, args.train_split, args.image_size
    )
    
    # Create model
    num_classes = len(label_map)
    model = get_model(args.model, num_classes=num_classes)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(args.epochs, args.save_dir)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    val_loss, val_acc, predictions, targets = trainer.validate()
    
    # Print results
    logger.info(f"Final validation accuracy: {val_acc:.2f}%")
    logger.info(f"Best validation accuracy: {trainer.best_val_accuracy:.2f}%")
    
    # Plot training history
    trainer.plot_training_history(f"{args.save_dir}/training_history.png")
    
    # Plot confusion matrix
    class_names = list(label_map.keys())
    cm = trainer.plot_confusion_matrix(
        predictions, targets, class_names, 
        f"{args.save_dir}/confusion_matrix.png"
    )
    
    # Print classification report
    trainer.print_classification_report(predictions, targets, class_names)
    
    # Save label mapping
    with open(f"{args.save_dir}/label_mapping.json", 'w') as f:
        import json
        json.dump(label_map, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main() 