#!/usr/bin/env python3
"""
Evaluation script for mathematical expression classification models.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import create_predictor_from_checkpoint
from src.dataset import LateXImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(checkpoint_dir: str, test_csv: str, test_image_dir: str, 
                  device: str = 'auto', batch_size: int = 32):
    """
    Evaluate a trained model on test data.
    
    Args:
        checkpoint_dir: Directory containing model checkpoint and label mapping
        test_csv: Path to test CSV file
        test_image_dir: Path to test image directory
        device: Device to use
        batch_size: Batch size for evaluation
    """
    logger.info(f"Evaluating model from {checkpoint_dir}")
    
    # Create predictor
    predictor = create_predictor_from_checkpoint(checkpoint_dir, device)
    
    # Load test dataset
    test_dataset = LateXImageDataset(test_csv, test_image_dir)
    
    # Set up transforms (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    test_dataset.transform = transform
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    logger.info("Running evaluation...")
    for i, (images, targets) in enumerate(test_loader):
        # Convert to PIL images for prediction
        for j, (image, target) in enumerate(zip(images, targets)):
            # Convert tensor back to PIL
            image_pil = transforms.ToPILImage()(image)
            
            # Make prediction
            result = predictor.predict_from_pil(image_pil)
            
            all_predictions.append(result['predicted_class_idx'])
            all_targets.append(target.item())
            all_confidences.append(result['confidence'])
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    confidences = np.array(all_confidences)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    
    # Get class names
    class_names = list(predictor.label_mapping.keys())
    
    # Print results
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Average Confidence: {np.mean(confidences):.4f}")
    
    # Classification report
    report = classification_report(targets, predictions, 
                                 target_names=class_names, digits=3)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{checkpoint_dir}/test_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_acc_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': per_class_accuracy,
        'Support': cm.sum(axis=1)
    })
    
    print("\nPer-Class Accuracy:")
    print(class_acc_df.to_string(index=False))
    
    # Confidence analysis
    confidence_df = pd.DataFrame({
        'True_Class': [class_names[t] for t in targets],
        'Predicted_Class': [class_names[p] for p in predictions],
        'Confidence': confidences,
        'Correct': targets == predictions
    })
    
    print("\nConfidence Analysis:")
    print(f"Correct predictions - Mean confidence: {confidence_df[confidence_df['Correct']]['Confidence'].mean():.3f}")
    print(f"Incorrect predictions - Mean confidence: {confidence_df[~confidence_df['Correct']]['Confidence'].mean():.3f}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'mean_confidence': float(np.mean(confidences)),
        'per_class_accuracy': {class_names[i]: float(acc) for i, acc in enumerate(per_class_accuracy)},
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'confidences': confidences.tolist()
    }
    
    with open(f"{checkpoint_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {checkpoint_dir}/evaluation_results.json")
    
    return results


def analyze_errors(checkpoint_dir: str, test_csv: str, test_image_dir: str, 
                  device: str = 'auto', num_examples: int = 10):
    """
    Analyze prediction errors to understand model weaknesses.
    """
    logger.info("Analyzing prediction errors...")
    
    # Create predictor
    predictor = create_predictor_from_checkpoint(checkpoint_dir, device)
    
    # Load test dataset
    test_dataset = LateXImageDataset(test_csv, test_image_dir)
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    test_dataset.transform = transform
    
    # Find errors
    errors = []
    for i in range(len(test_dataset)):
        image, target = test_dataset[i]
        image_pil = transforms.ToPILImage()(image)
        
        result = predictor.predict_from_pil(image_pil)
        predicted_class = result['predicted_class_idx']
        
        if predicted_class != target:
            errors.append({
                'index': i,
                'true_class': test_dataset.idx_to_label[target],
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'latex': test_dataset.labels_frame.iloc[i]['latex'],
                'complexity': test_dataset.labels_frame.iloc[i]['complexity']
            })
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"\nFound {len(errors)} errors out of {len(test_dataset)} samples")
    print(f"Error rate: {len(errors)/len(test_dataset):.3f}")
    
    # Show top errors
    print(f"\nTop {min(num_examples, len(errors))} most confident errors:")
    for i, error in enumerate(errors[:num_examples]):
        print(f"{i+1}. True: {error['true_class']}, Predicted: {error['predicted_class']}")
        print(f"   Confidence: {error['confidence']:.3f}")
        print(f"   LaTeX: {error['latex']}")
        print(f"   Complexity: {error['complexity']}")
        print()
    
    # Error analysis by class
    error_by_class = {}
    for error in errors:
        true_class = error['true_class']
        if true_class not in error_by_class:
            error_by_class[true_class] = []
        error_by_class[true_class].append(error)
    
    print("Errors by true class:")
    for class_name, class_errors in error_by_class.items():
        print(f"  {class_name}: {len(class_errors)} errors")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description='Evaluate mathematical expression classifier')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing model checkpoint and label mapping')
    parser.add_argument('--test-csv', type=str, default='data/labels.csv',
                       help='Path to test CSV file')
    parser.add_argument('--test-image-dir', type=str, default='data/images',
                       help='Path to test image directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--analyze-errors', action='store_true',
                       help='Analyze prediction errors')
    parser.add_argument('--num-error-examples', type=int, default=10,
                       help='Number of error examples to show')
    
    args = parser.parse_args()
    
    # Check if checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
        return
    
    # Run evaluation
    results = evaluate_model(
        args.checkpoint_dir,
        args.test_csv,
        args.test_image_dir,
        args.device,
        args.batch_size
    )
    
    # Analyze errors if requested
    if args.analyze_errors:
        errors = analyze_errors(
            args.checkpoint_dir,
            args.test_csv,
            args.test_image_dir,
            args.device,
            args.num_error_examples
        )


if __name__ == "__main__":
    main() 