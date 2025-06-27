import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

from .models import get_model

logger = logging.getLogger(__name__)


class MathExpressionPredictor:
    """
    Class for making predictions on mathematical expression images.
    """
    
    def __init__(self, model_path: str, label_mapping_path: str, 
                 device: str = 'auto', image_size: int = 224):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            label_mapping_path: Path to the label mapping JSON file
            device: Device to use ('auto', 'cpu', 'cuda')
            image_size: Size to resize images to
        """
        self.image_size = image_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Predictor initialized on device: {self.device}")
        logger.info(f"Classes: {list(self.label_mapping.keys())}")
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model architecture from checkpoint
        # This is a simple heuristic - you might want to save model type in checkpoint
        num_classes = len(self.label_mapping)
        
        # Try to infer model type from state dict keys
        state_dict = checkpoint['model_state_dict']
        if 'layer1.0.conv1.weight' in state_dict:
            model_type = 'resnet'
        else:
            model_type = 'cnn'
        
        # Create model
        model = get_model(model_type, num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        logger.info(f"Loaded {model_type} model from {model_path}")
        return model
    
    def predict(self, image_path: str) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        # Get class name
        predicted_label = self.idx_to_label[predicted_class]
        
        # Get all class probabilities
        class_probabilities = {}
        for idx, prob in enumerate(probabilities[0]):
            class_name = self.idx_to_label[idx]
            class_probabilities[class_name] = prob.item()
        
        return {
            'predicted_class': predicted_label,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'predicted_class_idx': predicted_class
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_pil(self, image: Image.Image) -> Dict:
        """
        Make prediction from a PIL Image object.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        # Get class name
        predicted_label = self.idx_to_label[predicted_class]
        
        # Get all class probabilities
        class_probabilities = {}
        for idx, prob in enumerate(probabilities[0]):
            class_name = self.idx_to_label[idx]
            class_probabilities[class_name] = prob.item()
        
        return {
            'predicted_class': predicted_label,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'predicted_class_idx': predicted_class
        }
    
    def get_top_k_predictions(self, image_path: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for an image.
        
        Args:
            image_path: Path to the image file
            k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples
        """
        result = self.predict(image_path)
        probabilities = result['class_probabilities']
        
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), 
                            key=lambda x: x[1], reverse=True)
        
        return sorted_probs[:k]


def create_predictor_from_checkpoint(checkpoint_dir: str, device: str = 'auto') -> MathExpressionPredictor:
    """
    Convenience function to create a predictor from a checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing model checkpoint and label mapping
        device: Device to use
        
    Returns:
        Initialized MathExpressionPredictor
    """
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    label_mapping_path = os.path.join(checkpoint_dir, 'label_mapping.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(label_mapping_path):
        raise FileNotFoundError(f"Label mapping not found: {label_mapping_path}")
    
    return MathExpressionPredictor(model_path, label_mapping_path, device)


# Example usage
if __name__ == "__main__":
    # Example of how to use the predictor
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict mathematical expression class')
    parser.add_argument('--model-dir', type=str, required=True, 
                       help='Directory containing model checkpoint and label mapping')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to image to predict')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = create_predictor_from_checkpoint(args.model_dir, args.device)
    
    # Make prediction
    result = predictor.predict(args.image)
    
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nAll class probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.3f}") 