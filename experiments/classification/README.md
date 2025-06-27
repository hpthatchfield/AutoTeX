# Classification Experiments

This folder contains the baseline classification experiments that served as a stepping stone toward the main LaTeX generation goal.

## What's Here

- **models.py**: CNN and ResNet architectures for mathematical expression classification
- **trainer.py**: Training infrastructure for classification models
- **inference.py**: Inference pipeline for classification
- **train_model.py**: Training script for classification models
- **evaluate_model.py**: Evaluation script for classification models

## Purpose

The classification experiments demonstrate:
1. **Computer Vision Fundamentals**: CNN architectures and training
2. **Data Pipeline**: Image loading, preprocessing, and augmentation
3. **Model Evaluation**: Confusion matrices, per-class accuracy
4. **Software Engineering**: Clean code structure and testing

## Relationship to Main Project

The classification work serves as:
- **Feature Extractor**: The CNN encoder can be reused for the generation model
- **Baseline Performance**: Establishes what's achievable with classification
- **Learning Foundation**: Understanding of mathematical expression recognition

## Results

The classification model achieved:
- ~95% accuracy on 5 mathematical expression classes
- Good performance across integral, derivative, sum, fraction, and limit categories
- Robust to image variations (noise, rotation, background)

## Next Steps

The classification work has been completed and the focus has shifted to:
- **LaTeX Generation**: Sequence-to-sequence models
- **Attention Mechanisms**: Visual attention for better alignment
- **Evaluation Metrics**: BLEU score, exact match accuracy

See the main project README for the current focus on LaTeX generation. 