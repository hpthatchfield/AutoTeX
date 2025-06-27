# AutoTeX: LaTeX Code Generation from Mathematical Images

A machine learning project that generates LaTeX code from images of mathematical expressions. This project demonstrates learning progression from classification to sequence generation.

## Project Goals

- **Primary Goal**: Learn and implement sequence-to-sequence models for LaTeX generation
- **Secondary Goal**: Demonstrate understanding of computer vision and natural language generation
- **Learning Focus**: Build a working system step by step, starting simple and improving

## Learning Approach

### Why Start Simple?

1. **Focus on Fundamentals**: Learn core concepts before adding complexity
2. **Easier Debugging**: Simpler models are easier to understand and fix
3. **Progressive Learning**: Build confidence with basic implementation first
4. **Realistic Timeline**: Achievable goals for a learning project

### Model Architecture (Simple Version)

```
Image → CNN Encoder → Feature Vector → LSTM Decoder → LaTeX Tokens
```

**Components:**
- **CNN Encoder**: 3-4 convolutional layers to extract image features
- **LSTM Decoder**: 1-2 LSTM layers to generate LaTeX tokens
- **Teacher Forcing**: Use ground truth during training
- **Greedy Decoding**: Simple token-by-token generation

## File Structure

```
AutoTeX/
├── src/simple_latex_model.py    # Simple CNN+LSTM model (main focus)
├── scripts/train_simple_latex.py # Simple training script
├── experiments/classification/   # Baseline classifier (completed)
└── data/                        # Generated dataset
```

## Quick Start

```bash
# Generate dataset
python scripts/generate_dataset.py

# Train simple model
python scripts/train_simple_latex.py --epochs 30
```

## Learning Progression

### Current Focus (Simple Implementation)
- Basic CNN + LSTM architecture
- Simple tokenization (space-separated)
- Teacher forcing training
- Greedy decoding inference
- Basic accuracy evaluation

### Future Improvements (After Basic Version Works)
- Add attention mechanism
- Implement beam search
- Use proper LaTeX tokenization
- Add BLEU score evaluation

## Why This Approach?

1. **Shows Learning**: Demonstrates progression from simple to complex
2. **Realistic Goals**: Achievable for someone learning generative models
3. **Debugging Friendly**: Simple enough to understand and fix issues
4. **Foundation Building**: Creates base for more advanced features

## License

MIT

## Author

H Perry Hatchfield

---

*This project demonstrates learning progression in machine learning, from basic classification to sequence generation, with a focus on understanding fundamentals before adding complexity.*
