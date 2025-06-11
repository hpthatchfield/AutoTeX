This repository is a project created by H Perry Hatchfield

# AutoTeX

A machine learning pipeline to classify mathematical expressions (printed or handwritten) into categories like integrals, derivatives, matrices, and more—serving as a foundation for LaTeX string prediction and document understanding.

## Project Goals

- Detect and classify visual math expressions using deep learning
- Demonstrate end-to-end ML engineering: data generation, preprocessing, training, and deployment
- Build toward LaTeX OCR automation with explainable, modular components

## Why This Project?

Academic and research documents are rich in mathematical content, but understanding math structure from images remains a challenge. 
This project explores a path toward automating that process—starting with classification as a stepping stone to full LaTeX generation.

## Project Structure
autotex/
├── data/ # Raw and processed datasets
├── notebooks/ # Interactive dev notebooks
├── src/ # Core code (models, training, inference)
├── tests/ # Unit tests and evaluation scripts
├── README.md
├── requirements.txt
└── .gitignore

## Tech Stack

- Python 3.x
- PyTorch / torchvision
- Matplotlib (for LaTeX image rendering)
- OpenCV (for preprocessing)
- Jupyter (for rapid dev and demo)

## Future Directions

- Extend from classification → sequence prediction (LaTeX code)
- Add saliency/explainability tools (e.g. Grad-CAM)
- Train on real handwritten math (e.g. CROHME dataset)

## Contributions

Currently in active development. Designed to showcase ML engineering ability, development domain understanding, and documentation clarity.

## License

MIT