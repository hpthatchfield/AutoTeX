import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.render import render_latex_to_image

# Example LaTeX strings per category
examples = {
    "integral": [r"\int x^2 dx", r"\int_0^\infty e^{-x} dx"],
    "derivative": [r"\frac{d}{dx} x^3", r"\frac{\partial^2 u}{\partial t^2}"],
    "sum": [r"\sum_{n=1}^\infty \frac{1}{n^2}"],
}

output_dir = "data/images"
os.makedirs(output_dir, exist_ok=True)

records = []

for category, formulas in examples.items():
    for i, latex in enumerate(formulas):
        filename = f"{category}_{i}.png"
        output_path = os.path.join(output_dir, filename)
        try:
            render_latex_to_image(latex, output_path)
            records.append({"filename": filename, "label": category, "latex": latex})
        except Exception as e:
            print(f"Error rendering {latex}: {e}")

# Save labels to CSV
df = pd.DataFrame(records)
df.to_csv("data/labels.csv", index=False)