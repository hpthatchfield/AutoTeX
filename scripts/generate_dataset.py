import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.render import render_latex_to_image
from typing import List, Dict
import re


# Example LaTeX strings per category
# examples = {
#     "integral": [r"\int x^2 dx", r"\int_0^\infty e^{-x} dx"],
#     "derivative": [r"\frac{d}{dx} x^3", r"\frac{\partial^2 u}{\partial t^2}"],
#     "sum": [r"\sum_{n=1}^\infty \frac{1}{n^2}"],
# }

def generate_features(expr: str) -> str:
    expr_wrapped = f"{{{expr}}}"  # Always wrap input in braces to avoid invalid LaTeX syntax

    features = [
        f"{expr_wrapped}^2",
        f"e^{{-{expr_wrapped}}}",
        f"\\sin({expr_wrapped})",
        f"\\cos({expr_wrapped})",
        f"\\ln({expr_wrapped})",
        f"\\frac{{1}}{{{expr_wrapped}}}",
        f"\\sqrt{{{expr_wrapped}}}",
        f"\\frac{{d}}{{dx}} {expr_wrapped}",
        f"\\frac{{\\partial^2 {expr_wrapped}}}{{\\partial t^2}}",
        f"\\frac{{1}}{{{expr_wrapped}^2}}",
    ]
    return np.random.choice(features)



def generate_latex_examples(num_per_class: int, seed: int = 42) -> Dict[str, List[str]]:
    np.random.seed(seed)
    """
    Generate LaTeX examples for different mathematical operations
    """
    examples = {
    "integral": [r"\int x^2 dx", r"\int_0^\infty e^{-x} dx"],
    "derivative": [r"\frac{d}{dx} x^3", r"\frac{\partial^2 u}{\partial t^2}"],
    "sum": [r"\sum_{n=1}^\infty \frac{1}{n^2}"],
    }   
    
    for i in range(num_per_class):
        ### First up, we do integrals
        ### choose an initial object for the integrand 
        initial_integrand_options = ["x", "t", "u", "v", "y", "z", "w", "\\theta", "\phi", "\\alpha", "\\beta"]
        lower_limit_options = ["0", "1", "\\theta", "a_1", "b_1", "-\\infty"]
        upper_limit_options = ["\\infty", "1", "\\theta", "a_2", "b_2", "0"]
        init_integrand = np.random.choice(initial_integrand_options)
        integrand = init_integrand
        lower = np.random.choice(lower_limit_options)
        upper = np.random.choice(upper_limit_options)
        while np.random.rand() < 0.5:
            integrand = generate_features(integrand)
        examples["integral"].append(rf"\int {integrand} \, d{init_integrand}")


        initial_derivative_options = ["x", "t", "u", "v", "y", "z", "w", "\\theta", "\phi", "\\alpha", "\\beta"]
        ### Next up, we do derivatives
        init_derivative = np.random.choice(initial_derivative_options)
        derivative = init_derivative
        while np.random.rand() < 0.5:
            derivative = generate_features(derivative)
        examples["derivative"].append(rf"\frac{{d}}{{dx}} {derivative}")

        ### Finally, we do sums
        initial_sum_options = ["n", "k", "m", "i", "j"]
        init_sum = np.random.choice(initial_sum_options)
        sum_expression = init_sum
        while np.random.rand() < 0.5:
            sum_expression = generate_features(sum_expression)
        examples["sum"].append(rf"\sum_{{{init_sum}=1}}^{{\infty}} \frac{{1}}{{{sum_expression}^2}}")


    return examples

if __name__ == "__main__":

    output_dir = "data/images"
    os.makedirs(output_dir, exist_ok=True)

    records = []
    examples = generate_latex_examples(num_per_class=30)

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

    for category in examples:
        print(f"Sample from {category}: {examples[category][0]}")
