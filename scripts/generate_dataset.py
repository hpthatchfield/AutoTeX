import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.render import render_latex_to_image
from typing import List, Dict, Tuple
import re


# Example LaTeX strings per category
# examples = {
#     "integral": [r"\int x^2 dx", r"\int_0^\infty e^{-x} dx"],
#     "derivative": [r"\frac{d}{dx} x^3", r"\frac{\partial^2 u}{\partial t^2}"],
#     "sum": [r"\sum_{n=1}^\infty \frac{1}{n^2}"],
# }

def generate_matrix(size: Tuple[int, int]) -> str:
    """Generate a random matrix of given size."""
    rows = []
    for _ in range(size[0]):
        row = []
        for _ in range(size[1]):
            # Generate random matrix elements
            if np.random.random() < 0.3:
                row.append(str(np.random.randint(-10, 10)))
            else:
                row.append(generate_simple_expression())
        rows.append(" & ".join(row))
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"

def generate_simple_expression() -> str:
    """Generate a simple mathematical expression."""
    options = [
        lambda: str(np.random.randint(-10, 10)),
        lambda: f"x^{{{np.random.randint(2, 5)}}}",
        lambda: f"\\sin(x)",
        lambda: f"\\cos(x)",
        lambda: f"\\sqrt{{{np.random.randint(1, 10)}}}",
        lambda: f"\\frac{{{np.random.randint(1, 10)}}}{{{np.random.randint(1, 10)}}}"
    ]
    return np.random.choice(options)()

def generate_features(expr: str) -> str:
    """Generate more complex features from a base expression."""
    expr_wrapped = f"{{{expr}}}"

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
        f"\\sum_{{i=1}}^{{n}} {expr_wrapped}",
        f"\\prod_{{i=1}}^{{n}} {expr_wrapped}",
        f"\\lim_{{x \\to \\infty}} {expr_wrapped}",
        f"\\int {expr_wrapped} dx",
        f"\\int_{{a}}^{{b}} {expr_wrapped} dx"
    ]
    return f"{{{np.random.choice(features)}}}"

def generate_latex_examples(num_per_class: int, seed: int = 42) -> Dict[str, List[str]]:
    """Generate diverse LaTeX examples for different mathematical operations."""
    np.random.seed(seed)
    
    examples = {
        "integral": [],
        "derivative": [],
        "sum": [],
        "matrix": [],
        "fraction": [],
        "limit": [],
        "product": []
    }
    
    for i in range(num_per_class):
        # Generate integrals
        init_integrand = np.random.choice(["x", "t", "u", "v", "y", "z", "w", "\\theta", "\\phi", "\\alpha", "\\beta"])
        integrand = init_integrand
        lower = np.random.choice(["0", "1", "\\theta", "a", "-\\infty"])
        upper = np.random.choice(["\\infty", "1", "\\theta", "b", "0"])
        while np.random.rand() < 0.5:
            integrand = generate_features(integrand)
        examples["integral"].append(rf"\int^{{{upper}}}_{{{lower}}} {integrand} \, d{init_integrand}")

        # Generate derivatives
        init_derivative = np.random.choice(["x", "t", "u", "v", "y", "z", "w", "\\theta", "\\phi", "\\alpha", "\\beta"])
        derivative = init_derivative
        while np.random.rand() < 0.5:
            derivative = generate_features(derivative)
        examples["derivative"].append(rf"\frac{{d}}{{dx}} {derivative}")

        # Generate sums
        init_sum = np.random.choice(["n", "k", "m", "i", "j"])
        sum_expression = init_sum
        while np.random.rand() < 0.5:
            sum_expression = generate_features(sum_expression)
        examples["sum"].append(rf"\sum_{{{init_sum}=1}}^{{\infty}} \frac{{1}}{{{sum_expression}^2}}")

        # Generate matrices
        matrix_size = (np.random.randint(2, 4), np.random.randint(2, 4))
        examples["matrix"].append(generate_matrix(matrix_size))

        # Generate complex fractions
        numerator = generate_simple_expression()
        denominator = generate_simple_expression()
        examples["fraction"].append(f"\\frac{{{numerator}}}{{{denominator}}}")

        # Generate limits
        limit_var = np.random.choice(["x", "n", "t"])
        limit_to = np.random.choice(["\\infty", "0", "1", "-\\infty"])
        limit_expr = generate_simple_expression()
        examples["limit"].append(f"\\lim_{{{limit_var} \\to {limit_to}}} {limit_expr}")

        # Generate products
        init_prod = np.random.choice(["n", "k", "m", "i", "j"])
        prod_expression = init_prod
        while np.random.rand() < 0.5:
            prod_expression = generate_features(prod_expression)
        examples["product"].append(f"\\prod_{{{init_prod}=1}}^{{\infty}} {prod_expression}")

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
                # Randomly choose rendering parameters
                success = render_latex_to_image(
                    latex, 
                    output_path,
                    dpi=np.random.choice([150, 200, 250]),
                    font_size=None,  # Will be randomly chosen
                    background_color=np.random.choice(['white', '#f0f0f0', '#e8e8e8']),
                    add_noise=np.random.random() < 0.3,
                    rotation_range=(-2, 2)
                )
                if success:
                    records.append({
                        "filename": filename,
                        "label": category,
                        "latex": latex,
                        "complexity": len(re.findall(r'\\', latex))  # Simple complexity metric
                    })
            except Exception as e:
                print(f"Error rendering {latex}: {e}")

    # Save labels to CSV
    df = pd.DataFrame(records)
    df.to_csv("data/labels.csv", index=False)

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total images generated: {len(df)}")
    print("\nSamples per category:")
    print(df['label'].value_counts())
    print("\nComplexity distribution:")
    print(df['complexity'].describe())
