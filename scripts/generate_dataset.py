import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.render import render_latex_to_image
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('latex_generation.log')
    ]
)

def generate_simple_expression() -> str:
    """Generate a simple mathematical expression.
    
    Returns a valid LaTeX expression that can be used as a building block
    for more complex expressions.
    """
    options = [
        # Simple numbers
        lambda: str(np.random.randint(-10, 10)),
        # Powers
        lambda: f"x^{{{np.random.randint(2, 5)}}}",
        # Trig functions
        lambda: f"\\sin(x)",
        lambda: f"\\cos(x)",
        # Square roots
        lambda: f"\\sqrt{{{np.random.randint(1, 10)}}}",
        # Simple fractions
        lambda: f"\\frac{{{np.random.randint(1, 10)}}}{{{np.random.randint(1, 10)}}}"
    ]
    return np.random.choice(options)()

def add_complexity(expr: str, complexity_level: int = 1) -> str:
    """Add complexity to an expression based on the complexity level.
    
    Args:
        expr: Base expression to add complexity to
        complexity_level: Level of complexity to add (1-3)
    
    Returns:
        More complex expression
    """
    # Wrap expression in braces for proper nesting
    expr_wrapped = f"{{{expr}}}"
    
    # Define complexity features with their weights
    features = [
        # Level 1 (simple) features - no nesting
        (0.3, f"{expr_wrapped}^{{2}}"),
        (0.2, f"\\sin\\left({expr_wrapped}\\right)"),
        (0.2, f"\\cos\\left({expr_wrapped}\\right)"),
        (0.2, f"\\frac{{1}}{{{expr_wrapped}}}"),
        (0.1, f"\\sqrt{{{expr_wrapped}}}"),
        
        # Level 2 (medium) features - single nesting
        (0.15, f"e^{{{expr_wrapped}}}"),  # Changed from e^{-{expr}} to avoid double nesting
        (0.15, f"\\ln\\left({expr_wrapped}\\right)"),
        (0.15, f"\\frac{{d}}{{dx}} {expr_wrapped}"),
        (0.15, f"\\frac{{1}}{{{expr_wrapped}}}"),  # Simplified to avoid double nesting
        (0.15, f"\\sum_{{i=1}}^{{n}} {expr_wrapped}"),
        
        # Level 3 (complex) features - careful nesting
        (0.1, f"\\frac{{\\partial}}{{\\partial t}} {expr_wrapped}"),  # Simplified partial derivative
        (0.1, f"\\int {expr_wrapped} \\, dx"),
        (0.1, f"\\int_{{a}}^{{b}} {expr_wrapped} \\, dx"),
        (0.1, f"\\lim_{{x \\to \\infty}} {expr_wrapped}"),
        (0.1, f"\\prod_{{i=1}}^{{n}} {expr_wrapped}")
    ]
    
    # Adjust weights based on complexity level
    weights = [w * (1 if i < 5 else 0.5 if i < 10 else 0.25) for i, (w, _) in enumerate(features)]
    weights = [w/sum(weights) for w in weights]
    
    # Choose feature based on weights
    templates = [t for _, t in features]
    chosen_feature = np.random.choice(templates, p=weights)
    return f"{{{chosen_feature}}}"

def determine_complexity() -> int:
    """Determine how many layers of complexity to add to an expression.
    
    Returns:
        Number of complexity layers to add (0-3)
    """
    # Define probabilities for each complexity level
    # This creates a distribution where:
    # - 30% of expressions are simple (0 layers)
    # - 40% have 1 layer
    # - 20% have 2 layers
    # - 10% have 3 layers
    probabilities = [0.3, 0.4, 0.2, 0.1]
    return np.random.choice([0, 1, 2, 3], p=probabilities)

def generate_integral() -> Tuple[str, int]:
    """Generate a valid integral expression with variable complexity."""
    integrand = generate_simple_expression()
    lower = np.random.choice(["0", "1", "a", "-\\infty"])
    upper = np.random.choice(["\\infty", "1", "b", "0"])
    
    # Add complexity layers based on determined complexity
    complexity = determine_complexity()
    for _ in range(complexity):
        integrand = add_complexity(integrand, complexity_level=min(_ + 1, 3))
    
    return f"\\int_{{{lower}}}^{{{upper}}} {integrand} \\, dx", complexity

def generate_derivative() -> Tuple[str, int]:
    """Generate a valid derivative expression with variable complexity."""
    expr = generate_simple_expression()
    
    # Add complexity layers based on determined complexity
    complexity = determine_complexity()
    for _ in range(complexity):
        expr = add_complexity(expr, complexity_level=min(_ + 1, 3))
    
    return f"\\frac{{d}}{{dx}} {expr}", complexity

def generate_sum() -> Tuple[str, int]:
    """Generate a valid sum expression with variable complexity."""
    expr = generate_simple_expression()
    
    # Add complexity layers based on determined complexity
    complexity = determine_complexity()
    for _ in range(complexity):
        expr = add_complexity(expr, complexity_level=min(_ + 1, 3))
    
    return f"\\sum_{{i=1}}^{{\\infty}} {expr}", complexity

def generate_fraction() -> Tuple[str, int]:
    """Generate a valid fraction expression with variable complexity."""
    numerator = generate_simple_expression()
    denominator = generate_simple_expression()
    
    # Add complexity layers based on determined complexity
    complexity = determine_complexity()
    for _ in range(complexity):
        if np.random.random() < 0.5:
            numerator = add_complexity(numerator, complexity_level=min(_ + 1, 3))
        else:
            denominator = add_complexity(denominator, complexity_level=min(_ + 1, 3))
    
    return f"\\frac{{{numerator}}}{{{denominator}}}", complexity

def generate_limit() -> Tuple[str, int]:
    """Generate a valid limit expression with variable complexity."""
    expr = generate_simple_expression()
    limit_to = np.random.choice(["\\infty", "0", "1", "-\\infty"])
    
    # Add complexity layers based on determined complexity
    complexity = determine_complexity()
    for _ in range(complexity):
        expr = add_complexity(expr, complexity_level=min(_ + 1, 3))
    
    return f"\\lim_{{x \\to {limit_to}}} {expr}", complexity

def generate_latex_examples(num_per_class: int, seed: int = 42) -> Dict[str, List[Tuple[str, int]]]:
    """Generate LaTeX examples for different mathematical operations.
    
    Args:
        num_per_class: Number of examples to generate per category
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping categories to lists of (LaTeX expression, complexity) tuples
    """
    np.random.seed(seed)
    logging.info(f"Generating {num_per_class} examples per category")
    
    # Define categories and their generation functions
    categories = {
        "integral": generate_integral,
        "derivative": generate_derivative,
        "sum": generate_sum,
        "fraction": generate_fraction,
        "limit": generate_limit
    }
    
    examples = {category: [] for category in categories}
    
    # Generate examples for each category
    for category, generator in categories.items():
        logging.info(f"Generating {num_per_class} examples for {category}")
        for _ in range(num_per_class):
            try:
                expr, complexity = generator()
                examples[category].append((expr, complexity))
                logging.debug(f"Generated {category} expression: {expr} (complexity: {complexity})")
            except Exception as e:
                logging.error(f"Error generating {category} expression: {str(e)}")
    
    return examples

if __name__ == "__main__":
    try:
        # Create output directory
        output_dir = "data/images"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

        # Generate examples
        examples = generate_latex_examples(num_per_class=30)
        
        if not examples:
            logging.error("No examples were generated")
            sys.exit(1)

        # Render examples to images
        records = []
        for category, formulas in examples.items():
            logging.info(f"Processing {len(formulas)} formulas for {category}")
            for i, (latex, complexity) in enumerate(formulas):
                filename = f"{category}_{i}.png"
                output_path = os.path.join(output_dir, filename)
                try:
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
                            "complexity": complexity  # Use the complexity we determined during generation
                        })
                        logging.debug(f"Successfully rendered: {latex}")
                    else:
                        logging.warning(f"Failed to render: {latex}")
                except Exception as e:
                    logging.error(f"Error rendering {latex}: {str(e)}")

        if not records:
            logging.error("No images were successfully rendered")
            sys.exit(1)

        # Save labels to CSV
        df = pd.DataFrame(records)
        df.to_csv("data/labels.csv", index=False)
        logging.info("Saved labels to CSV")

        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total images generated: {len(df)}")
        print("\nSamples per category:")
        print(df['label'].value_counts())
        print("\nComplexity distribution:")
        print(df['complexity'].describe())
        print("\nSample expressions from each category:")
        for category in examples:
            print(f"\n{category}:")
            for expr, complexity in examples[category][:3]:  # Show first 3 examples
                print(f"  {expr} (complexity: {complexity})")
                
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)
