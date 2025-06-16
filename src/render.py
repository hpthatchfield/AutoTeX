# src/render.py

import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

def render_latex_to_image(latex_str, output_path, dpi=200, 
                         font_size=None, 
                         background_color='white',
                         add_noise=False,
                         rotation_range=(-2, 2)):
    """
    Render LaTeX to image with variations.
    
    Args:
        latex_str: LaTeX string to render
        output_path: Output path for the image
        dpi: DPI for the image
        font_size: Font size (random if None)
        background_color: Background color
        add_noise: Whether to add noise to the image
        rotation_range: Range for random rotation in degrees
    """
    # Randomize font size if not specified
    if font_size is None:
        font_size = np.random.randint(16, 24)
    
    fig = plt.figure(figsize=(10, 10))
    try:
        text = fig.text(0, 0, f"${latex_str}$", fontsize=font_size)
    except Exception as e:
        print(f"Error rendering {latex_str}: {e}")
        plt.close(fig)
        return False
        
    fig.canvas.draw()

    bbox = text.get_window_extent()
    width, height = bbox.width / dpi, bbox.height / dpi
    fig.set_size_inches((width, height))

    plt.axis('off')
    
    # Save to temporary file first
    temp_path = output_path + '.temp.png'
    fig.savefig(temp_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, 
                facecolor=background_color)
    plt.close(fig)
    
    # Apply post-processing
    img = Image.open(temp_path)
    
    # Random rotation
    if rotation_range:
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        img = img.rotate(angle, expand=True, fillcolor=background_color)
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, 5, img.size + (3,))
        noise = noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + noise, 0, 255).astype(np.uint8))
    
    # Save final image
    img.save(output_path)
    os.remove(temp_path)
    return True
