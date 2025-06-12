# src/render.py

import matplotlib.pyplot as plt
import os

def render_latex_to_image(latex_str, output_path, dpi=200):
    fig = plt.figure()
    text = fig.text(0, 0, f"${latex_str}$", fontsize=20)
    fig.canvas.draw()

    bbox = text.get_window_extent()
    width, height = bbox.width / dpi, bbox.height / dpi
    fig.set_size_inches((width, height))

    plt.axis('off')
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
