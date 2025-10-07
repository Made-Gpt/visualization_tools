import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from PIL import Image
import os

color_list_0 = [
    [0.0, 0.0, 0.0],      # 0: Background
    [0.84, 0.48, 0.48],   # 1: Ceiling
    [0.48, 0.84, 0.48],   # 2: Floor

    [0.48, 0.48, 0.84],   # 3: Wall

    [0.84, 0.84, 0.48],   # 4: Window
    [0.84, 0.48, 0.84],   # 5: Chair 

    # [0.48, 0.84, 0.84],   # 6: Bed
    [0.63, 0.87, 0.96],   # 6: Bed

    [0.79, 0.68, 0.85],   # 7: Sofa
    [0.96, 0.72, 0.48],   # 8: Table
    [0.6, 0.72, 0.48],    # 9: TVs
    [0.48, 0.72, 0.72],   # 10: Furniture

    # [0.48, 0.60, 0.96],  # 11: Object
    [0.32, 0.54, 0.78],   # 11: Object

    [0.96, 0.96, 0.96],   # 12: Empty 
]


color_list_1 = [
    [0.80, 0.27, 0.27],  # Ceiling - red
    [0.31, 0.65, 0.15],  # Floor - green
    [0.63, 0.87, 0.96],  # Wall - light blue
    [0.32, 0.54, 0.78],  # Window - blue
    [0.82, 0.84, 0.51],  # Chair - light yellow green
    [0.97, 0.71, 0.42],  # Bed - orange
    [0.67, 0.45, 0.70],  # Sofa - purple
    [0.22, 0.54, 0.75],  # Table - dark blue
    [0.63, 0.75, 0.27],  # TVs - olive green
    [0.94, 0.53, 0.12],  # Furniture - orange
    [0.55, 0.72, 0.90],  # Objects - light blue variant
    [0.79, 0.68, 0.85],  # Objects 2 - pale purple
]

class_names = [
    "Background",
    "Ceiling", "Floor", "Wall", "Window", "Chair", "Bed",
    "Sofa", "Table", "TVs", "Furniture", "Object", "Empty"
]

def show_colors(colors, labels):
    n = len(colors) 
    fig, ax = plt.subplots(figsize=(8, n * 0.4))
    for i, (color, name) in enumerate(zip(colors, labels)):
        rect = mpatches.Rectangle((0, i), 1, 1, color=color)
        ax.add_patch(rect)
        ax.text(1.1, i + 0.5, f"{i}: {name}", va='center', fontsize=10)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, n)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def generate_color_images(color_list, class_names, output_dir="color_images"):
    """
    Generate 16x16 images for each color in the color list
    
    Args:
        color_list: List of RGB colors in normalized format (0.0-1.0)
        class_names: List of class names corresponding to each color
        output_dir: Directory to save the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (color, name) in enumerate(zip(color_list, class_names)):
        # Convert normalized color to 0-255 range
        rgb = [int(c * 255) for c in color]
        
        # Create 16x16 image array
        img_array = np.full((16, 16, 3), rgb, dtype=np.uint8)
        
        # Create PIL Image
        img = Image.fromarray(img_array)
        
        # Save image
        filename = f"color_{i:02d}_{name}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        print(f"Saved: {filepath}")
 
# show_colors(color_list_0, class_names)
generate_color_images(color_list_0, class_names, output_dir="/home/made/code/occnet/EmbodiedOcc/data/colors")
