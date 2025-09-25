import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- Configuration ---
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go one level up to the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_ROOT = os.path.join(SCRIPT_DIR, "crop_demo_data_raw")
ASSET_ROOT = os.path.join(SCRIPT_DIR, "crop_demo_assets")
OUTPUT_DIR = os.path.join(ASSET_ROOT, "charts")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "crop_explainer_grid.png")

# --- Main Logic ---
def create_crop_explainer():
    print("Generating Crop Intelligence Explainer Grid...")
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    data = np.load(os.path.join(DATA_ROOT, "multimodal_jessore.npy"))
    
    # Calculate key features for visualization
    # Assuming standard band order: R,G,B,RE,NIR,SWIR1,SWIR2,...
    nir, red, green = data[:,:,4], data[:,:,0], data[:,:,1]
    
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # Load final outputs (with error handling)
    try:
        rgb_preview = np.array(Image.open(os.path.join(ASSET_ROOT, "jessore_rgb_preview.png")))
        stress_overlay = np.array(Image.open(os.path.join(ASSET_ROOT, "jessore_stress_overlay.png")))
    except FileNotFoundError as e:
        print(f"Warning: Could not load model outputs: {e}")
        print("Creating placeholder images...")
        # Create placeholder images from the data
        h, w = data.shape[:2]
        rgb_preview = (data[:,:,:3] * 255).astype(np.uint8)  # Use first 3 bands as RGB
        # Create a simple stress map based on NDVI
        stress_map = 1 - ndvi  # Invert NDVI to show stress
        stress_overlay = plt.cm.viridis(stress_map)[:,:,:3]  # Convert to RGB
        stress_overlay = (stress_overlay * 255).astype(np.uint8)
    
    # Create the 2x2 grid plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='black')
    plt.style.use('dark_background')

    titles = [
        '1. Input: True-Color Satellite View', '2. Feature: Normalized Vegetation Index (NDVI)',
        '3. Feature: Normalized Water Index (NDWI)', '4. AI Insight: Unsupervised Stress Discovery'
    ]

    images = [rgb_preview, ndvi, ndwi, stress_overlay]
    cmaps = ['viridis', 'viridis', 'coolwarm', 'viridis']

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(images[i], cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=16, color='white', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        if i > 0: # Add colorbars to feature maps
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(pad=2.0)
    plt.suptitle("HawkEYE Crop Intelligence: The Discovery Process", fontsize=24, color='white', y=1.03)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', pad_inches=0.1, facecolor='black')

    print(f"Explainer grid saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    create_crop_explainer()