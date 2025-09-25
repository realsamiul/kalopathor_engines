import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import sobel

# --- Configuration ---
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go one level up to the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, "demo_data_raw")
ASSET_ROOT = os.path.join(PROJECT_ROOT, "demo_assets")
OUTPUT_DIR = os.path.join(ASSET_ROOT, "charts")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "flood_explainer_grid.png")

# --- Main Logic ---
def create_flood_explainer():
    print("Generating Flood Intelligence Explainer Grid...")

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all necessary data
    s1_pre_structured = np.load(os.path.join(DATA_ROOT, 's1_pre.npy'))
    s1_flood_structured = np.load(os.path.join(DATA_ROOT, 's1_flood.npy'))
    s2_flood_structured = np.load(os.path.join(DATA_ROOT, 's2_flood.npy'))
    dem_raw = np.load(os.path.join(DATA_ROOT, 'dem.npy'))

    # Unpack structured arrays
    s1_pre = np.stack([s1_pre_structured['VV'], s1_pre_structured['VH']], axis=-1).astype(np.float32)
    s1_flood = np.stack([s1_flood_structured['VV'], s1_flood_structured['VH']], axis=-1).astype(np.float32)
    s2_flood = np.stack([s2_flood_structured['B4'], s2_flood_structured['B3'], s2_flood_structured['B2'], s2_flood_structured['B8']], axis=-1).astype(np.float32)
    dem = dem_raw.astype(np.float32)

    # Normalize for visualization
    def normalize(arr):
        min_val, max_val = arr.min(), arr.max()
        return (arr - min_val) / (max_val - min_val + 1e-6)

    s1_pre_vv_norm = normalize(s1_pre[:,:,0])
    s2_rgb_norm = normalize(s2_flood[:,:,:3])
    sar_change = normalize(np.abs(s1_flood[:,:,0] - s1_pre[:,:,0]))


    # Recreate the pseudo-mask for visualization
    vv = normalize(s1_flood[:,:,0])
    nir, red = normalize(s2_flood[:,:,3]), normalize(s2_flood[:,:,0])
    ndwi = (nir - red) / (nir + red + 1e-6)
    slope = sobel(dem if dem.ndim == 2 else dem[:,:,0])
    pseudo_mask = (vv < 0.2) & (ndwi > 0.1) & (slope < 0.05)

    # Load the final model outputs (with error handling)
    try:
        final_mask = np.array(Image.open(os.path.join(ASSET_ROOT, "hero/final_mask.png")))
        prediction_overlay = np.array(Image.open(os.path.join(ASSET_ROOT, "hero/prediction_overlay.png")))
    except FileNotFoundError as e:
        print(f"Warning: Could not load model outputs: {e}")
        print("Creating placeholder images...")
        # Create placeholder images
        h, w = pseudo_mask.shape
        final_mask = (pseudo_mask * 255).astype(np.uint8)
        prediction_overlay = np.stack([s2_rgb_norm[:,:,0], s2_rgb_norm[:,:,1], 
                                     np.maximum(s2_rgb_norm[:,:,2], pseudo_mask)], axis=-1)
        prediction_overlay = (prediction_overlay * 255).astype(np.uint8)


    # Create the 2x3 grid plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='black')
    plt.style.use('dark_background')

    titles = [
        '1. Input: Pre-Flood SAR', '2. Input: During-Flood Optical', '3. Input: SAR Change Feature',
        '4. Insight: Physics-Informed Label', '5. AI Output: Flood Mask', '6. Final Result: Data Fusion'
    ]

    images = [
        s1_pre_vv_norm, s2_rgb_norm, sar_change,
        pseudo_mask, final_mask, prediction_overlay
    ]

    cmaps = ['gray', 'viridis', 'hot', 'magma', 'gray', 'viridis']

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=16, color='white', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

    plt.tight_layout(pad=2.0)
    plt.suptitle("HawkEYE Flood Intelligence: The Reasoning Process", fontsize=24, color='white', y=1.03)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', pad_inches=0.1, facecolor='black')

    print(f"Explainer grid saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    create_flood_explainer()