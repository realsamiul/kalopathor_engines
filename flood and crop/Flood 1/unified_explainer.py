"""
Unified Explainer Generator for HawkEYE Demos
Automatically detects and generates explainer grids for both Flood and Crop demos
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import sobel

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def detect_demo_type():
    """Detect whether this is a flood or crop demo based on available files"""
    flood_data_dir = os.path.join(SCRIPT_DIR, "flood", "demo_data_raw")
    crop_data_dir = os.path.join(SCRIPT_DIR, "crop_demo_final", "crop_demo_data_raw")
    
    if os.path.exists(flood_data_dir) and os.path.exists(os.path.join(flood_data_dir, "s1_pre.npy")):
        return "flood"
    elif os.path.exists(crop_data_dir) and os.path.exists(os.path.join(crop_data_dir, "multimodal_jessore.npy")):
        return "crop"
    else:
        raise ValueError("Could not detect demo type. Please ensure data files are present.")

def create_flood_explainer():
    """Create flood intelligence explainer grid"""
    print("Generating Flood Intelligence Explainer Grid...")
    
    # Paths
    DATA_ROOT = os.path.join(SCRIPT_DIR, "flood", "demo_data_raw")
    ASSET_ROOT = os.path.join(SCRIPT_DIR, "flood", "demo_assets")
    OUTPUT_DIR = os.path.join(ASSET_ROOT, "charts")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "flood_explainer_grid.png")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all necessary data
    s1_pre_structured = np.load(os.path.join(DATA_ROOT, 's1_pre.npy'))
    s1_flood_structured = np.load(os.path.join(DATA_ROOT, 's1_flood.npy'))
    s2_flood_structured = np.load(os.path.join(DATA_ROOT, 's2_flood.npy'))
    dem_raw = np.load(os.path.join(DATA_ROOT, 'dem.npy'))

    # Unpack structured arrays
    s1_pre = np.stack([s1_pre_structured['VV'], s1_pre_structured['VH']], axis=-1).astype(np.float32)
    s1_flood = np.stack([s1_flood_structured['VV'], s1_flood_structured['VH']], axis=-1).astype(np.float32)
    s2_flood = np.stack([s2_flood_structured['B4'], s2_flood_structured['B3'], 
                        s2_flood_structured['B2'], s2_flood_structured['B8']], axis=-1).astype(np.float32)
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

    # Load the final model outputs
    final_mask = np.array(Image.open(os.path.join(ASSET_ROOT, "hero/final_mask.png")))
    prediction_overlay = np.array(Image.open(os.path.join(ASSET_ROOT, "hero/prediction_overlay.png")))

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
    plt.close()

    print(f"Flood explainer grid saved to {OUTPUT_FILE}")

def create_crop_explainer():
    """Create crop intelligence explainer grid"""
    print("Generating Crop Intelligence Explainer Grid...")
    
    # Paths
    DATA_ROOT = os.path.join(SCRIPT_DIR, "crop_demo_final", "crop_demo_data_raw")
    ASSET_ROOT = os.path.join(SCRIPT_DIR, "crop_demo_final", "crop_demo_assets")
    OUTPUT_DIR = os.path.join(ASSET_ROOT, "charts")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "crop_explainer_grid.png")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    data = np.load(os.path.join(DATA_ROOT, "multimodal_jessore.npy"))
    
    # Calculate key features for visualization
    # Assuming standard band order: R,G,B,RE,NIR,SWIR1,SWIR2,...
    nir, red, green = data[:,:,4], data[:,:,0], data[:,:,1]
    
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    
    # Load final outputs
    rgb_preview = np.array(Image.open(os.path.join(ASSET_ROOT, "jessore_rgb_preview.png")))
    stress_overlay = np.array(Image.open(os.path.join(ASSET_ROOT, "jessore_stress_overlay.png")))
    
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
    plt.close()

    print(f"Crop explainer grid saved to {OUTPUT_FILE}")

def main():
    """Main function that detects demo type and generates appropriate explainer"""
    try:
        demo_type = detect_demo_type()
        print(f"Detected {demo_type} demo")
        
        if demo_type == "flood":
            create_flood_explainer()
        elif demo_type == "crop":
            create_crop_explainer()
            
        print("Explainer generation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have run the appropriate demo pipeline first.")
        print("For flood demo: Run flood/02_demo_model_train.py")
        print("For crop demo: Run crop_demo_final/run_corrected_crop_demo.py")

if __name__ == '__main__':
    main()
