"""
rapid_preprocessing.py - Fixed to handle GeoTIFF files
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from scipy import ndimage
import cv2
from PIL import Image
import rasterio
import warnings
warnings.filterwarnings('ignore')

class RapidFloodPreprocessor:
    def __init__(self):
        self.data_dir = "data/optimized"
        self.output_dir = "data/rapid_processed"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.tile_size = 128
        self.stride = 64
        
    def load_geotiff(self, filepath: str) -> np.ndarray:
        """Load GeoTIFF file"""
        try:
            # Try rasterio first
            with rasterio.open(filepath) as src:
                data = src.read(1)  # Read first band
                return data.astype(np.float32)
        except:
            # Fallback to PIL
            try:
                img = Image.open(filepath)
                return np.array(img).astype(np.float32)
            except:
                # Last resort - numpy
                return np.load(filepath, allow_pickle=True).astype(np.float32)
    
    def process_flood_pair(self, pre_flood_path: str, flood_path: str, mask_path: str):
        """Process before/during flood pair with real mask"""
        
        # Load data
        pre_flood = self.load_geotiff(pre_flood_path)
        flood = self.load_geotiff(flood_path)
        mask = self.load_geotiff(mask_path)
        
        # Ensure 2D arrays
        if len(pre_flood.shape) > 2:
            pre_flood = pre_flood[..., 0] if pre_flood.shape[-1] < pre_flood.shape[0] else pre_flood[0]
        if len(flood.shape) > 2:
            flood = flood[..., 0] if flood.shape[-1] < flood.shape[0] else flood[0]
        if len(mask.shape) > 2:
            mask = mask[..., 0] if mask.shape[-1] < mask.shape[0] else mask[0]
        
        # Ensure consistent shapes
        h = min(pre_flood.shape[0], flood.shape[0], mask.shape[0])
        w = min(pre_flood.shape[1], flood.shape[1], mask.shape[1])
        
        pre_flood = pre_flood[:h, :w]
        flood = flood[:h, :w]
        mask = mask[:h, :w]
        
        # Normalize
        pre_flood = np.nan_to_num(pre_flood, 0)
        flood = np.nan_to_num(flood, 0)
        
        # Normalize to 0-1 range
        if pre_flood.max() > 1:
            pre_flood = (pre_flood - pre_flood.min()) / (pre_flood.max() - pre_flood.min() + 1e-8)
        if flood.max() > 1:
            flood = (flood - flood.min()) / (flood.max() - flood.min() + 1e-8)
        
        # Binary mask
        mask = (mask > 0).astype(np.float32)
        
        # Calculate change
        change = flood - pre_flood
        
        # Create training tiles
        tiles = []
        
        # Ensure we can create tiles
        if h < self.tile_size or w < self.tile_size:
            # Pad if too small
            pad_h = max(0, self.tile_size - h)
            pad_w = max(0, self.tile_size - w)
            
            pre_flood = np.pad(pre_flood, ((0, pad_h), (0, pad_w)), mode='constant')
            flood = np.pad(flood, ((0, pad_h), (0, pad_w)), mode='constant')
            change = np.pad(change, ((0, pad_h), (0, pad_w)), mode='constant')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
            
            h, w = pre_flood.shape
        
        for i in range(0, h - self.tile_size + 1, self.stride):
            for j in range(0, w - self.tile_size + 1, self.stride):
                tile_pre = pre_flood[i:i+self.tile_size, j:j+self.tile_size]
                tile_flood = flood[i:i+self.tile_size, j:j+self.tile_size]
                tile_change = change[i:i+self.tile_size, j:j+self.tile_size]
                tile_mask = mask[i:i+self.tile_size, j:j+self.tile_size]
                
                # Include all tiles (even without water) for training diversity
                tiles.append({
                    'pre': tile_pre,
                    'flood': tile_flood,
                    'change': tile_change,
                    'mask': tile_mask
                })
        
        # If no tiles created, create at least one
        if len(tiles) == 0:
            tiles.append({
                'pre': pre_flood[:self.tile_size, :self.tile_size],
                'flood': flood[:self.tile_size, :self.tile_size],
                'change': change[:self.tile_size, :self.tile_size],
                'mask': mask[:self.tile_size, :self.tile_size]
            })
        
        return tiles
    
    def run(self):
        """Execute rapid preprocessing"""
        print("="*60)
        print("RAPID PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load catalog
        catalog_path = f"{self.data_dir}/catalog.json"
        if not os.path.exists(catalog_path):
            print("❌ Catalog not found. Run acquisition first.")
            return
            
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
        
        all_tiles = []
        
        for event_name, event_data in catalog['events'].items():
            print(f"\n📍 Processing {event_name}")
            
            data_files = event_data['data']
            
            if 's1_pre' in data_files and 's1_flood' in data_files and 'flood_mask' in data_files:
                try:
                    tiles = self.process_flood_pair(
                        data_files['s1_pre'],
                        data_files['s1_flood'],
                        data_files['flood_mask']
                    )
                    
                    all_tiles.extend(tiles)
                    print(f"   ✓ Generated {len(tiles)} tiles")
                except Exception as e:
                    print(f"   ⚠️ Error processing {event_name}: {str(e)}")
        
        if len(all_tiles) == 0:
            print("❌ No tiles generated!")
            return
        
        # Quick train/val split
        np.random.seed(42)
        np.random.shuffle(all_tiles)
        
        split_idx = int(len(all_tiles) * 0.8)
        train_tiles = all_tiles[:split_idx]
        val_tiles = all_tiles[split_idx:]
        
        # Ensure we have at least one tile in validation
        if len(val_tiles) == 0 and len(train_tiles) > 1:
            val_tiles = [train_tiles[-1]]
            train_tiles = train_tiles[:-1]
        
        # Save processed data
        for split_name, tiles in [('train', train_tiles), ('val', val_tiles)]:
            if tiles:
                # Stack all tiles
                pre_stack = np.stack([t['pre'] for t in tiles])
                flood_stack = np.stack([t['flood'] for t in tiles])
                change_stack = np.stack([t['change'] for t in tiles])
                mask_stack = np.stack([t['mask'] for t in tiles])
                
                # Add channel dimension if needed
                if len(pre_stack.shape) == 3:
                    pre_stack = np.expand_dims(pre_stack, axis=1)
                if len(flood_stack.shape) == 3:
                    flood_stack = np.expand_dims(flood_stack, axis=1)
                if len(change_stack.shape) == 3:
                    change_stack = np.expand_dims(change_stack, axis=1)
                
                np.savez_compressed(
                    f"{self.output_dir}/{split_name}_data.npz",
                    pre=pre_stack,
                    flood=flood_stack,
                    change=change_stack,
                    masks=mask_stack
                )
                
                print(f"✓ Saved {split_name}: {len(tiles)} tiles")
        
        print(f"\n✓ Total tiles: {len(all_tiles)}")
        print(f"✓ Output: {self.output_dir}")

if __name__ == "__main__":
    preprocessor = RapidFloodPreprocessor()
    preprocessor.run()