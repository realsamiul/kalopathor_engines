"""
Configuration settings for the flood detection system
"""

# Google Cloud Project Configuration
GCP_PROJECT = "hyperion-472805"

# Asset directory for storing data
ASSET_DIR = "data"

# Scale for satellite imagery processing (meters per pixel)
SCALE = 30

# Additional configuration
MAX_PIXELS = 1e7
TILE_SIZE = 128
STRIDE = 64

# Model configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 10

# Data paths
DATA_DIR = "data/optimized"
OUTPUT_DIR = "data/rapid_processed"
MODEL_PATH = "optimized_flood_model.pt"
