"""
HawkEYE Demo - Step 1: Focused Data Acquisition
Pulls multi-modal data (SAR, Optical, DEM) for a single, high-quality
flood event in Bangladesh for the investor demo.
"""
import os
import json
import logging
import requests
import ee
from datetime import datetime

# --- Configuration -----------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GCP_PROJECT = "hyperion-472805"  # Confirmed Project ID
SCALE = 30  # Use 30m scale for speed during the demo pull

# --- Directory Setup (Unique for this demo) ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

DATA_ROOT = "demo_data_raw"
ensure_dir(DATA_ROOT)

# --- Main Data Pull Logic ----------------------------
def pull_demo_data():
    """Initializes GEE and pulls data for the Sylhet 2022 flood."""
    try:
        ee.Initialize(project=GCP_PROJECT)
        log.info(f"Earth Engine initialized for project {GCP_PROJECT}")
    except Exception as e:
        log.error(f"Earth Engine authentication failed: {e}")
        log.error("Please re-run: earthengine authenticate --project hyperion-472805")
        return

    # Focused Flood Event for Demo
    event = {
        'name': 'Bangladesh_Sylhet_2022',
        'aoi': [91.5, 24.7, 91.9, 25.0],
        'pre_date': ('2022-05-01', '2022-05-20'),
        'flood_date': ('2022-06-15', '2022-06-22'),
    }

    log.info(f"Acquiring data for demo event: {event['name']}")
    aoi = ee.Geometry.Rectangle(event['aoi'])

    # 1. Get Sentinel-1 SAR data
    s1_pre = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(*event['pre_date']).first().clip(aoi)
    s1_flood = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(*event['flood_date']).first().clip(aoi)

    # 2. Get Sentinel-2 Optical data (least cloudy)
    s2_flood = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(aoi)
                .filterDate(*event['flood_date'])
                .sort('CLOUDY_PIXEL_PERCENTAGE')
                .first()
                .clip(aoi))

    # 3. Get Digital Elevation Model (DEM)
    dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)

    # --- Download all data layers ---
    datasets = {
        's1_pre': s1_pre.select(['VV', 'VH']),
        's1_flood': s1_flood.select(['VV', 'VH']),
        's2_flood': s2_flood.select(['B4', 'B3', 'B2', 'B8']), # R,G,B,NIR
        'dem': dem.select('elevation')
    }

    for name, image in datasets.items():
        log.info(f"Downloading {name}...")
        url = image.getDownloadURL({'scale': SCALE, 'crs': 'EPSG:4326', 'region': aoi, 'format': 'NPY'})
        filepath = os.path.join(DATA_ROOT, f"{name}.npy")
        
        try:
            with open(filepath, "wb") as f:
                f.write(requests.get(url).content)
            log.info(f" -> Successfully saved to {filepath}")
        except Exception as e:
            log.error(f" -> FAILED to download {name}. Reason: {e}")

    log.info("\nData acquisition for the demo is complete.")


if __name__ == "__main__":
    pull_demo_data()