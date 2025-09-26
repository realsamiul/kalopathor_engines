# Complete VIIRS Nightlight Data Acquisition Script
# This provides the exact code for immediate use

viirs_acquisition_code = '''
# STEP 1: Install Earth Engine API (run in terminal)
# pip install earthengine-api

# STEP 2: Authentication (one-time setup)
import ee
import pandas as pd
from datetime import datetime

# Uncomment and run this ONCE for authentication
# ee.Authenticate()  # This opens browser - follow prompts

# STEP 3: Initialize and extract data
ee.Initialize()

# Dhaka region of interest
dhaka_bounds = ee.Geometry.Rectangle([90.35, 23.70, 90.45, 23.85])

# Get VIIRS monthly data
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \\
          .filterDate('2022-01-01', '2025-09-30') \\
          .select(['avg_rad'])

def extract_nightlight(image):
    # Calculate mean radiance for Dhaka
    mean_radiance = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=dhaka_bounds,
        scale=500,
        maxPixels=1e9
    )
    
    return image.set({
        'date': image.date().format('YYYY-MM-dd'),
        'avg_radiance': mean_radiance.get('avg_rad')
    })

# Process collection
processed = viirs.map(extract_nightlight)

# Convert to pandas DataFrame
def ee_to_pandas(collection):
    features = collection.getInfo()['features']
    data = []
    for feature in features:
        props = feature['properties']
        data.append({
            'date': props['date'],
            'nightlight_radiance': props.get('avg_radiance', 0)
        })
    return pd.DataFrame(data)

# Extract data
nightlight_df = ee_to_pandas(processed)
nightlight_df['date'] = pd.to_datetime(nightlight_df['date'])
nightlight_df = nightlight_df.sort_values('date')

# Save to CSV
nightlight_df.to_csv('dhaka_nightlights_2022_2025.csv', index=False)
print(f"✓ Extracted {len(nightlight_df)} months of nightlight data")
print(nightlight_df.head())
'''

print("VIIRS NIGHTLIGHT ACQUISITION CODE:")
print("="*50)
print(viirs_acquisition_code)