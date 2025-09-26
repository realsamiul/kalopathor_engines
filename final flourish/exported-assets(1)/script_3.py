# Bangladesh Met Department Data - Complete Acquisition Guide
bmd_instructions = '''
BANGLADESH METEOROLOGICAL DEPARTMENT DATA ACQUISITION

PRIMARY METHOD: Multiple Sources for Complete Coverage

Step 1: Mendeley Dataset (Fastest - 543,839 records)
1. Go to: https://data.mendeley.com/datasets/tbrhznpwg9/1
2. Click "Download" button (blue button, top right)
3. Unzip the downloaded file
4. Open "BD_Weather.csv" - contains ALL stations 2000-2023

Variables in Mendeley Dataset:
- Station_Name: Weather station location
- Date: YYYY-MM-DD format
- Max_Temp_C: Maximum temperature (°C)
- Min_Temp_C: Minimum temperature (°C)
- Rainfall_mm: Daily rainfall (mm)
- Relative_Humidity_Percent: Humidity (%)
- Bright_Sunshine_Hours: Sunshine duration
- Wind_Speed_ms: Wind speed (m/s)
- Cloud_Coverage_Octas: Cloud cover (0-8 scale)

Step 2: Bangladesh Open Data Portal (Government Source)
1. Go to: http://data.gov.bd/dataset?topics=environment
2. Look for "Climate Data" or "Weather Data" links
3. Download CSV files for:
   - Daily maximum/minimum temperature
   - Average humidity
   - Daily rainfall
   - Sunshine hours (1948-2014)

Step 3: Kaggle Alternative Sources
1. Go to: https://www.kaggle.com/datasets/emonreza/65-years-of-weather-data-bangladesh-preprocessed
2. Click "Download" (requires free Kaggle account)
3. Contains monthly averages 1960-2024

Step 4: OASIS Hub Historical Data
1. Go to: https://oasishub.co/dataset/bangladesh-historical-daily-rainfall-record-1948-2014-bangladesh-meteorological-department
2. Register for free account
3. Download historical rainfall 1948-2014

Data Processing Script for Mendeley Dataset:

import pandas as pd

# Load the complete BMD dataset
df = pd.read_csv('BD_Weather.csv')

# Filter for Dhaka region stations
dhaka_stations = ['Dhaka', 'Tejgaon', 'Savar']
dhaka_data = df[df['Station_Name'].isin(dhaka_stations)]

# Convert date column
dhaka_data['Date'] = pd.to_datetime(dhaka_data['Date'])

# Filter for our target period (2022-2025)
dhaka_recent = dhaka_data[
    (dhaka_data['Date'] >= '2022-01-01') & 
    (dhaka_data['Date'] <= '2025-09-30')
]

# Group by date (average across Dhaka stations)
dhaka_daily = dhaka_recent.groupby('Date').agg({
    'Max_Temp_C': 'mean',
    'Min_Temp_C': 'mean',
    'Rainfall_mm': 'sum',  # Sum rainfall across stations
    'Relative_Humidity_Percent': 'mean',
    'Bright_Sunshine_Hours': 'mean',
    'Wind_Speed_ms': 'mean',
    'Cloud_Coverage_Octas': 'mean'
}).reset_index()

# Calculate average temperature
dhaka_daily['Avg_Temp_C'] = (dhaka_daily['Max_Temp_C'] + dhaka_daily['Min_Temp_C']) / 2

# Save processed data for HAWKEYE
dhaka_daily.to_csv('dhaka_weather_2022_2025.csv', index=False)
print(f"✓ Processed {len(dhaka_daily)} days of Dhaka weather data")

ALTERNATIVE BACKUP SOURCES:

1. Visual Crossing Weather API (If BMD fails)
   URL: https://www.visualcrossing.com/weather-history/
   Instructions:
   - Enter "Dhaka, Bangladesh"
   - Set date range: 2022-01-01 to 2025-09-30
   - Download CSV (free tier: 1000 records)

2. Meteostat API (Alternative)
   Python Library: pip install meteostat
   
   from meteostat import Point, Daily
   from datetime import datetime
   
   # Dhaka coordinates
   dhaka = Point(23.7667, 90.3833, 8)
   
   # Get daily data
   start = datetime(2022, 1, 1)
   end = datetime(2025, 9, 30)
   data = Daily(dhaka, start, end)
   data = data.fetch()
   
   # Variables: tavg, tmin, tmax, prcp, wdir, wspd, pres

EXPECTED OUTPUT FORMAT:
Date,Avg_Temp_C,Max_Temp_C,Min_Temp_C,Humidity_Percent,Rainfall_mm,Sunshine_Hours
2022-01-01,18.5,25.2,11.8,72,0.0,8.2
2022-01-02,19.1,26.0,12.2,68,2.3,7.8
...

Time to Complete: 10-15 minutes (including download and processing)
'''

print("BMD WEATHER DATA COMPLETE GUIDE:")
print("="*50)
print(bmd_instructions)