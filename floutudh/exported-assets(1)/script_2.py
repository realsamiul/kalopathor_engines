# OpenWeatherMap Complete Setup Guide
weather_api_code = '''
OPENWEATHERMAP API SETUP - EXACT INSTRUCTIONS

Step 1: Register for API Key (2 minutes)
1. Go to: https://openweathermap.org/api
2. Click "Subscribe" under "One Call API 3.0" (Free tier: 1000 calls/day)
3. Create account with email/password
4. Verify email
5. Go to "API keys" tab in your profile
6. Copy your API key

Step 2: Install Required Libraries
Run in terminal:
pip install requests pandas datetime

Step 3: Complete Weather Data Collection Script

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Your API configuration
API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your key
DHAKA_LAT = 23.8103
DHAKA_LON = 90.4125

# Base URLs for different data types
CURRENT_URL = "http://api.openweathermap.org/data/2.5/weather"
HISTORICAL_URL = "http://api.openweathermap.org/data/3.0/onecall/timemachine"
FORECAST_URL = "http://api.openweathermap.org/data/3.0/onecall"

def get_current_weather():
    """Get current weather data"""
    params = {
        'lat': DHAKA_LAT,
        'lon': DHAKA_LON,
        'appid': API_KEY,
        'units': 'metric'
    }
    
    response = requests.get(CURRENT_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            'datetime': datetime.now(),
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'rainfall': data.get('rain', {}).get('1h', 0),
            'weather_desc': data['weather'][0]['description']
        }
    return None

def get_historical_weather(start_date, end_date):
    """Get historical weather data (last 5 days free)"""
    weather_data = []
    current_date = start_date
    
    while current_date <= end_date:
        timestamp = int(current_date.timestamp())
        
        params = {
            'lat': DHAKA_LAT,
            'lon': DHAKA_LON,
            'dt': timestamp,
            'appid': API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(HISTORICAL_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            current_data = data['data'][0]
            
            weather_data.append({
                'date': current_date.date(),
                'temperature': current_data['temp'],
                'humidity': current_data['humidity'],
                'pressure': current_data['pressure'],
                'rainfall': current_data.get('rain', {}).get('1h', 0),
                'weather_desc': current_data['weather'][0]['description']
            })
        
        current_date += timedelta(days=1)
        time.sleep(1)  # Rate limiting
    
    return weather_data

def get_daily_aggregated_data():
    """Get comprehensive weather data for HAWKEYE"""
    
    # Current weather
    current = get_current_weather()
    
    # Last 5 days (free tier limit)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    historical = get_historical_weather(start_date, end_date)
    
    # Create DataFrame
    df_historical = pd.DataFrame(historical)
    
    # Add current data
    current_df = pd.DataFrame([current])
    current_df['date'] = current_df['datetime'].dt.date
    
    # Combine
    all_data = pd.concat([df_historical, current_df[['date', 'temperature', 'humidity', 'pressure', 'rainfall']]])
    
    return all_data

# Execute data collection
if __name__ == "__main__":
    weather_df = get_daily_aggregated_data()
    weather_df.to_csv('dhaka_weather_recent.csv', index=False)
    print(f"✓ Collected {len(weather_df)} days of weather data")
    print(weather_df.head())

EXTENDED HISTORICAL DATA OPTIONS:

Option 1: Visual Crossing (Alternative Source)
1. Go to: https://www.visualcrossing.com/weather-history/
2. Enter "Dhaka, Bangladesh"
3. Set date range: 2022-01-01 to 2025-09-30
4. Download CSV (free tier: 1000 records/day)

Option 2: World Bank Climate Portal
1. Go to: https://climateknowledgeportal.worldbank.org/
2. Select "Bangladesh"
3. Download historical climate data
4. Choose monthly/daily resolution

VARIABLES FOR HAWKEYE COMPATIBILITY:
- temperature (°C)
- humidity (%)
- pressure (hPa)  
- rainfall (mm)
- weather_description
'''

print("OPENWEATHERMAP COMPLETE SETUP:")
print("="*50)
print(weather_api_code)