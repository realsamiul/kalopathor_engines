# Process Existing Data Files for HAWKEYE
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def process_weather_data():
    """Process the existing BD_weather.csv file"""
    print("🌤️ Processing BMD Weather Data...")
    
    try:
        # Load the weather data
        df = pd.read_csv('BD_weather.csv')
        print(f"  📊 Loaded {len(df)} weather records")
        print(f"  📅 Date range: {df.Year.min()}-{df.Year.max()}")
        print(f"  🏢 Stations: {df.Station.nunique()}")
        
        # Filter for Dhaka region stations
        dhaka_stations = ['Dhaka', 'Tejgaon', 'Savar', 'Dhaka/Tejgaon']
        dhaka_data = df[df['Station'].str.contains('|'.join(dhaka_stations), case=False, na=False)]
        
        if len(dhaka_data) == 0:
            # If no Dhaka stations found, use all data and assume it's representative
            dhaka_data = df.copy()
            print("  ⚠️ No Dhaka-specific stations found, using all data")
        else:
            print(f"  🎯 Found {len(dhaka_data)} Dhaka region records")
        
        # Create date column
        dhaka_data['date'] = pd.to_datetime(dhaka_data[['Year', 'Month', 'Day']])
        
        # Filter for target period (2022-2025)
        dhaka_recent = dhaka_data[
            (dhaka_data['date'] >= '2022-01-01') & 
            (dhaka_data['date'] <= '2025-09-30')
        ]
        
        if len(dhaka_recent) == 0:
            # If no recent data, use the most recent available
            dhaka_recent = dhaka_data[dhaka_data['date'] >= '2020-01-01']
            print("  ⚠️ No 2022-2025 data, using 2020+ data")
        
        # Group by date and calculate averages
        weather_daily = dhaka_recent.groupby('date').agg({
            'Temperature': 'mean',
            'Humidity': 'mean', 
            'Rainfall': 'sum',  # Sum rainfall across stations
            'Sunshine': 'mean'
        }).reset_index()
        
        # Rename columns for HAWKEYE compatibility
        weather_daily = weather_daily.rename(columns={
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'Rainfall': 'rainfall',
            'Sunshine': 'sunshine_hours'
        })
        
        # Fill missing values
        weather_daily = weather_daily.interpolate(method='linear')
        
        # Save processed data
        weather_daily.to_csv('dhaka_weather_2022_2025.csv', index=False)
        print(f"  ✅ Processed {len(weather_daily)} days of weather data")
        print(f"  📁 Saved to: dhaka_weather_2022_2025.csv")
        
        return weather_daily
        
    except Exception as e:
        print(f"  ❌ Error processing weather data: {e}")
        return None

def process_economic_data():
    """Process existing economic data"""
    print("💰 Processing Economic Data...")
    
    try:
        # Load GDP growth data
        gdp_df = pd.read_csv('export-and-import-rice-data-of-bangladesh-All-2025-09-26_0648/gdp-growth-rate-in-bangladesh.csv')
        print(f"  📊 Loaded {len(gdp_df)} GDP records")
        
        # Create date column (assuming year-end)
        gdp_df['date'] = pd.to_datetime(gdp_df['Year'].astype(str) + '-12-31')
        
        # Filter for target period
        gdp_recent = gdp_df[gdp_df['date'] >= '2022-01-01']
        
        # Create monthly interpolated data
        monthly_data = []
        for year in range(2022, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 9:
                    break
                    
                date = pd.Timestamp(year, month, 1)
                
                # Get GDP growth for the year
                year_gdp = gdp_recent[gdp_recent['Year'] == year]
                if len(year_gdp) > 0:
                    gdp_growth = year_gdp['GDP Growth Rate'].iloc[0]
                else:
                    # Estimate based on recent trend
                    gdp_growth = 6.0 + (year - 2022) * 0.1
                
                monthly_data.append({
                    'date': date,
                    'gdp_growth': gdp_growth,
                    'inflation': 5.5 + (year - 2022) * 0.2,  # Estimated inflation
                    'exchange_rate': 110 + (year - 2022) * 2,  # Estimated USD/BDT
                    'remittances': 20000 + (year - 2022) * 1000  # Estimated millions USD
                })
        
        economic_df = pd.DataFrame(monthly_data)
        economic_df.to_csv('bangladesh_economic_indicators_2022_2025.csv', index=False)
        print(f"  ✅ Created {len(economic_df)} economic records")
        print(f"  📁 Saved to: bangladesh_economic_indicators_2022_2025.csv")
        
        return economic_df
        
    except Exception as e:
        print(f"  ❌ Error processing economic data: {e}")
        return None

def process_population_data():
    """Process population data from existing sources"""
    print("👥 Processing Population Data...")
    
    try:
        # Create population estimates based on known Bangladesh data
        monthly_data = []
        
        # Bangladesh population estimates (millions)
        base_pop_2022 = 165.2  # BBS 2022 Census
        growth_rate = 0.01  # ~1% annual growth
        
        for year in range(2022, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 9:
                    break
                    
                date = pd.Timestamp(year, month, 1)
                
                # Calculate population with growth
                years_from_2022 = year - 2022 + (month - 1) / 12
                total_pop = base_pop_2022 * (1 + growth_rate) ** years_from_2022
                
                monthly_data.append({
                    'date': date,
                    'total_population': total_pop * 1e6,  # Convert to actual numbers
                    'dhaka_population_estimated': total_pop * 0.24 * 1e6,  # ~24% in Dhaka Division
                    'urban_population_pct': 39.5 + years_from_2022 * 0.5,  # Urbanization trend
                    'population_density_per_km2': (total_pop * 1e6) / 148460,  # Bangladesh area
                    'population_growth_rate': growth_rate * 100
                })
        
        population_df = pd.DataFrame(monthly_data)
        population_df.to_csv('bangladesh_population_monthly_2022_2025.csv', index=False)
        print(f"  ✅ Created {len(population_df)} population records")
        print(f"  📁 Saved to: bangladesh_population_monthly_2022_2025.csv")
        
        return population_df
        
    except Exception as e:
        print(f"  ❌ Error processing population data: {e}")
        return None

def create_dummy_dengue_data():
    """Create dummy dengue data for testing (since we don't have real dengue data yet)"""
    print("🦟 Creating Dummy Dengue Data for Testing...")
    
    try:
        # Create realistic dengue case patterns
        monthly_data = []
        
        for year in range(2022, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 9:
                    break
                    
                date = pd.Timestamp(year, month, 1)
                
                # Seasonal pattern (higher in monsoon months)
                if month in [6, 7, 8, 9]:  # Monsoon season
                    base_cases = 150
                elif month in [5, 10]:  # Pre/post monsoon
                    base_cases = 80
                else:  # Dry season
                    base_cases = 30
                
                # Add some year-to-year variation
                year_factor = 1 + (year - 2022) * 0.1
                cases = int(base_cases * year_factor * np.random.uniform(0.8, 1.2))
                
                monthly_data.append({
                    'date': date,
                    'new_cases': cases,
                    'total_cases': sum([d['new_cases'] for d in monthly_data]) + cases,
                    'deaths': max(0, int(cases * 0.02 * np.random.uniform(0.5, 1.5))),
                    'dhaka_cases': int(cases * 0.4),  # ~40% in Dhaka
                    'recovery': int(cases * 0.95)  # 95% recovery rate
                })
        
        dengue_df = pd.DataFrame(monthly_data)
        dengue_df.to_csv('bangladesh_dengue_cases_2022_2025.csv', index=False)
        print(f"  ✅ Created {len(dengue_df)} dengue records")
        print(f"  📁 Saved to: bangladesh_dengue_cases_2022_2025.csv")
        print("  ⚠️ Note: This is dummy data for testing. Replace with real dengue data when available.")
        
        return dengue_df
        
    except Exception as e:
        print(f"  ❌ Error creating dummy dengue data: {e}")
        return None

def create_dummy_nightlight_data():
    """Create dummy nightlight data for testing"""
    print("🌃 Creating Dummy Nightlight Data for Testing...")
    
    try:
        monthly_data = []
        
        for year in range(2022, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 9:
                    break
                    
                date = pd.Timestamp(year, month, 1)
                
                # Simulate economic activity proxy
                base_radiance = 25 + (year - 2022) * 2  # Economic growth
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month / 12)  # Seasonal variation
                radiance = base_radiance * seasonal_factor * np.random.uniform(0.9, 1.1)
                
                monthly_data.append({
                    'date': date,
                    'nightlight_radiance': round(radiance, 2)
                })
        
        nightlight_df = pd.DataFrame(monthly_data)
        nightlight_df.to_csv('dhaka_nightlights_2022_2025.csv', index=False)
        print(f"  ✅ Created {len(nightlight_df)} nightlight records")
        print(f"  📁 Saved to: dhaka_nightlights_2022_2025.csv")
        print("  ⚠️ Note: This is dummy data for testing. Replace with real VIIRS data when available.")
        
        return nightlight_df
        
    except Exception as e:
        print(f"  ❌ Error creating dummy nightlight data: {e}")
        return None

def main():
    """Process all existing data"""
    print("🚀 Processing Existing Data for HAWKEYE")
    print("=" * 50)
    
    # Process each dataset
    weather_df = process_weather_data()
    economic_df = process_economic_data()
    population_df = process_population_data()
    dengue_df = create_dummy_dengue_data()
    nightlight_df = create_dummy_nightlight_data()
    
    # Summary
    print(f"\n📋 Processing Summary")
    print("-" * 25)
    
    datasets = {
        'Weather': weather_df,
        'Economic': economic_df, 
        'Population': population_df,
        'Dengue': dengue_df,
        'Nightlights': nightlight_df
    }
    
    successful = 0
    total_records = 0
    
    for name, df in datasets.items():
        if df is not None:
            successful += 1
            total_records += len(df)
            print(f"  ✅ {name}: {len(df)} records")
        else:
            print(f"  ❌ {name}: Failed")
    
    print(f"\n🎉 Successfully processed {successful}/5 datasets")
    print(f"📊 Total records: {total_records}")
    
    if successful >= 3:
        print("✅ Sufficient data for HAWKEYE analysis!")
        print("🔍 Run data_validation.py to check compatibility")
    else:
        print("⚠️ Need more datasets for reliable HAWKEYE analysis")

if __name__ == "__main__":
    main()
