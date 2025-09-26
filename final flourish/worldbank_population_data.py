# World Bank Population Data Acquisition - Automated
import requests
import pandas as pd
from datetime import datetime
import time

def get_wb_population_data():
    """Get World Bank population indicators for Bangladesh"""
    
    indicators = {
        'total_population': 'SP.POP.TOTL',
        'population_growth': 'SP.POP.GROW', 
        'urban_population': 'SP.URB.TOTL',
        'urban_population_pct': 'SP.URB.TOTL.IN.ZS',
        'population_density': 'EN.POP.DNST',
        'age_dependency_ratio': 'SP.POP.DPND',
        'life_expectancy': 'SP.DYN.LE00.IN',
        'fertility_rate': 'SP.DYN.TFRT.IN'
    }
    
    all_data = []
    
    print("👥 Fetching World Bank Population Data for Bangladesh...")
    
    for name, indicator in indicators.items():
        print(f"  📊 Fetching {name}...")
        url = f"https://api.worldbank.org/v2/country/bgd/indicator/{indicator}"
        params = {
            'date': '2015:2025',  # Extended range for trends
            'format': 'json',
            'per_page': 100
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    for item in data[1]:
                        if item['value'] is not None:
                            all_data.append({
                                'year': int(item['date']),
                                'indicator': name,
                                'value': item['value']
                            })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    wb_pop_df = pd.DataFrame(all_data)
    return wb_pop_df

def create_unified_population_dataset():
    """Combine all population data sources"""
    
    # Get World Bank data
    wb_pop = get_wb_population_data()
    
    if len(wb_pop) == 0:
        print("❌ No World Bank population data retrieved")
        return None
    
    # Pivot World Bank data
    wb_pivot = wb_pop.pivot_table(
        index='year', 
        columns='indicator', 
        values='value'
    ).reset_index()
    
    # Add census data for 2022 baseline
    census_2022 = {
        'year': 2022,
        'total_population_census': 165158616,  # BBS 2022 Census
        'urban_percentage_census': 39.1,
        'rural_percentage_census': 60.9,
        'dhaka_division_population': 39907785,
        'dhaka_metro_population': 9589190
    }
    
    # Create monthly interpolated data for HAWKEYE compatibility
    monthly_data = []
    
    for year in range(2022, 2026):
        for month in range(1, 13):
            if year == 2025 and month > 9:
                break
                
            date = pd.Timestamp(year, month, 1)
            
            # Get base population for year
            year_data = wb_pivot[wb_pivot['year'] == year]
            if len(year_data) > 0:
                base_pop = year_data['total_population'].iloc[0] if 'total_population' in year_data.columns else 165000000
                urban_pct = year_data['urban_population_pct'].iloc[0] if 'urban_population_pct' in year_data.columns else 39.5
                pop_density = year_data['population_density'].iloc[0] if 'population_density' in year_data.columns else 1115
            else:
                # Fallback estimates
                base_pop = 165000000 + (year - 2022) * 1500000  # ~1.5M growth per year
                urban_pct = 39.5 + (year - 2022) * 0.5  # Urbanization trend
                pop_density = 1115 + (year - 2022) * 10  # Density increase
            
            monthly_data.append({
                'date': date,
                'total_population': base_pop,
                'dhaka_population_estimated': base_pop * 0.24,  # ~24% in Dhaka Division
                'urban_population_pct': urban_pct,
                'population_density_per_km2': pop_density,
                'population_growth_rate': 1.0 + (year - 2022) * 0.01,  # ~1% annual growth
                'age_dependency_ratio': 50.0 + (year - 2022) * 0.5  # Dependency trend
            })
    
    monthly_df = pd.DataFrame(monthly_data)
    return monthly_df

# Execute population data collection
if __name__ == "__main__":
    population_final = create_unified_population_dataset()
    
    if population_final is not None:
        population_final.to_csv('bangladesh_population_monthly_2022_2025.csv', index=False)
        print(f"✅ Created monthly population dataset with {len(population_final)} records")
        print(f"📁 Saved to: bangladesh_population_monthly_2022_2025.csv")
        print("\nColumns:", list(population_final.columns))
        print(population_final.head())
    else:
        print("❌ Failed to create population dataset")
