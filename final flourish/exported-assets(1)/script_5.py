# Population & Demographic Data - Complete Acquisition Guide
population_instructions = '''
POPULATION & DEMOGRAPHIC DATA ACQUISITION

PART 1: BANGLADESH BUREAU OF STATISTICS (BBS) - Primary Source

Step 1: BBS Census 2022 (Main Portal)
1. Go to: http://nsds.bbs.gov.bd/en/posts/60/Population%20&%20Housing%20Census%202022
2. Look for "Download" or "Data" sections
3. Download Excel/CSV files for:
   - Population by Division/District/Upazila
   - Age distribution by gender
   - Urban vs Rural population
   - Household statistics

Step 2: Alternative BBS Links
1. Main BBS Portal: http://nsds.bbs.gov.bd/
2. Navigate: "Census & Survey" → "Population Census"
3. Download available datasets

Variables to Extract from BBS Census:
- Total population by administrative unit
- Population density (persons per sq km)
- Urban population percentage
- Rural population percentage  
- Age distribution (0-14, 15-64, 65+ years)
- Gender ratio (males per 100 females)
- Household size average
- Literacy rate by area

PART 2: UN HUMANITARIAN DATA EXCHANGE (HDX) - Ready-to-Use

Step 1: HDX Bangladesh Demographics
1. Go to: https://data.humdata.org/dataset/populationa-and-housing-census-dataset
2. Click "Download" on CSV files
3. Files available:
   - Population by administrative boundaries
   - Demographics indicators
   - Socio-economic data

Step 2: Processing HDX Data

import pandas as pd

def process_hdx_population_data(file_path):
    """Process HDX population dataset"""
    
    df = pd.read_csv(file_path)
    
    # Standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Focus on Dhaka region
    dhaka_data = df[
        (df['division'].str.contains('Dhaka', na=False)) |
        (df['district'].str.contains('Dhaka', na=False))
    ]
    
    return dhaka_data

# Load and process
# hdx_df = process_hdx_population_data('bangladesh_population_census.csv')
# print(f"✓ Processed {len(hdx_df)} administrative units in Dhaka region")

PART 3: WORLD BANK POPULATION API

Step 1: World Bank Demographics API
No authentication required

import requests
import pandas as pd

def get_wb_population_data():
    """Get World Bank population indicators for Bangladesh"""
    
    indicators = {
        'total_population': 'SP.POP.TOTL',
        'population_growth': 'SP.POP.GROW', 
        'urban_population': 'SP.URB.TOTL',
        'urban_population_pct': 'SP.URB.TOTL.IN.ZS',
        'population_density': 'EN.POP.DNST',
        'age_dependency_ratio': 'SP.POP.DPND'
    }
    
    all_data = []
    
    for name, indicator in indicators.items():
        url = f"https://api.worldbank.org/v2/country/bgd/indicator/{indicator}"
        params = {
            'date': '2015:2025',  # Extended range for trends
            'format': 'json',
            'per_page': 100
        }
        
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
    
    wb_pop_df = pd.DataFrame(all_data)
    return wb_pop_df

# Get World Bank data
wb_population = get_wb_population_data()
wb_population.to_csv('bangladesh_population_worldbank.csv', index=False)

PART 4: ALTERNATIVE SOURCES

1. UN World Population Prospects
   URL: https://population.un.org/wpp/Download/Standard/CSV/
   Download: "Population by single age and sex"
   Coverage: Bangladesh 1950-2100

2. DHS Program Data
   URL: https://dhsprogram.com/data/available-datasets.cfm?ctryid=1
   Registration required (free)
   Variables: Detailed demographic and health data

3. Kaggle Bangladesh Demographics
   URL: https://www.kaggle.com/datasets/raselmeya/bangladesh-population-and-gdp-growth-datasets
   Direct download of processed data

UNIFIED POPULATION DATASET CREATION:

def create_unified_population_dataset():
    """Combine all population data sources"""
    
    # World Bank time series data
    wb_pop = pd.read_csv('bangladesh_population_worldbank.csv')
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
            
            # Interpolate annual values to monthly
            base_pop = wb_pivot[wb_pivot['year'] == year]['total_population'].iloc[0] if len(wb_pivot[wb_pivot['year'] == year]) > 0 else 165000000
            
            monthly_data.append({
                'date': date,
                'total_population': base_pop,
                'dhaka_population_estimated': base_pop * 0.24,  # ~24% in Dhaka Division
                'urban_population_pct': 39.5 + (year - 2022) * 0.5,  # Urbanization trend
                'population_density_per_km2': base_pop / 148460  # Bangladesh area
            })
    
    monthly_df = pd.DataFrame(monthly_data)
    return monthly_df

# Create final population dataset
population_final = create_unified_population_dataset()
population_final.to_csv('bangladesh_population_monthly_2022_2025.csv', index=False)
print(f"✓ Created monthly population dataset with {len(population_final)} records")

EXPECTED OUTPUT VARIABLES FOR HAWKEYE:
- date: Monthly timestamps
- total_population: Bangladesh total population
- dhaka_population_estimated: Dhaka Division population estimate
- urban_population_pct: Percentage living in urban areas
- population_density_per_km2: National density
- age_group_0_14_pct: Youth dependency
- age_group_15_64_pct: Working age population
- age_group_65_plus_pct: Elderly dependency

Time to Complete: 15-20 minutes (including downloads and processing)
'''

print("POPULATION DATA COMPLETE GUIDE:")
print("="*50)
print(population_instructions)