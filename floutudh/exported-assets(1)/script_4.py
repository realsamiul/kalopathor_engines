# Economic Data APIs - Complete Setup Guide
economic_api_code = '''
ECONOMIC DATA ACQUISITION - COMPLETE API SETUP

PART 1: WORLD BANK API (No Authentication Required)

Step 1: World Bank API Setup
Base URL: https://api.worldbank.org/v2/country/bgd/indicator/
No API key required - Direct access

Key Economic Indicators for Bangladesh:
- NY.GDP.MKTP.KD.ZG: GDP growth (annual %)
- FP.CPI.TOTL.ZG: Inflation rate (%)
- PA.NUS.FCRF: Exchange rate (LCU per USD)
- BX.TRF.PWKR.CD.DT: Personal remittances received (USD)
- NY.GDP.PCAP.CD: GDP per capita (current USD)

Complete World Bank API Script:

import requests
import pandas as pd
from datetime import datetime

def get_worldbank_data(indicator, start_year=2022, end_year=2025):
    """Get World Bank data for Bangladesh"""
    
    url = f"https://api.worldbank.org/v2/country/bgd/indicator/{indicator}"
    params = {
        'date': f'{start_year}:{end_year}',
        'format': 'json',
        'per_page': 100
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if len(data) > 1:
            records = []
            for item in data[1]:  # Skip metadata
                if item['value'] is not None:
                    records.append({
                        'date': item['date'],
                        'indicator': indicator,
                        'value': item['value'],
                        'country': item['country']['value']
                    })
            return records
    return []

# Get all key indicators
indicators = {
    'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
    'inflation': 'FP.CPI.TOTL.ZG', 
    'exchange_rate': 'PA.NUS.FCRF',
    'remittances': 'BX.TRF.PWKR.CD.DT',
    'gdp_per_capita': 'NY.GDP.PCAP.CD'
}

all_data = []
for name, indicator_code in indicators.items():
    print(f"Fetching {name}...")
    data = get_worldbank_data(indicator_code)
    for record in data:
        record['indicator_name'] = name
        all_data.append(record)

# Convert to DataFrame
wb_df = pd.DataFrame(all_data)
wb_df['date'] = pd.to_datetime(wb_df['date'], format='%Y')
wb_df.to_csv('bangladesh_worldbank_data.csv', index=False)
print(f"✓ Downloaded {len(wb_df)} World Bank records")

PART 2: BANGLADESH BANK DATA (Manual Download + API)

Step 1: Manual Download from Bangladesh Bank
1. Go to: https://www.bb.org.bd/en/index.php/econdata/
2. Click on each section:
   - "Exchange Rate" → Download Excel/CSV
   - "Monetary Survey" → Download monthly data
   - "Balance of Payments" → Download quarterly data
   - "Foreign Exchange Reserve" → Download daily data

Step 2: Key Variables to Extract:
- USD/BDT exchange rate (daily)
- Money supply (M1, M2, M3) - monthly
- Foreign exchange reserves - daily
- Import/Export values - monthly
- Remittance inflows - monthly

Step 3: Data Processing Script:

import pandas as pd

def process_bb_exchange_rate(file_path):
    """Process Bangladesh Bank exchange rate data"""
    # Assuming downloaded Excel file
    df = pd.read_excel(file_path)
    
    # Clean and standardize columns
    df = df.dropna()
    df['Date'] = pd.to_datetime(df.iloc[:, 0])  # First column is date
    df['USD_BDT'] = pd.to_numeric(df.iloc[:, 1])  # Exchange rate
    
    # Filter for target period
    df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2025-09-30')]
    
    return df[['Date', 'USD_BDT']]

def process_bb_reserves(file_path):
    """Process foreign exchange reserves data"""
    df = pd.read_excel(file_path)
    
    # Clean data
    df['Date'] = pd.to_datetime(df.iloc[:, 0])
    df['Reserves_USD_Million'] = pd.to_numeric(df.iloc[:, 1])
    
    # Filter for target period
    df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2025-09-30')]
    
    return df[['Date', 'Reserves_USD_Million']]

# Process downloaded files
# exchange_df = process_bb_exchange_rate('bb_exchange_rate.xlsx')
# reserves_df = process_bb_reserves('bb_reserves.xlsx')

PART 3: ALTERNATIVE SOURCES (If Primary Fails)

1. Trading Economics API
   URL: https://tradingeconomics.com/bangladesh/indicators
   Variables: GDP, inflation, unemployment, trade balance
   Note: Requires paid subscription for API access

2. IMF Data API (Free)
   URL: https://www.imf.org/external/datamapper/api/help
   
   # IMF API example
   import requests
   
   # Get Bangladesh data from IMF
   imf_url = "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/BGD"
   response = requests.get(imf_url)
   if response.status_code == 200:
       imf_data = response.json()
       print("IMF GDP Growth Data:", imf_data)

3. FRED Economic Data (Alternative)
   URL: https://fred.stlouisfed.org/
   Search: "Bangladesh" for available indicators
   
UNIFIED ECONOMIC DATA OUTPUT:

def create_unified_economic_dataset():
    """Combine all economic data sources"""
    
    # Load World Bank data
    wb_df = pd.read_csv('bangladesh_worldbank_data.csv')
    
    # Pivot World Bank data
    wb_pivot = wb_df.pivot_table(
        index='date', 
        columns='indicator_name', 
        values='value'
    ).reset_index()
    
    # Add Bangladesh Bank data (when available)
    # bb_exchange = pd.read_csv('bb_exchange_processed.csv')
    # bb_reserves = pd.read_csv('bb_reserves_processed.csv')
    
    # Merge all sources
    # economic_df = wb_pivot.merge(bb_exchange, left_on='date', right_on='Date', how='left')
    # economic_df = economic_df.merge(bb_reserves, left_on='date', right_on='Date', how='left')
    
    economic_df = wb_pivot  # Start with World Bank for now
    economic_df['date'] = pd.to_datetime(economic_df['date'])
    
    # Interpolate missing values for monthly data
    economic_df = economic_df.sort_values('date')
    economic_df = economic_df.interpolate(method='linear')
    
    return economic_df

# Create final dataset
economic_final = create_unified_economic_dataset()
economic_final.to_csv('bangladesh_economic_indicators_2022_2025.csv', index=False)
print(f"✓ Created unified economic dataset with {len(economic_final)} records")
'''

print("ECONOMIC DATA COMPLETE API GUIDE:")
print("="*50)
print(economic_api_code)