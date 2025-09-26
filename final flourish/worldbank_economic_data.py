# World Bank Economic Data Acquisition - Automated
import requests
import pandas as pd
from datetime import datetime
import time

def get_worldbank_data(indicator, start_year=2022, end_year=2025):
    """Get World Bank data for Bangladesh"""
    
    url = f"https://api.worldbank.org/v2/country/bgd/indicator/{indicator}"
    params = {
        'date': f'{start_year}:{end_year}',
        'format': 'json',
        'per_page': 100
    }
    
    try:
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
    except Exception as e:
        print(f"Error fetching {indicator}: {e}")
    return []

# Get all key indicators
indicators = {
    'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
    'inflation': 'FP.CPI.TOTL.ZG', 
    'exchange_rate': 'PA.NUS.FCRF',
    'remittances': 'BX.TRF.PWKR.CD.DT',
    'gdp_per_capita': 'NY.GDP.PCAP.CD',
    'unemployment': 'SL.UEM.TOTL.ZS',
    'trade_balance': 'NE.TRD.GNFS.ZS'
}

print("🌍 Fetching World Bank Economic Data for Bangladesh...")
all_data = []

for name, indicator_code in indicators.items():
    print(f"  📊 Fetching {name}...")
    data = get_worldbank_data(indicator_code)
    for record in data:
        record['indicator_name'] = name
        all_data.append(record)
    time.sleep(0.5)  # Rate limiting

# Convert to DataFrame
wb_df = pd.DataFrame(all_data)
if len(wb_df) > 0:
    wb_df['date'] = pd.to_datetime(wb_df['date'], format='%Y')
    
    # Pivot to wide format
    wb_pivot = wb_df.pivot_table(
        index='date', 
        columns='indicator_name', 
        values='value'
    ).reset_index()
    
    # Interpolate missing values
    wb_pivot = wb_pivot.sort_values('date')
    wb_pivot = wb_pivot.interpolate(method='linear')
    
    # Save to CSV
    wb_pivot.to_csv('bangladesh_economic_indicators_2022_2025.csv', index=False)
    print(f"✅ Downloaded {len(wb_pivot)} economic records")
    print(f"📁 Saved to: bangladesh_economic_indicators_2022_2025.csv")
    print("\nColumns:", list(wb_pivot.columns))
    print(wb_pivot.head())
else:
    print("❌ No data retrieved")
