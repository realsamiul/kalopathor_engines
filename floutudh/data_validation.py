# Data Validation Script for HAWKEYE Integration
import pandas as pd
from datetime import datetime
import os

def validate_datasets():
    """Validate all collected datasets for HAWKEYE compatibility"""
    
    print("🔍 HAWKEYE Data Validation Report")
    print("=" * 50)
    
    datasets = {
        'Economic': 'bangladesh_economic_indicators_2022_2025.csv',
        'Population': 'bangladesh_population_monthly_2022_2025.csv',
        'Weather': 'dhaka_weather_2022_2025.csv',
        'Nightlights': 'dhaka_nightlights_2022_2025.csv',
        'Dengue': 'bangladesh_dengue_cases_2022_2025.csv'
    }
    
    validation_results = {}
    all_dataframes = {}
    
    for name, filename in datasets.items():
        print(f"\n📊 Validating {name} Data...")
        
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['date'] = pd.to_datetime(df['date'])
                
                # Basic validation
                record_count = len(df)
                date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                missing_dates = df['date'].isnull().sum()
                
                validation_results[name] = {
                    'status': '✅ Available',
                    'records': record_count,
                    'date_range': date_range,
                    'missing_dates': missing_dates,
                    'columns': list(df.columns)
                }
                
                all_dataframes[name] = df
                
                print(f"  ✅ {record_count} records")
                print(f"  📅 Date range: {date_range}")
                print(f"  📋 Columns: {len(df.columns)}")
                
            except Exception as e:
                validation_results[name] = {
                    'status': f'❌ Error: {str(e)}',
                    'records': 0,
                    'date_range': 'N/A',
                    'missing_dates': 'N/A',
                    'columns': []
                }
                print(f"  ❌ Error reading file: {e}")
        else:
            validation_results[name] = {
                'status': '⏳ Not collected yet',
                'records': 0,
                'date_range': 'N/A',
                'missing_dates': 'N/A',
                'columns': []
            }
            print(f"  ⏳ File not found: {filename}")
    
    # Check temporal overlap
    print(f"\n🔄 Temporal Overlap Analysis")
    print("-" * 30)
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 9, 30)
    
    available_datasets = {k: v for k, v in all_dataframes.items() if len(v) > 0}
    
    if len(available_datasets) > 1:
        # Find common date range
        min_dates = [df['date'].min() for df in available_datasets.values()]
        max_dates = [df['date'].max() for df in available_datasets.values()]
        
        common_start = max(min_dates)
        common_end = min(max_dates)
        
        print(f"📅 Common date range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
        
        # Check for overlapping periods
        overlap_days = (common_end - common_start).days
        print(f"📊 Overlap period: {overlap_days} days")
        
        if overlap_days > 30:
            print("✅ Sufficient temporal overlap for HAWKEYE analysis")
        else:
            print("⚠️ Limited temporal overlap - may affect model performance")
    
    # HAWKEYE Variable Compatibility Check
    print(f"\n🎯 HAWKEYE Variable Compatibility")
    print("-" * 35)
    
    required_variables = {
        'nightlight_radiance': 'Nightlights',
        'infected': 'Dengue',
        'temperature': 'Weather',
        'humidity': 'Weather',
        'rainfall': 'Weather',
        'population_density': 'Population',
        'gdp_growth': 'Economic'
    }
    
    for var, source in required_variables.items():
        if source in available_datasets:
            df = available_datasets[source]
            if var in df.columns:
                print(f"  ✅ {var} found in {source}")
            else:
                print(f"  ⚠️ {var} missing from {source} (columns: {list(df.columns)})")
        else:
            print(f"  ❌ {var} unavailable (source {source} not collected)")
    
    # Summary
    print(f"\n📋 Collection Status Summary")
    print("-" * 30)
    
    collected = sum(1 for result in validation_results.values() if result['status'].startswith('✅'))
    total = len(validation_results)
    
    print(f"📊 Datasets collected: {collected}/{total}")
    print(f"📈 Total records: {sum(result['records'] for result in validation_results.values())}")
    
    if collected >= 3:
        print("✅ Sufficient data for basic HAWKEYE analysis")
    elif collected >= 2:
        print("⚠️ Partial data - HAWKEYE will work with limited features")
    else:
        print("❌ Insufficient data - need at least 2 datasets")
    
    return validation_results, all_dataframes

def create_combined_dataset():
    """Create a unified dataset for HAWKEYE"""
    
    print(f"\n🔗 Creating Combined Dataset for HAWKEYE")
    print("-" * 40)
    
    # Load available datasets
    datasets = {}
    
    for name, filename in [
        ('economic', 'bangladesh_economic_indicators_2022_2025.csv'),
        ('population', 'bangladesh_population_monthly_2022_2025.csv'),
        ('weather', 'dhaka_weather_2022_2025.csv'),
        ('nightlights', 'dhaka_nightlights_2022_2025.csv'),
        ('dengue', 'bangladesh_dengue_cases_2022_2025.csv')
    ]:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['date'] = pd.to_datetime(df['date'])
                datasets[name] = df
                print(f"  ✅ Loaded {name}: {len(df)} records")
            except Exception as e:
                print(f"  ❌ Error loading {name}: {e}")
    
    if len(datasets) == 0:
        print("❌ No datasets available for combination")
        return None
    
    # Start with the dataset that has the most records
    base_dataset = max(datasets.items(), key=lambda x: len(x[1]))
    combined_df = base_dataset[1].copy()
    
    print(f"  📊 Using {base_dataset[0]} as base dataset")
    
    # Merge other datasets
    for name, df in datasets.items():
        if name != base_dataset[0]:
            try:
                # Merge on date
                combined_df = combined_df.merge(
                    df, 
                    on='date', 
                    how='outer', 
                    suffixes=('', f'_{name}')
                )
                print(f"  ✅ Merged {name}")
            except Exception as e:
                print(f"  ❌ Error merging {name}: {e}")
    
    # Sort by date and fill missing values
    combined_df = combined_df.sort_values('date')
    combined_df = combined_df.interpolate(method='linear')
    
    # Save combined dataset
    combined_df.to_csv('hawkeye_combined_dataset.csv', index=False)
    print(f"  📁 Saved combined dataset: {len(combined_df)} records")
    print(f"  📋 Columns: {len(combined_df.columns)}")
    
    return combined_df

if __name__ == "__main__":
    # Run validation
    validation_results, dataframes = validate_datasets()
    
    # Create combined dataset if we have data
    if len(dataframes) > 0:
        combined_df = create_combined_dataset()
        
        if combined_df is not None:
            print(f"\n🎉 HAWKEYE Dataset Ready!")
            print(f"📁 Combined dataset: hawkeye_combined_dataset.csv")
            print(f"📊 Total records: {len(combined_df)}")
            print(f"📅 Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    else:
        print(f"\n⚠️ No datasets available yet. Run the data collection scripts first.")
