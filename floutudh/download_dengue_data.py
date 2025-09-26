# Download Dengue Press Releases from DGHS Website
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
import time
import os

def download_dengue_press_releases():
    """Download all dengue press releases from DGHS website"""
    
    print("🦟 Downloading Dengue Press Releases from DGHS...")
    print("=" * 50)
    
    # Base URL for dengue press releases
    base_url = "https://old.dghs.gov.bd/index.php/bd/home/5200-daily-dengue-status-report"
    
    try:
        # Get the main page
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all press release links
        press_release_links = []
        
        # Look for links containing "ডেঙ্গু প্রেস রিলিজ" (Dengue Press Release)
        links = soup.find_all('a', href=True)
        
        for link in links:
            link_text = link.get_text(strip=True)
            if 'ডেঙ্গু প্রেস রিলিজ' in link_text or 'Dengue Press Release' in link_text:
                href = link['href']
                if href.startswith('/'):
                    href = 'https://old.dghs.gov.bd' + href
                elif not href.startswith('http'):
                    href = 'https://old.dghs.gov.bd/' + href
                
                # Extract date from link text
                date_match = re.search(r'(\d{2})/(\d{2})/(\d{4})', link_text)
                if date_match:
                    day, month, year = date_match.groups()
                    date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    # Filter for 2022-2024
                    if 2022 <= date_obj.year <= 2024:
                        press_release_links.append({
                            'date': date_obj,
                            'date_str': date_str,
                            'url': href,
                            'title': link_text
                        })
        
        print(f"📊 Found {len(press_release_links)} press releases (2022-2024)")
        
        if len(press_release_links) == 0:
            print("❌ No press releases found for 2022-2024")
            return None
        
        # Sort by date
        press_release_links.sort(key=lambda x: x['date'])
        
        # Create directory for downloads
        os.makedirs('dengue_press_releases', exist_ok=True)
        
        # Download each press release
        downloaded_data = []
        successful_downloads = 0
        
        for i, release in enumerate(press_release_links[:50]):  # Limit to first 50 for testing
            try:
                print(f"  📥 Downloading {release['date_str']} ({i+1}/{min(50, len(press_release_links))})")
                
                # Download the press release
                release_response = requests.get(release['url'], timeout=30)
                release_response.raise_for_status()
                
                # Parse the content
                release_soup = BeautifulSoup(release_response.content, 'html.parser')
                
                # Extract text content
                content = release_soup.get_text()
                
                # Look for numbers that might be case counts
                numbers = re.findall(r'\d+', content)
                
                # Try to extract case information
                cases_info = extract_case_info(content)
                
                # Save raw HTML
                filename = f"dengue_press_releases/dengue_{release['date_str']}.html"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(release_response.text)
                
                downloaded_data.append({
                    'date': release['date_str'],
                    'url': release['url'],
                    'title': release['title'],
                    'new_cases': cases_info.get('new_cases', 0),
                    'total_cases': cases_info.get('total_cases', 0),
                    'deaths': cases_info.get('deaths', 0),
                    'dhaka_cases': cases_info.get('dhaka_cases', 0),
                    'recovery': cases_info.get('recovery', 0),
                    'filename': filename
                })
                
                successful_downloads += 1
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"    ❌ Error downloading {release['date_str']}: {e}")
                continue
        
        print(f"\n✅ Successfully downloaded {successful_downloads} press releases")
        
        # Create CSV with extracted data
        if downloaded_data:
            df = pd.DataFrame(downloaded_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Save to CSV
            df.to_csv('bangladesh_dengue_cases_2022_2025.csv', index=False)
            print(f"📁 Saved dengue data to: bangladesh_dengue_cases_2022_2025.csv")
            print(f"📊 Total records: {len(df)}")
            
            return df
        
        return None
        
    except Exception as e:
        print(f"❌ Error accessing DGHS website: {e}")
        return None

def extract_case_info(content):
    """Extract case information from press release content"""
    
    cases_info = {
        'new_cases': 0,
        'total_cases': 0,
        'deaths': 0,
        'dhaka_cases': 0,
        'recovery': 0
    }
    
    try:
        # Look for common patterns in Bengali/English
        # New cases patterns
        new_cases_patterns = [
            r'নতুন\s*(\d+)',
            r'new\s*cases?\s*(\d+)',
            r'আজ\s*(\d+)',
            r'today\s*(\d+)',
            r'(\d+)\s*নতুন'
        ]
        
        for pattern in new_cases_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                cases_info['new_cases'] = int(match.group(1))
                break
        
        # Total cases patterns
        total_patterns = [
            r'মোট\s*(\d+)',
            r'total\s*cases?\s*(\d+)',
            r'সর্বমোট\s*(\d+)',
            r'cumulative\s*(\d+)'
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                cases_info['total_cases'] = int(match.group(1))
                break
        
        # Deaths patterns
        death_patterns = [
            r'মৃত্যু\s*(\d+)',
            r'deaths?\s*(\d+)',
            r'মারা\s*গেছে\s*(\d+)',
            r'(\d+)\s*মৃত্যু'
        ]
        
        for pattern in death_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                cases_info['deaths'] = int(match.group(1))
                break
        
        # Dhaka cases patterns
        dhaka_patterns = [
            r'ঢাকায়\s*(\d+)',
            r'dhaka\s*(\d+)',
            r'ঢাকা\s*(\d+)'
        ]
        
        for pattern in dhaka_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                cases_info['dhaka_cases'] = int(match.group(1))
                break
        
    except Exception as e:
        print(f"    ⚠️ Error extracting case info: {e}")
    
    return cases_info

def create_dengue_summary():
    """Create a summary of downloaded dengue data"""
    
    print("\n📊 Creating Dengue Data Summary...")
    
    try:
        df = pd.read_csv('bangladesh_dengue_cases_2022_2025.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"📊 Total records: {len(df)}")
        
        if 'new_cases' in df.columns:
            print(f"🦟 Total new cases: {df['new_cases'].sum()}")
            print(f"📈 Average daily cases: {df['new_cases'].mean():.1f}")
            print(f"📊 Max daily cases: {df['new_cases'].max()}")
        
        if 'deaths' in df.columns:
            print(f"💀 Total deaths: {df['deaths'].sum()}")
        
        # Monthly summary
        df['month'] = df['date'].dt.to_period('M')
        monthly_summary = df.groupby('month').agg({
            'new_cases': 'sum',
            'deaths': 'sum'
        }).reset_index()
        
        print(f"\n📅 Monthly Summary (Top 5 months):")
        top_months = monthly_summary.nlargest(5, 'new_cases')
        for _, row in top_months.iterrows():
            print(f"  {row['month']}: {row['new_cases']} cases, {row['deaths']} deaths")
        
        return df
        
    except Exception as e:
        print(f"❌ Error creating summary: {e}")
        return None

if __name__ == "__main__":
    # Download dengue data
    dengue_df = download_dengue_press_releases()
    
    if dengue_df is not None:
        # Create summary
        create_dengue_summary()
        
        print(f"\n🎉 Dengue data collection complete!")
        print(f"📁 Files saved:")
        print(f"  - bangladesh_dengue_cases_2022_2025.csv")
        print(f"  - dengue_press_releases/ (HTML files)")
    else:
        print(f"\n❌ Dengue data collection failed")
        print(f"💡 Try manual download from: https://old.dghs.gov.bd/index.php/bd/home/5200-daily-dengue-status-report")
