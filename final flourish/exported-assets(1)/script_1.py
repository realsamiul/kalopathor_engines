# Dengue Data Acquisition - Step-by-Step Instructions
dengue_instructions = '''
PRIMARY METHOD: DGHS Dengue Dashboard (Manual Download)

EXACT CLICK-BY-CLICK INSTRUCTIONS:

Step 1: Access Current Dashboard Data
1. Go to: https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php
2. Right-click on the page → "Save Page As" → Save as HTML
3. Look for downloadable tables/CSV export buttons (if available)

Step 2: Historical Press Releases (More Data)
1. Go to: https://old.dghs.gov.bd/index.php/bd/home/5200-daily-dengue-status-report
2. This page has daily press releases from 2022-2024
3. Click each "Dengue Press Release [DATE]" link
4. Download PDF/CSV files if available
5. For manual extraction: Copy data tables from each day

Step 3: Data Variables to Extract
- Date (daily)
- New cases (by district)
- Deaths (by age/gender)
- Total cumulative cases
- Hospital admissions
- Recovery numbers

Expected Output Format:
Date,NewCases,Deaths,TotalCases,Dhaka_Cases,Chittagong_Cases,Recovery
2024-01-01,45,2,12543,23,8,35
...

ALTERNATIVE SOURCES (If Primary Fails):

1. WHO Bangladesh Data
   URL: https://data.who.int/countries/050
   - Click "View More" on disease data
   - Download dengue-related indicators

2. Academic Datasets (Mendeley/ResearchGate)
   - Search "Bangladesh dengue surveillance data 2022-2025"
   - Download published datasets with DOIs

3. IEDCR Data Portal
   URL: https://champs.iedcr.gov.bd/data
   - Access summarized surveillance data
   - Real-time updates available
'''

print("DENGUE DATA ACQUISITION GUIDE:")
print("="*50)
print(dengue_instructions)