import pandas as pd

# Load raw data
df = pd.read_parquet("data/raw/windfor/windfor.parquet")

# Check dates with only 1 settlement period
date_counts = df['settlementDate'].value_counts().sort_index()
problematic_dates = date_counts[date_counts < 48]

print(f"Total dates: {len(date_counts)}")
print(f"Dates with < 48 periods: {len(problematic_dates)}")
print("\nFirst 10 problematic dates:")
print(problematic_dates.head(10))

# Check if it's a settlement period issue
print("\nChecking settlement periods for 2024-01-07:")
jan_7_data = df[df['settlementDate'] == '2024-01-07']
print(f"Total records: {len(jan_7_data)}")
print(f"Settlement periods present: {sorted(jan_7_data['settlementPeriod'].unique())}")

# Check for duplicates in the pivot index
print("\nChecking for duplicate timestamps:")
duplicate_check = df.groupby(['settlementDate', 'settlementPeriod', 'startTime']).size()
duplicates = duplicate_check[duplicate_check > 2]  # Should be exactly 2 (offshore + onshore)
if len(duplicates) > 0:
    print(f"Found {len(duplicates)} timestamps with more than 2 records:")
    print(duplicates.head(10))
else:
    print("No unexpected duplicates found")

# Check raw record count for boundary dates
print("\nRaw record counts for chunk boundary dates:")
for date in ['2024-01-07', '2024-01-14', '2024-01-21']:
    count = len(df[df['settlementDate'] == date])
    print(f"  {date}: {count} records")