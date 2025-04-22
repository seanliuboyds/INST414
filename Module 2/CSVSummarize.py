import pandas as pd

# List of CSV files to combine
csv_files = ['Askpolitics.csv', 'Conservative.csv', 'Conspiracy.csv', 'Democrat.csv']

# Read and combine CSVs
dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Save combined CSV
combined_df.to_csv('Module2.csv', index=False, encoding='utf-8')
print(f"Combined {len(dfs)} CSVs into Module2.csv with {len(combined_df)} rows")