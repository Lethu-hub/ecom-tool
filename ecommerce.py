import os
import pandas as pd
import numpy as np
import glob
import warnings
from scipy import stats

# Ensure the directory exists
output_dir = 'ecommerce datasets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 1: Find the split part files
csv_parts = sorted(glob.glob('ecommerce dataset/Amazon_Part_*.csv'))
print("CSV Parts found:", csv_parts)

if not csv_parts:
    print("Files in 'ecommerce dataset':", os.listdir('ecommerce dataset'))
    raise ValueError("No CSV files found to concatenate in path 'ecommerce dataset/' with pattern 'Amazon_Part_*.csv'.")

# Step 2: Merge them
df_combined = pd.concat([pd.read_csv(f) for f in csv_parts], ignore_index=True)

# Step 3: Save it under the original name
df_combined.to_csv(os.path.join(output_dir, 'Amazon Sale Report.csv'), index=False)
print("✅ Combined file saved as 'ecommerce datasets/Amazon Sale Report.csv'")

# Load datasets
amazonsales_data = pd.read_csv(os.path.join(output_dir, 'Amazon Sale Report.csv'))
cloudwarehouse_data = pd.read_csv('ecommerce dataset/Cloud Warehouse Compersion Chart.csv')
expenseiigf_data = pd.read_csv('ecommerce dataset/Expense IIGF.csv')
internationalsale_data = pd.read_csv('ecommerce dataset/International sale Report.csv')
may2022_data = pd.read_csv('ecommerce dataset/May-2022.csv')
plmarch2021_data = pd.read_csv('ecommerce dataset/P  L March 2021.csv')
salereport_data = pd.read_csv('ecommerce dataset/Sale Report.csv')

# Concatenate all datasets into one DataFrame
data = pd.concat([
    amazonsales_data, cloudwarehouse_data, expenseiigf_data, internationalsale_data,
    may2022_data, plmarch2021_data, salereport_data
], ignore_index=True)

print("Combined DataFrame info:")
print(data.info())

# Save combined dataset to CSV, overwrite if file exists
data.to_csv('e-commerce.csv', index=False)

# Function to analyze a single column
def analyze_column(df, col_name):
    description = {
        'Column Name': col_name,
        'Empty Rows': df[col_name].isnull().sum(),
        'Data Type': df[col_name].dtype
    }

    if pd.api.types.is_numeric_dtype(df[col_name]):
        description['Min'] = df[col_name].min()
        description['Max'] = df[col_name].max()
        description['Mean'] = df[col_name].mean()
        description['Median'] = df[col_name].median()
        mode_result = stats.mode(df[col_name], keepdims=True)
        description['Mode'] = mode_result.mode[0] if mode_result.mode.size > 0 else None
        if mode_result.mode.size > 1:
            description['Other Modes'] = list(mode_result.mode[1:])
    elif pd.api.types.is_object_dtype(df[col_name]):
        description['Unique Values'] = df[col_name].unique()

    return description

# Analyze all columns
def analyze_dataset(df):
    analyses = [analyze_column(df, col) for col in df.columns]
    return pd.DataFrame(analyses)

# Fill missing numeric columns with random values between IQR
def fill_missing_with_iqr_random(df, seed=42):
    np.random.seed(seed)
    df_filled = df.copy()
    for col in df_filled.select_dtypes(include=[np.number]).columns:
        q1 = df_filled[col].quantile(0.25)
        q3 = df_filled[col].quantile(0.75)
        missing_mask = df_filled[col].isna()
        if missing_mask.any():
            df_filled.loc[missing_mask, col] = np.random.uniform(q1, q3, size=missing_mask.sum())
    return df_filled
# Load and preprocess function to be used by Streamlit app
def load_and_preprocess():
    df = pd.read_csv('e-commerce.csv')
    df = fill_missing_with_iqr_random(df)
    return df

if __name__ == '__main__':
    # For standalone execution and debugging
    data = load_and_preprocess()
    analysis = analyze_dataset(data)
    print(analysis)
