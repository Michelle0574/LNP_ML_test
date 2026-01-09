import pandas as pd
import numpy as np

# Read files
change_df = pd.read_csv('/root/drug_delivery/LNP_ML/data/data_files_to_merge/Chinese_Academy_of_Sciences/change.csv', 
                        header=None, names=['change'])
main_df = pd.read_csv('/root/drug_delivery/LNP_ML/data/data_files_to_merge/Chinese_Academy_of_Sciences/main_data.csv')

print(f"Total samples: {len(change_df)}")
print(f"Original main data shape: {main_df.shape}")
print(f"Original columns: {list(main_df.columns)}")

# ===== IMPORTANT: Remove old toxicity columns if they exist =====
old_toxicity_cols = ['chronic_toxicity_none', 'acute_toxicity_none', 'acute_toxicity_present',
                     'chronic_toxicity', 'acute_toxicity', 'toxic']
cols_to_drop = [col for col in old_toxicity_cols if col in main_df.columns]
if cols_to_drop:
    print(f"\nDropping old toxicity columns: {cols_to_drop}")
    main_df = main_df.drop(columns=cols_to_drop)
print(f"After dropping old columns shape: {main_df.shape}")

# Convert change values to string for easier handling
change_values = change_df['change'].astype(str).str.strip()

# Initialize new column - 单列：有毒/无毒
toxic = []

# Conversion logic - 合并逻辑
for val in change_values:
    if val == '0':
        # No toxicity at all
        toxic.append(0)
    elif val == '1':
        # No acute toxicity, chronic unknown - 保守处理为未知
        toxic.append(np.nan)  # 或者可以设为 0，取决于您的需求
    elif val == '2':
        # Has acute toxicity (and chronic)
        toxic.append(1)
    else:  # '-' or empty or any other value
        # Missing data
        toxic.append(np.nan)

# Add new column to main dataframe
main_df['toxic'] = toxic

# Verify statistics
print("\n" + "="*80)
print("Conversion Statistics:")
print("="*80)

# Count original values
print("\nOriginal change.csv values:")
for val in ['0', '1', '2', '-', '']:
    count = (change_values == val).sum()
    if count > 0:
        print(f"  '{val}': {count} samples")

# Count new values
print("\nNew toxic values:")
print(f"  0 (not toxic): {(main_df['toxic'] == 0).sum()}")
print(f"  1 (toxic): {(main_df['toxic'] == 1).sum()}")
print(f"  NaN (unknown): {main_df['toxic'].isna().sum()}")

# Verify final shape
print(f"\nFinal main data shape: {main_df.shape}")
print(f"Final columns: {list(main_df.columns)}")

# Save updated file
main_df.to_csv('/root/drug_delivery/LNP_ML/data/data_files_to_merge/Chinese_Academy_of_Sciences/main_data.csv', 
               index=False)

print("\n" + "="*80)
print("✅ Conversion complete! File saved.")
print("="*80)

# Show a few examples
print("\nSample rows (first 10):")
if 'smiles' in main_df.columns:
    print(main_df[['smiles', 'toxic']].head(10).to_string())
else:
    print(main_df[['toxic']].head(10).to_string())