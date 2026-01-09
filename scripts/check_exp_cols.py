import pandas as pd

# Load all_data.csv and check exp columns
df = pd.read_csv('../data/all_data.csv', nrows=1)

exp_prefixes = [
    "Route_of_administration_",
    "Cargo_type_",
    "Model_type_",
    "Batch_or_individual_or_barcoded_",
    "Purity_"
]

exp_cols = []
for prefix in exp_prefixes:
    cols = [col for col in df.columns if col.startswith(prefix)]
    exp_cols.extend(cols)
    print(f"{prefix}: {len(cols)} columns - {cols}")

print(f"\nTotal exp columns: {len(exp_cols)}")
print(f"All exp columns: {exp_cols}")
