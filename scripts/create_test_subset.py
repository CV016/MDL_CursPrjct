import pandas as pd

df = pd.read_csv("scripts/test_traffic.csv")

# Ensure we sample up to 8 rows per phase
sampled_df = df.groupby("phase", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 8), random_state=42)).reset_index(drop=True)

print(f"Created subset with {len(sampled_df)} total rows.")
print("Counts per phase:")
print(sampled_df["phase"].value_counts())

sampled_df.to_csv("scripts/test_subset_traffic.csv", index=False)
