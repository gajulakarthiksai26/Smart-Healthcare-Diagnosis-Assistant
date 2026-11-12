
import pandas as pd

# Load original dataset
df = pd.read_csv("DataSet.csv")

# Check required column
if "Disease_Name" not in df.columns:
    raise ValueError("Disease_Name column not found!")

# Find the maximum number of records in any class
target_count = df["Disease_Name"].value_counts().max()

# Balance all disease classes
balanced_groups = []

for disease, group in df.groupby("Disease_Name"):
    balanced_group = group.sample(
        n=target_count,
        replace=len(group) < target_count,
        random_state=42
    )
    balanced_groups.append(balanced_group)

# Combine and shuffle
balanced_df = pd.concat(
    balanced_groups,
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save balanced dataset
balanced_df.to_csv(
    "DataSet_balanced.csv",
    index=False
)

# Display results
print("BALANCED DATASET CREATED SUCCESSFULLY")
print("Total records:", len(balanced_df))
print("Total disease classes:",
      balanced_df["Disease_Name"].nunique())

print("\nRecords per disease:")
print(balanced_df["Disease_Name"].value_counts())
