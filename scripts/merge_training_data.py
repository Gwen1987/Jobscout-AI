import pandas as pd

# Load original and new gray-area examples
original = pd.read_csv("data/preprocessed_train_data_v2.csv")
gray = pd.read_csv("data/gray_area_examples_v2.csv")

# Merge
combined = pd.concat([original, gray], ignore_index=True)

# Save to new versioned file
combined.to_csv("data/preprocessed_train_data_v3.csv", index=False)

print("✅ Merged dataset saved to preprocessed_train_data_v3.csv")
