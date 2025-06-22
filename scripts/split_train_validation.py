import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full training dataset (with true labels)
df = pd.read_csv("data/preprocessed_train_data_v2.csv")  

# Split into train and validation (80% / 20%)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["fraudulent"])

# Save both datasets
train_df.to_csv("data/train_data.csv", index=False)
val_df.to_csv("data/validation_data.csv", index=False)

print(f"✅ Split complete: {len(train_df)} train rows, {len(val_df)} validation rows")
