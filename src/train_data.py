import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from pathlib import Path

# Step 1: Load and clean the dataset
data_path = Path("../data/fake_job_postings.csv")
df = pd.read_csv(data_path)
# TODO: Drop rows with missing 'description' or target
df.dropna(subset=['description'], inplace=True)

# Step 2: Create train/test split
# TODO: Use sklearn train_test_split with stratify on 'fraudulent'
X = df[['title', 'description']]
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Step 3: Prepare TextVectorization layer
# TODO: Use TextVectorization on 'description' or combine 'title' + 'description'
# TODO: Create a tf.data.Dataset pipeline

# Step 4: Build model
# TODO: Use Keras Sequential
# Example: TextVectorization -> Embedding -> GlobalAveragePooling -> Dense layers -> Sigmoid

# Step 5: Compile the model
# TODO: Use binary_crossentropy and metrics=['accuracy', 'Precision', 'Recall']

# Step 6: Train the model
# TODO: Fit with validation_split or validation_data

# Step 7: Save the model
# TODO: Save to models/saved_model.h5
