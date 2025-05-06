import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load the dataset
df = pd.read_csv("dataset/upload.csv")

# Convert categorical values to numeric
df['protocol_type'] = df['protocol_type'].astype('category').cat.codes
df['flag'] = df['flag'].astype('category').cat.codes
df['service'] = df['service'].astype('category').cat.codes  # 🔹 Fix: Encode service column

# Select features and labels
X = df.drop(columns=['labels'])  # Features
y = df['labels']  # Target labels

# Encode target labels as numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save label encoding for future predictions
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Normalize numerical values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save processed data
np.save("dataset/X_train.npy", X_train)
np.save("dataset/X_test.npy", X_test)
np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)

print("✅ Preprocessing Completed! Data saved.")
