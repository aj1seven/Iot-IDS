import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# ✅ Load dataset
df = pd.read_csv("dataset/train.csv")

# ✅ Select relevant features
selected_features = ['protocol_type', 'flag', 'src_bytes', 'dst_bytes', 'hot', 'count', 'srv_count']

# ✅ Encode categorical columns
label_encoders = {}
for column in ['protocol_type', 'flag', 'labels']:  
    le = LabelEncoder()
    df[column] = df[column].astype(str)  # Convert to string before encoding
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoder

# ✅ Prepare features and labels
X = df[selected_features]
y = df['labels']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# ✅ Save trained model and preprocessing tools
joblib.dump(clf, "models/iot_intrusion_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model training completed and saved in the 'models/' folder!")
