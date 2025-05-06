import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("models/iot_intrusion_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Example input data (7 features)
test_data = ['tcp', 'SF', 491, 0, 0, 2, 2]  # Modify values as needed

# Encode categorical values
for i, column in enumerate(['protocol_type', 'flag']):
    test_data[i] = label_encoders[column].transform([test_data[i]])[0]

# Convert to NumPy array and scale
test_array = np.array(test_data, dtype=float).reshape(1, -1)
test_array = scaler.transform(test_array)

# Predict
result = model.predict(test_array)[0]
print(f"🔹 Predicted Attack Type: {result}")
