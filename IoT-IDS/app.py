import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)
model = joblib.load("models/iot_intrusion_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
dos_attacks = ["back", "land", "neptune", "smurf", "teardrop", "pod"]
r2l_attacks = ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"]
u2r_attacks = ["buffer_overflow", "loadmodule", "perl", "rootkit"]
probe_attacks = ["ipsweep", "nmap", "portsweep", "satan"]

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [request.form[key] for key in request.form.keys()]
        for i, column in enumerate(['protocol_type', 'flag']):
            if input_data[i] in label_encoders[column].classes_:
                input_data[i] = label_encoders[column].transform([input_data[i]])[0]
            else:
                return jsonify({"error": f"Invalid value for {column}: {input_data[i]}"})
        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        input_array = scaler.transform(input_array)

        pred = model.predict(input_array)[0]
        confidence = np.max(model.predict_proba(input_array)) * 100  # Confidence Score
        attack_label = label_encoders["labels"].inverse_transform([pred])[0]

        category = categorize_attack(attack_label)

        return jsonify({
            "attack": attack_label,
            "category": category,
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        df = pd.read_csv(file)
        required_columns = ['protocol_type', 'flag', 'src_bytes', 'dst_bytes', 'hot', 'count', 'srv_count']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "CSV file missing required columns."})
        for column in ['protocol_type', 'flag']:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].map(lambda x: label_encoders[column].transform([x])[0] if x in label_encoders[column].classes_ else -1)
        X_scaled = scaler.transform(df[required_columns].to_numpy())
        predictions = model.predict(X_scaled)
        confidences = np.max(model.predict_proba(X_scaled), axis=1) * 100
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            pred_label = label_encoders["labels"].inverse_transform([pred])[0]  # Decode label
            results.append({
                "index": i + 1,
                "attack": pred_label,
                "category": categorize_attack(pred_label),
                "confidence": f"{conf:.2f}%"
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})
def categorize_attack(attack_label):
    if attack_label in dos_attacks:
        return "DoS Attack"
    elif attack_label in r2l_attacks:
        return "R2L Attack"
    elif attack_label in u2r_attacks:
        return "U2R Attack"
    elif attack_label in probe_attacks:
        return "Probe Attack"
    else:
        return "Normal Traffic"

if __name__ == "__main__":
    app.run(debug=True)
