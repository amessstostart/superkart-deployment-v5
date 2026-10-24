from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("superkart_model.joblib")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "SuperKart API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        payload = request.get_json()
        data = payload.get("data", payload)
        df = pd.DataFrame(data)
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# BEST PRACTICE: Allow external traffic (0.0.0.0) on port 7860
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
