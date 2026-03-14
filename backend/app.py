from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np

# import your feature extraction
from model.feature_extraction import extract_features

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 🔥 Load trained model once
model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "AI Voice Spoof Detection Backend Running!"

@app.route("/detect", methods=["POST"])
def detect():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 🎯 Extract features from uploaded audio
    features = extract_features(file_path)
    features = features.reshape(1, -1)

    # 🎯 Predict using model
    prediction = model.predict(features)[0]

    # convert output to readable text
    if prediction == 1:
        result = "Spoofed Voice"
    else:
        result = "Genuine Voice"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
