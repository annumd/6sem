from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np
import uuid

from model.feature_extraction import extract_features

# Create flask app FIRST
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = joblib.load("model/model.pkl")


@app.route("/")
def home():
    return "AI Voice Spoof Detection Backend Running!"


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["audio"]

        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(file_path)

        features = extract_features(file_path)
        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

        result = "Spoofed Voice" if prediction == 1 else "Genuine Voice"

        os.remove(file_path)

        return jsonify({"result": result})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
