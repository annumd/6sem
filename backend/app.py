from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np
import uuid

from model.feature_extraction import extract_features

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load(os.path.join(os.getcwd(), "model", "model.pkl"))

@app.route("/")
def home():
    return "AI Voice Spoof Detection Backend Running!"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["audio"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(file_path)

        # ✅ FILE SIZE LIMIT (prevents crash)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        if file_size > 5:
            os.remove(file_path)
            return jsonify({"error": "File too large (max 5MB)"}), 400

        features = extract_features(file_path)
        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

        result = "Spoofed Voice" if prediction == 1 else "Genuine Voice"

        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({"result": result})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
