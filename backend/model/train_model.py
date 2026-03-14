import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from feature_extraction import extract_features

DATASET_PATH = "dataset"   # your dataset folder

X = []
y = []

for label in ["real", "spoof"]:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        features = extract_features(path)

        X.append(features)
        y.append(0 if label == "real" else 1)

X = np.array(X)
y = np.array(y)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# save
joblib.dump(model, "model.pkl")

print("Model trained and saved!")
