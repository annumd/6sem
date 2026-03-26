import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from feature_extraction import extract_features

DATASET_PATH = "dataset"

X = []
y = []

for label in ["real", "spoof"]:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        features = extract_features(path)

        if features is None:
            continue

        X.append(features)
        y.append(0 if label == "real" else 1)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

model = RandomForestClassifier(n_estimators=300, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "model.pkl")

print("Model trained and saved!")
