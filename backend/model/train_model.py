import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from feature_extraction import extract_features

DATASET_PATH = "../dataset"

X = []
y = []

for label in ["real", "spoof"]:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        features = extract_features(path)

        if features is None:
            print("❌ Failed file:", path)
            continue

        X.append(features)
        y.append(0 if label == "real" else 1)

print("Total samples used:", len(X))
print("Real:", sum(1 for i in y if i == 0))
print("Spoof:", sum(1 for i in y if i == 1))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# ✅ Scaler added here
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")  # ✅ Save scaler

print("Model trained and saved!")
print("Scaler saved!")
