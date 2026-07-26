from pathlib import Path
import sys

import numpy as np

from sklearn.ensemble import RandomForestClassifier

AI_ROOT = Path(__file__).resolve().parent
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data_loader import load_dataset
from preprocess import flatten_samples, preprocess_dataset


DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"


# ==========================
# 1. Load Dataset
# ==========================

X, raw_y = load_dataset(DATA_DIR)


# ==========================
# 2. Preprocess
# ==========================

(
    X_train,
    X_test,
    y_train,
    y_test,
    encoder,
    sensor_min,
    sensor_max,
) = preprocess_dataset(
    X,
    raw_y
)


# ==========================
# 3. Flatten
# ==========================

# (sample, frame, sensor)
# (164,50,16)

X_train_flat = flatten_samples(X_train)
X_test_flat = flatten_samples(X_test)


print("\n========== Flatten ==========")

print("X_train:", X_train_flat.shape)
print("X_test :", X_test_flat.shape)


# ==========================
# 4. Train KNN
# ==========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(
    X_train_flat,
    y_train
)


# ==========================
# 5. Prediction
# ==========================

y_pred = model.predict(
    X_test_flat
)


# ==========================
# 6. Evaluation
# ==========================

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


print("\n========== Result ==========")

accuracy = np.mean(y_pred == y_test)

print(
    f"Accuracy: {accuracy:.4f}"
)


print("\n========== Confusion Matrix ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


print("\n========== Classification Report ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)