from pathlib import Path

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_dataset
from preprocess import preprocess_dataset


DATA_DIR = Path("../dataset")


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

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


print("\n========== Flatten ==========")

print("X_train:", X_train_flat.shape)
print("X_test :", X_test_flat.shape)



# ==========================
# 4. Train KNN
# ==========================

model = KNeighborsClassifier(
    n_neighbors=3
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

'''accuracy = np.mean(
    y_pred == y_test
)


print("\n========== Result ==========")

print(
    f"Accuracy: {accuracy:.4f}"
)'''


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

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)