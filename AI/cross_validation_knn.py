from pathlib import Path

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_loader import load_dataset
from preprocess import preprocess_dataset


DATA_DIR = Path("../dataset")


# Load data
X, raw_y = load_dataset(DATA_DIR)


# Label encoding
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(raw_y)


# Flatten
X_flat = X.reshape(
    X.shape[0],
    -1
)


# ==========================
# Cross Validation
# ==========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for k in [1,3,5,7,9,11]:

    model = KNeighborsClassifier(
        n_neighbors=k
    )


    scores = cross_val_score(
        model,
        X_flat,
        y,
        cv=cv,
        scoring="accuracy"
    )


    print(
        f"k={k}: "
        f"{scores.mean():.4f} "
        f"+/- "
        f"{scores.std():.4f}"
    )