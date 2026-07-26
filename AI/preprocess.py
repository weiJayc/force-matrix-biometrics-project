import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_dataset(
    X,
    raw_y,
    test_size=0.2,
    random_state=42,
):
    """
    完整資料前處理

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    encoder
    sensor_min
    sensor_max
    """

    # Label Encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y)

    print("========== Label Mapping ==========")
    for index, label in enumerate(encoder.classes_):
        print(f"{label} -> {index}")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\n========== Dataset ==========")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    print("\n========== Train Label ==========")
    unique, count = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, count):
        print(u, c)

    print("\n========== Test Label ==========")
    unique, count = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, count):
        print(u, c)

    # Normalization (只使用 Train)
    train_flat = X_train.reshape(-1, 16)

    sensor_min = train_flat.min(axis=0)
    sensor_max = train_flat.max(axis=0)

    # 避免除以0
    sensor_range = sensor_max - sensor_min
    sensor_range[sensor_range == 0] = 1

    X_train = (X_train - sensor_min) / sensor_range
    X_test = (X_test - sensor_min) / sensor_range

    print("\n========== Normalize ==========")
    print("Train Min:", X_train.min())
    print("Train Max:", X_train.max())

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        sensor_min,
        sensor_max,
    )