import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def flatten_samples(X: np.ndarray) -> np.ndarray:
    """
    將樣本展平成一維向量。

    Parameters
    ----------
    X : ndarray
        形狀為 (N, 50, 16) 的樣本資料。

    Returns
    -------
    ndarray
        展平後的資料，形狀為 (N, 800)。
    """

    if X.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {X.shape}")

    return X.reshape(X.shape[0], -1)


def normalize_samples(
    X: np.ndarray,
    sensor_min: np.ndarray,
    sensor_max: np.ndarray,
) -> np.ndarray:
    """
    使用指定的 sensor min/max 對資料做正規化。

    Parameters
    ----------
    X : ndarray
        原始資料，形狀為 (N, 50, 16)。
    sensor_min : ndarray
        每個 sensor 的最小值，形狀為 (16,)。
    sensor_max : ndarray
        每個 sensor 的最大值，形狀為 (16,)。

    Returns
    -------
    ndarray
        正規化後的資料。
    """

    sensor_range = sensor_max - sensor_min
    sensor_range[sensor_range == 0] = 1

    return (X - sensor_min) / sensor_range


def normalize_with_train(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 train 資料計算 sensor 統計量，並套用於 train/test。

    Parameters
    ----------
    X_train : ndarray
        訓練資料，形狀為 (N, 50, 16)。
    X_test : ndarray
        測試資料，形狀為 (N, 50, 16)。

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        正規化後的 X_train、X_test、sensor_min、sensor_max。
    """

    if X_train.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {X_train.shape}")

    feature_dim = X_train.shape[2]
    train_flat = X_train.reshape(-1, feature_dim)

    sensor_min = train_flat.min(axis=0)
    sensor_max = train_flat.max(axis=0)

    X_train = normalize_samples(X_train, sensor_min, sensor_max)
    X_test = normalize_samples(X_test, sensor_min, sensor_max)

    return X_train, X_test, sensor_min, sensor_max


def preprocess_dataset(
    X: np.ndarray,
    raw_y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    完整資料前處理。

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
    X_train, X_test, sensor_min, sensor_max = normalize_with_train(
        X_train,
        X_test,
    )

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