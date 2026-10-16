from typing import Optional, Sequence

import numpy as np

from preprocess import flatten_samples, normalize_samples


DEFAULT_CONTACT_THRESHOLD: float = 1000.0
RAW_SENSOR_COUNT: int = 16
ENGINEERED_FEATURE_COUNT: int = 7
HYBRID_FEATURE_COUNT: int = RAW_SENSOR_COUNT + ENGINEERED_FEATURE_COUNT
ENGINEERED_FEATURE_ORDER: tuple[str, ...] = (
    "pressure_sum",
    "max_pressure",
    "contact_area",
    "cop_x",
    "cop_y",
    "left_right_ratio",
    "top_bottom_ratio",
)
ENGINEERED_FEATURE_INDEX = {
    feature_name: index for index, feature_name in enumerate(ENGINEERED_FEATURE_ORDER)
}


def extract_frame_features(
    frame: np.ndarray,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> np.ndarray:
    """
    從單一 frame 的 16 個 sensor 值提取物理意義特徵。

    Parameters
    ----------
    frame : ndarray
        單一 frame 的 16 個 sensor 值，形狀為 (16,)。
    contact_threshold : float
        判定為接觸的壓力門檻值。

    Returns
    -------
    ndarray
        長度為 7 的特徵向量，包含：
        pressure_sum, max_pressure, contact_area,
        cop_x, cop_y, left_right_ratio, top_bottom_ratio
    """

    frame = np.asarray(frame, dtype=np.float32)

    if frame.shape != (RAW_SENSOR_COUNT,):
        raise ValueError(f"Expected frame shape ({RAW_SENSOR_COUNT},), got {frame.shape}")

    matrix = frame.reshape(4, 4)

    pressure_sum = float(np.sum(frame))
    max_pressure = float(np.max(frame))
    contact_area = float(np.sum(frame > contact_threshold))

    if pressure_sum > 0:
        x_positions = np.array([0, 1, 2, 3], dtype=np.float32)
        y_positions = np.array([0, 1, 2, 3], dtype=np.float32)

        x_coords = np.tile(x_positions, (4, 1))
        y_coords = np.tile(y_positions.reshape(-1, 1), (1, 4))

        cop_x = float(np.sum(matrix * x_coords) / pressure_sum)
        cop_y = float(np.sum(matrix * y_coords) / pressure_sum)
    else:
        cop_x = 0.0
        cop_y = 0.0

    left_pressure = float(np.sum(matrix[:, :2]))
    right_pressure = float(np.sum(matrix[:, 2:]))
    left_right_ratio = (
        left_pressure / right_pressure if right_pressure > 0 else 0.0
    )

    top_pressure = float(np.sum(matrix[:2, :]))
    bottom_pressure = float(np.sum(matrix[2:, :]))
    top_bottom_ratio = (
        top_pressure / bottom_pressure if bottom_pressure > 0 else 0.0
    )

    return np.array(
        [
            pressure_sum,
            max_pressure,
            contact_area,
            cop_x,
            cop_y,
            left_right_ratio,
            top_bottom_ratio,
        ],
        dtype=np.float32,
    )


def extract_dataset_features(
    X: np.ndarray,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> np.ndarray:
    """
    將完整資料集從原始 pressure values 轉換為工程化特徵。

    Parameters
    ----------
    X : ndarray
        原始資料，形狀為 (samples, 50, 16)。
    contact_threshold : float
        判定為接觸的壓力門檻值。

    Returns
    -------
    ndarray
        特徵資料，形狀為 (samples, 50, 7)。
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {X.shape}")

    if X.shape[2] != RAW_SENSOR_COUNT:
        raise ValueError(f"Expected last dimension {RAW_SENSOR_COUNT}, got {X.shape}")

    num_samples = X.shape[0]
    num_frames = X.shape[1]

    X_features = np.zeros((num_samples, num_frames, ENGINEERED_FEATURE_COUNT), dtype=np.float32)

    for sample_index in range(num_samples):
        for frame_index in range(num_frames):
            X_features[sample_index, frame_index] = extract_frame_features(
                X[sample_index, frame_index],
                contact_threshold=contact_threshold,
            )

    return X_features


def extract_feature_combination(
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> np.ndarray:
    """
    將原始 16 維 pressure sensor 與指定的工程化特徵結合為新特徵表示。

    Parameters
    ----------
    X : ndarray
        單一 frame 形狀為 (16,)，或整個資料集形狀為 (samples, 50, 16)。
    feature_names : Sequence[str] | None
        要附加的工程化特徵名稱，例如 ('pressure_sum',) 或 ('cop_x', 'cop_y')。
    contact_threshold : float
        判定為接觸的壓力門檻值。

    Returns
    -------
    ndarray
        單一 frame 的特徵向量，或資料集形狀為 (samples, 50, D)。
    """

    X = np.asarray(X, dtype=np.float32)

    if feature_names is None:
        feature_names = []

    feature_names = list(feature_names)

    for feature_name in feature_names:
        if feature_name not in ENGINEERED_FEATURE_INDEX:
            raise ValueError(f"Unknown feature name: {feature_name}")

    if X.ndim == 1:
        if X.shape != (RAW_SENSOR_COUNT,):
            raise ValueError(f"Expected frame shape ({RAW_SENSOR_COUNT},), got {X.shape}")

        if not feature_names:
            return X.copy()

        engineered_features = extract_frame_features(
            X,
            contact_threshold=contact_threshold,
        )
        selected_features = np.array(
            [engineered_features[ENGINEERED_FEATURE_INDEX[name]] for name in feature_names],
            dtype=np.float32,
        )
        return np.concatenate((X, selected_features)).astype(np.float32)

    if X.ndim == 3:
        if X.shape[2] != RAW_SENSOR_COUNT:
            raise ValueError(f"Expected last dimension {RAW_SENSOR_COUNT}, got {X.shape}")

        num_samples = X.shape[0]
        num_frames = X.shape[1]
        feature_dim = RAW_SENSOR_COUNT + len(feature_names)
        X_combined = np.zeros((num_samples, num_frames, feature_dim), dtype=np.float32)

        for sample_index in range(num_samples):
            for frame_index in range(num_frames):
                X_combined[sample_index, frame_index] = extract_feature_combination(
                    X[sample_index, frame_index],
                    feature_names=feature_names,
                    contact_threshold=contact_threshold,
                )

        return X_combined

    raise ValueError(f"Expected 1D or 3D input, got shape {X.shape}")


def prepare_feature_vectors(
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD,
    sensor_min: Optional[np.ndarray] = None,
    sensor_max: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    將原始 samples 轉成共用的特徵向量，並依照指定的 normalization statistics 做正規化。

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        (flattened_vectors, normalized_features, sensor_min, sensor_max)
    """

    if feature_names is None:
        feature_names = ENGINEERED_FEATURE_ORDER

    X = np.asarray(X, dtype=np.float32)
    features = extract_feature_combination(
        X,
        feature_names=feature_names,
        contact_threshold=contact_threshold,
    )

    if sensor_min is None or sensor_max is None:
        flat_features = features.reshape(-1, features.shape[-1])
        sensor_min = flat_features.min(axis=0).astype(np.float32)
        sensor_max = flat_features.max(axis=0).astype(np.float32)

    normalized_features = normalize_samples(features, sensor_min, sensor_max)
    flattened_vectors = flatten_samples(normalized_features).astype(np.float32)

    return flattened_vectors, normalized_features, sensor_min, sensor_max


def extract_hybrid_features(
    X: np.ndarray,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> np.ndarray:
    """
    將原始 16 維 pressure sensor 與 7 維工程化特徵結合為 23 維 Hybrid Feature。

    Parameters
    ----------
    X : ndarray
        單一 frame 形狀為 (16,)，或整個資料集形狀為 (samples, 50, 16)。
    contact_threshold : float
        判定為接觸的壓力門檻值。

    Returns
    -------
    ndarray
        單一 frame 的 23 維向量，或資料集形狀為 (samples, 50, 23)。
    """

    return extract_feature_combination(
        X,
        feature_names=ENGINEERED_FEATURE_ORDER,
        contact_threshold=contact_threshold,
    )
