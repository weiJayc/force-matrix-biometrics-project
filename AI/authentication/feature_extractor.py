from typing import Optional

import numpy as np


DEFAULT_CONTACT_THRESHOLD: float = 1000.0
FEATURE_COUNT: int = 7


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

    if frame.shape != (16,):
        raise ValueError(f"Expected frame shape (16,), got {frame.shape}")

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

    if X.shape[2] != 16:
        raise ValueError(f"Expected last dimension 16, got {X.shape}")

    num_samples = X.shape[0]
    num_frames = X.shape[1]

    X_features = np.zeros((num_samples, num_frames, FEATURE_COUNT), dtype=np.float32)

    for sample_index in range(num_samples):
        for frame_index in range(num_frames):
            X_features[sample_index, frame_index] = extract_frame_features(
                X[sample_index, frame_index],
                contact_threshold=contact_threshold,
            )

    return X_features
