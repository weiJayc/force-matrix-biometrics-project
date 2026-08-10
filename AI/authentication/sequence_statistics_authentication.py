from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np


# ============================================================
# Project Path
# ============================================================

AI_ROOT = Path(__file__).resolve().parents[1]

if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))


from data_loader import load_dataset
from authentication.feature_extractor import (
    ENGINEERED_FEATURE_ORDER,
    extract_feature_combination,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"

DEFAULT_USERS = (
    "amber",
    "jay",
    "666",
    "background",
)

RAW_SENSOR_COUNT = 16
FRAME_COUNT = 50
BASE_FEATURE_COUNT = 16 + len(ENGINEERED_FEATURE_ORDER)

# Each base feature gets these 10 sequence-level statistics.
SEQUENCE_STATISTIC_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "auc",
    "peak_position",
    "first_last_delta",
    "max_abs_change",
)

SEQUENCE_STATISTIC_COUNT = len(SEQUENCE_STATISTIC_NAMES)
SEQUENCE_FEATURE_COUNT = BASE_FEATURE_COUNT * SEQUENCE_STATISTIC_COUNT

DEFAULT_K_VALUE = 2.0


# ============================================================
# Dataset helpers
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the original dataset."""
    return load_dataset(DATA_DIR)


def get_user_samples(
    X: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
) -> np.ndarray:
    """Return all samples belonging to one user."""
    mask = np.asarray(raw_y) == user_id
    return X[mask]


def get_other_samples(
    X: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
) -> np.ndarray:
    """Return all samples not belonging to one user."""
    mask = np.asarray(raw_y) != user_id
    return X[mask]


# ============================================================
# Sequence feature extraction
# ============================================================

def _safe_auc(values: np.ndarray) -> float:
    """
    Calculate normalized area under the curve.

    The time axis is normalized to [0, 1], so AUC is not affected
    by the absolute number of frames.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size <= 1:
        return float(values[0]) if values.size == 1 else 0.0

    time_axis = np.linspace(
        0.0,
        1.0,
        values.size,
        dtype=np.float32,
    )

    return float(np.trapezoid(values, time_axis))


def _safe_peak_position(values: np.ndarray) -> float:
    """
    Return the normalized position of the maximum value.

    0.0 = beginning of the grip
    1.0 = end of the grip
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size <= 1:
        return 0.0

    peak_index = int(np.argmax(values))

    return float(peak_index / (values.size - 1))


def _safe_max_abs_change(values: np.ndarray) -> float:
    """
    Return the largest absolute frame-to-frame change.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size <= 1:
        return 0.0

    differences = np.diff(values)

    return float(np.max(np.abs(differences)))


def extract_sequence_statistics(
    sample: np.ndarray,
) -> np.ndarray:
    """
    Convert one complete grip recording into a sequence-level
    statistical feature vector.

    Input
    -----
    sample:
        Shape (50, 16)

    Processing
    ----------
    1. Convert every frame from 16 raw pressure values
       into 23 base features:
           - 16 raw sensor values
           - 7 engineered features

    2. For each of the 23 time-series features calculate:
           mean
           std
           min
           max
           range
           median
           auc
           peak_position
           first_last_delta
           max_abs_change

    Output
    ------
    Shape:
        (230,)
    """

    sample = np.asarray(sample, dtype=np.float32)

    if sample.ndim != 2:
        raise ValueError(
            f"Expected sample with 2 dimensions, got {sample.shape}"
        )

    if sample.shape[1] != RAW_SENSOR_COUNT:
        raise ValueError(
            f"Expected {RAW_SENSOR_COUNT} sensors, got {sample.shape}"
        )

    # --------------------------------------------------------
    # Convert every frame:
    #
    # 16 raw values
    # +
    # 7 engineered features
    #
    # => 23 features
    # --------------------------------------------------------

    base_features = extract_feature_combination(
        sample[np.newaxis, :, :],
        feature_names=ENGINEERED_FEATURE_ORDER,
    )[0]

    if base_features.shape[0] != FRAME_COUNT:
        raise ValueError(
            f"Expected {FRAME_COUNT} frames, got {base_features.shape[0]}"
        )

    if base_features.shape[1] != BASE_FEATURE_COUNT:
        raise ValueError(
            f"Expected {BASE_FEATURE_COUNT} base features, "
            f"got {base_features.shape[1]}"
        )

    sequence_features: list[float] = []

    # --------------------------------------------------------
    # Calculate statistics for each of the 23 features
    # --------------------------------------------------------

    for feature_index in range(BASE_FEATURE_COUNT):

        values = base_features[:, feature_index].astype(
            np.float32,
            copy=False,
        )

        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        range_value = max_value - min_value
        median_value = float(np.median(values))
        auc_value = _safe_auc(values)
        peak_position = _safe_peak_position(values)

        first_last_delta = float(
            values[-1] - values[0]
        )

        max_abs_change = _safe_max_abs_change(values)

        sequence_features.extend(
            [
                mean_value,
                std_value,
                min_value,
                max_value,
                range_value,
                median_value,
                auc_value,
                peak_position,
                first_last_delta,
                max_abs_change,
            ]
        )

    result = np.asarray(
        sequence_features,
        dtype=np.float32,
    )

    if result.shape != (SEQUENCE_FEATURE_COUNT,):
        raise ValueError(
            f"Unexpected sequence feature shape: {result.shape}"
        )

    if np.isnan(result).any():
        raise ValueError(
            "NaN detected in sequence-level features"
        )

    return result


def extract_dataset_sequence_statistics(
    X: np.ndarray,
) -> np.ndarray:
    """
    Convert the complete dataset into sequence-level features.

    Input:
        (samples, 50, 16)

    Output:
        (samples, 230)
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D dataset, got {X.shape}"
        )

    if X.shape[1] != FRAME_COUNT:
        raise ValueError(
            f"Expected {FRAME_COUNT} frames, got {X.shape}"
        )

    if X.shape[2] != RAW_SENSOR_COUNT:
        raise ValueError(
            f"Expected {RAW_SENSOR_COUNT} sensors, got {X.shape}"
        )

    sequence_features = np.zeros(
        (
            X.shape[0],
            SEQUENCE_FEATURE_COUNT,
        ),
        dtype=np.float32,
    )

    for sample_index in range(X.shape[0]):
        sequence_features[sample_index] = (
            extract_sequence_statistics(
                X[sample_index]
            )
        )

    return sequence_features


# ============================================================
# Per-user normalization
# ============================================================

def fit_user_normalization(
    registration_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate normalization statistics using ONLY the user's
    registration samples.

    Returns:
        normalized_registration
        feature_min
        feature_max
    """

    registration_features = np.asarray(
        registration_features,
        dtype=np.float32,
    )

    if registration_features.ndim != 2:
        raise ValueError(
            f"Expected 2D features, got {registration_features.shape}"
        )

    feature_min = np.min(
        registration_features,
        axis=0,
    ).astype(np.float32)

    feature_max = np.max(
        registration_features,
        axis=0,
    ).astype(np.float32)

    feature_range = feature_max - feature_min

    normalized = np.zeros_like(
        registration_features,
        dtype=np.float32,
    )

    usable_mask = feature_range != 0

    normalized[:, usable_mask] = (
        registration_features[:, usable_mask]
        - feature_min[usable_mask]
    ) / feature_range[usable_mask]

    # Constant features contain no information for distinguishing
    # samples inside this user's registration data.
    normalized[:, ~usable_mask] = 0.0

    return (
        normalized,
        feature_min,
        feature_max,
    )


def apply_user_normalization(
    features: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
) -> np.ndarray:
    """
    Normalize new samples using the min/max learned from
    the user's registration data.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    feature_min = np.asarray(
        feature_min,
        dtype=np.float32,
    )

    feature_max = np.asarray(
        feature_max,
        dtype=np.float32,
    )

    feature_range = feature_max - feature_min

    normalized = np.zeros_like(
        features,
        dtype=np.float32,
    )

    usable_mask = feature_range != 0

    normalized[:, usable_mask] = (
        features[:, usable_mask]
        - feature_min[usable_mask]
    ) / feature_range[usable_mask]

    normalized[:, ~usable_mask] = 0.0

    return normalized


# ============================================================
# Template / Threshold
# ============================================================

def compute_threshold(
    distances: np.ndarray,
    k_value: float = DEFAULT_K_VALUE,
) -> float:
    """
    Threshold = mean(distance) + k * std(distance)
    """

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    if distances.size == 0:
        raise ValueError(
            "Cannot calculate threshold from empty distances"
        )

    return float(
        np.mean(distances)
        + k_value * np.std(distances)
    )


def build_sequence_template(
    registration_features: np.ndarray,
    k_value: float = DEFAULT_K_VALUE,
) -> dict:
    """
    Build a sequence-level template from registration samples.
    """

    (
        normalized_registration,
        feature_min,
        feature_max,
    ) = fit_user_normalization(
        registration_features
    )

    feature_range = feature_max - feature_min
    usable_mask = feature_range != 0

    usable_registration = normalized_registration[
        :,
        usable_mask,
    ]

    template = np.mean(
        usable_registration,
        axis=0,
    ).astype(np.float32)

    registration_distances = np.linalg.norm(
        usable_registration - template,
        axis=1,
    ).astype(np.float32)

    threshold = compute_threshold(
        registration_distances,
        k_value=k_value,
    )

    return {
        "template": template,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "usable_mask": usable_mask,
        "threshold": threshold,
        "registration_distances": registration_distances,
    }


# ============================================================
# Authentication
# ============================================================

def calculate_distance(
    template_data: dict,
    sample_features: np.ndarray,
) -> float:
    """
    Calculate distance between one sample and a user's template.

    The sample is normalized using the registration user's
    min/max statistics.
    """

    normalized = apply_user_normalization(
        sample_features,
        template_data["feature_min"],
        template_data["feature_max"],
    )

    usable_mask = template_data["usable_mask"]

    normalized_sample = normalized[
        :,
        usable_mask,
    ]

    template = template_data["template"]

    distance = float(
        np.linalg.norm(
            normalized_sample[0] - template
        )
    )

    return distance


def authenticate_sample(
    template_data: dict,
    sample_features: np.ndarray,
) -> tuple[float, bool]:
    """
    Authenticate one sequence-level sample.
    """

    distance = calculate_distance(
        template_data,
        sample_features,
    )

    threshold = float(
        template_data["threshold"]
    )

    accepted = distance <= threshold

    return distance, accepted


# ============================================================
# Registration -> Authentication
# ============================================================

def build_template_for_user(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
    registration_count: int | None = None,
    k_value: float = DEFAULT_K_VALUE,
) -> dict:

    user_samples = sequence_features[
        np.asarray(raw_y) == user_id
    ]

    if user_samples.size == 0:
        raise ValueError(
            f"No samples found for user: {user_id}"
        )

    if registration_count is not None:

        if registration_count <= 0:
            raise ValueError(
                "registration_count must be greater than 0"
            )

        if registration_count > len(user_samples):
            raise ValueError(
                f"User {user_id} only has "
                f"{len(user_samples)} samples, "
                f"but registration_count={registration_count}"
            )

        registration_samples = user_samples[
            :registration_count
        ]

    else:
        registration_samples = user_samples

    return build_sequence_template(
        registration_samples,
        k_value=k_value,
    )


def evaluate_template_authentication(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    template_user: str,
    test_users: Iterable[str] | None = None,
    registration_count: int | None = None,
    k_value: float = DEFAULT_K_VALUE,
) -> list[dict[str, object]]:

    template_data = build_template_for_user(
        sequence_features,
        raw_y,
        template_user,
        registration_count=registration_count,
        k_value=k_value,
    )

    if test_users is None:
        test_users = DEFAULT_USERS

    results: list[dict[str, object]] = []

    for test_user in test_users:

        test_samples = sequence_features[
            np.asarray(raw_y) == test_user
        ]

        distances: list[float] = []

        accept_count = 0
        reject_count = 0

        for sample_features in test_samples:

            distance, accepted = authenticate_sample(
                template_data,
                sample_features[np.newaxis, :],
            )

            distances.append(distance)

            if accepted:
                accept_count += 1
            else:
                reject_count += 1

        acceptance_rate = (
            accept_count / len(test_samples)
            if len(test_samples)
            else 0.0
        )

        results.append(
            {
                "template_user": template_user,
                "test_user": test_user,
                "accept": accept_count,
                "reject": reject_count,
                "acceptance_rate": acceptance_rate,
                "average_distance": (
                    float(np.mean(distances))
                    if distances
                    else 0.0
                ),
            }
        )

    return results


# ============================================================
# FAR / FRR
# ============================================================

def evaluate_far_frr(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
    registration_count: int | None = None,
    k_value: float = DEFAULT_K_VALUE,
) -> dict[str, float]:

    template_data = build_template_for_user(
        sequence_features,
        raw_y,
        user_id,
        registration_count=registration_count,
        k_value=k_value,
    )

    user_mask = np.asarray(raw_y) == user_id
    impostor_mask = np.asarray(raw_y) != user_id

    genuine_samples = sequence_features[user_mask]
    impostor_samples = sequence_features[impostor_mask]

    genuine_distances: list[float] = []
    impostor_distances: list[float] = []

    for sample_features in genuine_samples:

        distance, _ = authenticate_sample(
            template_data,
            sample_features[np.newaxis, :],
        )

        genuine_distances.append(distance)

    for sample_features in impostor_samples:

        distance, _ = authenticate_sample(
            template_data,
            sample_features[np.newaxis, :],
        )

        impostor_distances.append(distance)

    genuine_array = np.asarray(
        genuine_distances,
        dtype=np.float32,
    )

    impostor_array = np.asarray(
        impostor_distances,
        dtype=np.float32,
    )

    threshold = float(
        template_data["threshold"]
    )

    far = float(
        np.mean(impostor_array <= threshold)
    )

    frr = float(
        np.mean(genuine_array > threshold)
    )

    acceptance_rate = float(
        np.mean(genuine_array <= threshold)
    )

    return {
        "user": user_id,
        "registration_samples": (
            registration_count
            if registration_count is not None
            else int(np.sum(user_mask))
        ),
        "threshold": threshold,
        "far": far,
        "frr": frr,
        "acceptance_rate": acceptance_rate,
        "genuine_mean_distance": float(
            np.mean(genuine_array)
        ),
        "genuine_std_distance": float(
            np.std(genuine_array)
        ),
        "impostor_mean_distance": float(
            np.mean(impostor_array)
        ),
        "impostor_std_distance": float(
            np.std(impostor_array)
        ),
    }


# ============================================================
# Few-Shot Experiment
# ============================================================

def run_few_shot_experiment(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    users: Iterable[str] | None = None,
    registration_counts: Iterable[int] = (
        5,
        10,
        20,
        50,
    ),
    k_value: float = DEFAULT_K_VALUE,
) -> list[dict[str, object]]:

    if users is None:
        users = DEFAULT_USERS

    rows: list[dict[str, object]] = []

    for user_id in users:

        for count in registration_counts:

            row = evaluate_far_frr(
                sequence_features,
                raw_y,
                user_id,
                registration_count=count,
                k_value=k_value,
            )

            rows.append(row)

    return rows


# ============================================================
# Printing
# ============================================================

def print_feature_information() -> None:

    print()
    print("========== Sequence Statistics Feature Information ==========")
    print(
        f"Base features              : "
        f"{BASE_FEATURE_COUNT}"
    )
    print(
        f"Statistics per feature    : "
        f"{SEQUENCE_STATISTIC_COUNT}"
    )
    print(
        f"Final sequence feature dim: "
        f"{SEQUENCE_FEATURE_COUNT}"
    )

    print()
    print("Statistics:")
    for name in SEQUENCE_STATISTIC_NAMES:
        print(f"  - {name}")


def print_template_results(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "========== Sequence Statistics "
        "Registration -> Authentication =========="
    )

    print(
        f"{'Template User':<14} "
        f"{'Test User':<14} "
        f"{'Accept':<8} "
        f"{'Reject':<8} "
        f"{'Acceptance Rate':<18} "
        f"{'Average Distance':<18}"
    )

    print("-" * 110)

    for row in results:

        print(
            f"{str(row['template_user']):<14} "
            f"{str(row['test_user']):<14} "
            f"{int(row['accept']):<8} "
            f"{int(row['reject']):<8} "
            f"{float(row['acceptance_rate']) * 100:>8.2f}% "
            f"{float(row['average_distance']):>14.4f}"
        )


def print_far_frr_results(
    rows: list[dict[str, object]],
) -> None:

    print()
    print(
        "========== Sequence Statistics "
        "FAR / FRR =========="
    )

    print(
        f"{'User':<14} "
        f"{'Reg Samples':<14} "
        f"{'Threshold':<12} "
        f"{'FAR':<10} "
        f"{'FRR':<10} "
        f"{'Acceptance Rate':<18} "
        f"{'Genuine Mean':<16} "
        f"{'Impostor Mean':<16}"
    )

    print("-" * 130)

    for row in rows:

        print(
            f"{str(row['user']):<14} "
            f"{int(row['registration_samples']):<14} "
            f"{float(row['threshold']):<12.4f} "
            f"{float(row['far']):<10.4f} "
            f"{float(row['frr']):<10.4f} "
            f"{float(row['acceptance_rate']):<18.4f} "
            f"{float(row['genuine_mean_distance']):<16.4f} "
            f"{float(row['impostor_mean_distance']):<16.4f}"
        )


def print_few_shot_results(
    rows: list[dict[str, object]],
) -> None:

    print()
    print(
        "========== Sequence Statistics "
        "Few-Shot Experiment =========="
    )

    print(
        f"{'User':<14} "
        f"{'Registration Samples':<22} "
        f"{'Threshold':<12} "
        f"{'FAR':<10} "
        f"{'FRR':<10} "
        f"{'Acceptance Rate':<18}"
    )

    print("-" * 105)

    for row in rows:

        print(
            f"{str(row['user']):<14} "
            f"{int(row['registration_samples']):<22} "
            f"{float(row['threshold']):<12.4f} "
            f"{float(row['far']):<10.4f} "
            f"{float(row['frr']):<10.4f} "
            f"{float(row['acceptance_rate']):<18.4f}"
        )


def print_dataset_information(
    X: np.ndarray,
    raw_y: np.ndarray,
) -> None:

    print()
    print("========== Dataset Information ==========")

    print(f"Original X shape: {X.shape}")

    labels, counts = np.unique(
        raw_y,
        return_counts=True,
    )

    for label, count in zip(labels, counts):
        print(f"{str(label):<14}: {int(count)} samples")


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("============================================================")
    print(" Sequence-Level Statistical Authentication Benchmark")
    print("============================================================")

    X, raw_y = load_data()

    print_dataset_information(
        X,
        raw_y,
    )

    # --------------------------------------------------------
    # Extract sequence-level features
    # --------------------------------------------------------

    sequence_features = (
        extract_dataset_sequence_statistics(X)
    )

    print()
    print(
        f"Sequence feature shape: "
        f"{sequence_features.shape}"
    )

    if np.isnan(sequence_features).any():
        raise ValueError(
            "NaN detected in sequence feature dataset"
        )

    print_feature_information()

    # --------------------------------------------------------
    # Registration -> Authentication
    #
    # This intentionally uses the same full-dataset setup
    # as your previous benchmark so you can compare the
    # methods directly.
    # --------------------------------------------------------

    for template_user in DEFAULT_USERS:

        results = evaluate_template_authentication(
            sequence_features,
            raw_y,
            template_user,
            test_users=DEFAULT_USERS,
            registration_count=None,
            k_value=DEFAULT_K_VALUE,
        )

        print_template_results(results)

    # --------------------------------------------------------
    # Per-user threshold
    # --------------------------------------------------------

    threshold_rows: list[dict[str, object]] = []

    for user_id in DEFAULT_USERS:

        template_data = build_template_for_user(
            sequence_features,
            raw_y,
            user_id,
            registration_count=None,
            k_value=DEFAULT_K_VALUE,
        )

        registration_distances = (
            template_data["registration_distances"]
        )

        threshold_rows.append(
            {
                "user": user_id,
                "threshold": float(
                    template_data["threshold"]
                ),
                "mean_distance": float(
                    np.mean(registration_distances)
                ),
                "std": float(
                    np.std(registration_distances)
                ),
            }
        )

    print()
    print(
        "========== Sequence Statistics "
        "Per-User Threshold =========="
    )

    print(
        f"{'User':<14} "
        f"{'Threshold':<12} "
        f"{'Mean Distance':<16} "
        f"{'Std':<12}"
    )

    print("-" * 65)

    for row in threshold_rows:

        print(
            f"{str(row['user']):<14} "
            f"{float(row['threshold']):<12.4f} "
            f"{float(row['mean_distance']):<16.4f} "
            f"{float(row['std']):<12.4f}"
        )

    # --------------------------------------------------------
    # FAR / FRR using all available samples
    # --------------------------------------------------------

    far_frr_rows: list[dict[str, object]] = []

    for user_id in DEFAULT_USERS:

        far_frr_rows.append(
            evaluate_far_frr(
                sequence_features,
                raw_y,
                user_id,
                registration_count=None,
                k_value=DEFAULT_K_VALUE,
            )
        )

    print_far_frr_results(
        far_frr_rows
    )

    # --------------------------------------------------------
    # Few-Shot experiment
    # --------------------------------------------------------

    few_shot_rows = run_few_shot_experiment(
        sequence_features,
        raw_y,
        users=DEFAULT_USERS,
        registration_counts=(
            5,
            10,
            20,
            50,
        ),
        k_value=DEFAULT_K_VALUE,
    )

    print_few_shot_results(
        few_shot_rows
    )

    print()
    print("============================================================")
    print(" Benchmark Finished")
    print("============================================================")


if __name__ == "__main__":
    main()