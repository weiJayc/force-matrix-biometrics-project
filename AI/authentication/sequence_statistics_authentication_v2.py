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


# ------------------------------------------------------------
# Sequence-level statistics
# ------------------------------------------------------------

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

SEQUENCE_FEATURE_COUNT = (
    BASE_FEATURE_COUNT * SEQUENCE_STATISTIC_COUNT
)


# ------------------------------------------------------------
# Formal access-control benchmark
# ------------------------------------------------------------

FORMAL_REGISTRATION_COUNT = 20
FORMAL_TEST_COUNT = 30

# 30 impostor samples:
# 10 from each of the other 3 classes
IMPOSTOR_TEST_PER_USER = 10

N_SPLITS = 20

RANDOM_SEED = 42

DEFAULT_K_VALUE = 2.0


# ------------------------------------------------------------
# Few-shot experiment
# ------------------------------------------------------------

FEW_SHOT_REGISTRATION_COUNTS = (
    5,
    10,
    20,
)


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


# ============================================================
# Sequence feature extraction
# ============================================================

def _safe_auc(values: np.ndarray) -> float:
    """Calculate normalized AUC."""

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
    Return normalized maximum position.

    0.0 = beginning
    1.0 = end
    """

    values = np.asarray(values, dtype=np.float32)

    if values.size <= 1:
        return 0.0

    peak_index = int(np.argmax(values))

    return float(
        peak_index / (values.size - 1)
    )


def _safe_max_abs_change(values: np.ndarray) -> float:
    """Return maximum absolute frame-to-frame change."""

    values = np.asarray(values, dtype=np.float32)

    if values.size <= 1:
        return 0.0

    differences = np.diff(values)

    return float(
        np.max(np.abs(differences))
    )


def extract_sequence_statistics(
    sample: np.ndarray,
) -> np.ndarray:
    """
    Convert one complete grip recording into
    sequence-level statistical features.

    Input:
        (50, 16)

    16 raw sensors
    + 7 engineered features
    = 23 base features

    Each base feature gets 10 statistics:

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

    Output:
        (253,)
    """

    sample = np.asarray(
        sample,
        dtype=np.float32,
    )

    if sample.ndim != 2:
        raise ValueError(
            f"Expected 2D sample, got {sample.shape}"
        )

    if sample.shape[0] != FRAME_COUNT:
        raise ValueError(
            f"Expected {FRAME_COUNT} frames, "
            f"got {sample.shape[0]}"
        )

    if sample.shape[1] != RAW_SENSOR_COUNT:
        raise ValueError(
            f"Expected {RAW_SENSOR_COUNT} sensors, "
            f"got {sample.shape[1]}"
        )

    base_features = extract_feature_combination(
        sample[np.newaxis, :, :],
        feature_names=ENGINEERED_FEATURE_ORDER,
    )[0]

    if base_features.shape != (
        FRAME_COUNT,
        BASE_FEATURE_COUNT,
    ):
        raise ValueError(
            f"Unexpected base feature shape: "
            f"{base_features.shape}"
        )

    sequence_features: list[float] = []

    for feature_index in range(BASE_FEATURE_COUNT):

        values = base_features[
            :,
            feature_index,
        ].astype(
            np.float32,
            copy=False,
        )

        mean_value = float(
            np.mean(values)
        )

        std_value = float(
            np.std(values)
        )

        min_value = float(
            np.min(values)
        )

        max_value = float(
            np.max(values)
        )

        range_value = (
            max_value - min_value
        )

        median_value = float(
            np.median(values)
        )

        auc_value = _safe_auc(values)

        peak_position = _safe_peak_position(
            values
        )

        first_last_delta = float(
            values[-1] - values[0]
        )

        max_abs_change = _safe_max_abs_change(
            values
        )

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

    if result.shape != (
        SEQUENCE_FEATURE_COUNT,
    ):
        raise ValueError(
            f"Unexpected sequence feature shape: "
            f"{result.shape}"
        )

    if np.isnan(result).any():
        raise ValueError(
            "NaN detected in sequence features"
        )

    return result


def extract_dataset_sequence_statistics(
    X: np.ndarray,
) -> np.ndarray:
    """
    Convert complete dataset.

    Input:
        (samples, 50, 16)

    Output:
        (samples, 253)
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D dataset, got {X.shape}"
        )

    if X.shape[1] != FRAME_COUNT:
        raise ValueError(
            f"Expected {FRAME_COUNT} frames, "
            f"got {X.shape}"
        )

    if X.shape[2] != RAW_SENSOR_COUNT:
        raise ValueError(
            f"Expected {RAW_SENSOR_COUNT} sensors, "
            f"got {X.shape}"
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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate min/max using ONLY registration data.
    """

    registration_features = np.asarray(
        registration_features,
        dtype=np.float32,
    )

    feature_min = np.min(
        registration_features,
        axis=0,
    ).astype(np.float32)

    feature_max = np.max(
        registration_features,
        axis=0,
    ).astype(np.float32)

    feature_range = (
        feature_max - feature_min
    )

    usable_mask = feature_range != 0

    normalized = np.zeros_like(
        registration_features,
        dtype=np.float32,
    )

    normalized[:, usable_mask] = (
        registration_features[:, usable_mask]
        - feature_min[usable_mask]
    ) / feature_range[usable_mask]

    normalized[:, ~usable_mask] = 0.0

    return (
        normalized,
        feature_min,
        feature_max,
        usable_mask,
    )


def apply_user_normalization(
    features: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
) -> np.ndarray:
    """
    Apply registration user's min/max
    to new samples.
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

    feature_range = (
        feature_max - feature_min
    )

    usable_mask = feature_range != 0

    normalized = np.zeros_like(
        features,
        dtype=np.float32,
    )

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

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    if distances.size == 0:
        raise ValueError(
            "Cannot calculate threshold "
            "from empty distances"
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
    Build template using ONLY registration samples.
    """

    (
        normalized_registration,
        feature_min,
        feature_max,
        usable_mask,
    ) = fit_user_normalization(
        registration_features
    )

    usable_registration = (
        normalized_registration[:, usable_mask]
    )

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
        "registration_distances":
            registration_distances,
    }


# ============================================================
# Authentication
# ============================================================

def calculate_distance(
    template_data: dict,
    sample_features: np.ndarray,
) -> float:

    normalized = apply_user_normalization(
        sample_features,
        template_data["feature_min"],
        template_data["feature_max"],
    )

    usable_mask = template_data[
        "usable_mask"
    ]

    normalized_sample = normalized[
        :,
        usable_mask,
    ]

    template = template_data[
        "template"
    ]

    return float(
        np.linalg.norm(
            normalized_sample[0]
            - template
        )
    )


def authenticate_sample(
    template_data: dict,
    sample_features: np.ndarray,
) -> tuple[float, bool]:

    distance = calculate_distance(
        template_data,
        sample_features,
    )

    threshold = float(
        template_data["threshold"]
    )

    accepted = (
        distance <= threshold
    )

    return distance, accepted


# ============================================================
# Template Identity
# ============================================================

def compare_template_to_users(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    template_data: dict,
    users: Iterable[str],
) -> list[dict[str, object]]:
    """
    Compare one template against all users.

    This tells us:
        "Who does this template look like?"
    """

    rows: list[dict[str, object]] = []

    for user_id in users:

        user_mask = (
            np.asarray(raw_y) == user_id
        )

        user_features = (
            sequence_features[user_mask]
        )

        distances = []

        for sample_features in user_features:

            distance, _ = authenticate_sample(
                template_data,
                sample_features[np.newaxis, :],
            )

            distances.append(distance)

        if distances:

            rows.append(
                {
                    "user": user_id,
                    "mean_distance": float(
                        np.mean(distances)
                    ),
                    "min_distance": float(
                        np.min(distances)
                    ),
                }
            )

    rows.sort(
        key=lambda row:
        float(row["mean_distance"])
    )

    return rows


# ============================================================
# Formal split
# ============================================================

def make_formal_split(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
    registration_count: int,
    test_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    user_indices = np.where(
        np.asarray(raw_y) == user_id
    )[0]

    if len(user_indices) < (
        registration_count + test_count
    ):
        raise ValueError(
            f"User {user_id} does not have enough "
            f"samples for registration={registration_count} "
            f"+ test={test_count}"
        )

    shuffled = rng.permutation(
        user_indices
    )

    registration_indices = shuffled[
        :registration_count
    ]

    test_indices = shuffled[
        registration_count:
        registration_count + test_count
    ]

    return (
        sequence_features[
            registration_indices
        ],
        sequence_features[
            test_indices
        ],
    )


def make_impostor_test_set(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    genuine_user: str,
    users: Iterable[str],
    samples_per_user: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:

    impostor_features: list[np.ndarray] = []
    impostor_labels: list[str] = []

    for user_id in users:

        if user_id == genuine_user:
            continue

        indices = np.where(
            np.asarray(raw_y) == user_id
        )[0]

        if len(indices) < samples_per_user:
            raise ValueError(
                f"User {user_id} does not have "
                f"enough samples for impostor test."
            )

        selected = rng.choice(
            indices,
            size=samples_per_user,
            replace=False,
        )

        impostor_features.append(
            sequence_features[selected]
        )

        impostor_labels.extend(
            [user_id] * samples_per_user
        )

    return (
        np.concatenate(
            impostor_features,
            axis=0,
        ),
        impostor_labels,
    )


# ============================================================
# Formal Access-Control Benchmark
# ============================================================

def run_formal_benchmark(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    users: Iterable[str] | None = None,
    registration_count: int = FORMAL_REGISTRATION_COUNT,
    test_count: int = FORMAL_TEST_COUNT,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
    k_value: float = DEFAULT_K_VALUE,
) -> list[dict[str, object]]:

    if users is None:
        users = DEFAULT_USERS

    users = tuple(users)

    all_rows: list[dict[str, object]] = []

    for split_index in range(
        1,
        n_splits + 1,
    ):

        rng = np.random.default_rng(
            seed + split_index
        )

        for user_id in users:

            (
                registration_features,
                genuine_test_features,
            ) = make_formal_split(
                sequence_features,
                raw_y,
                user_id,
                registration_count,
                test_count,
                rng,
            )

            template_data = build_sequence_template(
                registration_features,
                k_value=k_value,
            )

            (
                impostor_test_features,
                impostor_labels,
            ) = make_impostor_test_set(
                sequence_features,
                raw_y,
                user_id,
                users,
                IMPOSTOR_TEST_PER_USER,
                rng,
            )

            # ------------------------------------------------
            # Genuine test
            # ------------------------------------------------

            genuine_distances = []

            for sample_features in genuine_test_features:

                distance, _ = authenticate_sample(
                    template_data,
                    sample_features[np.newaxis, :],
                )

                genuine_distances.append(
                    distance
                )

            # ------------------------------------------------
            # Impostor test
            # ------------------------------------------------

            impostor_distances = []

            for sample_features in impostor_test_features:

                distance, _ = authenticate_sample(
                    template_data,
                    sample_features[np.newaxis, :],
                )

                impostor_distances.append(
                    distance
                )

            genuine_distances = np.asarray(
                genuine_distances,
                dtype=np.float32,
            )

            impostor_distances = np.asarray(
                impostor_distances,
                dtype=np.float32,
            )

            threshold = float(
                template_data["threshold"]
            )

            genuine_accept = (
                genuine_distances
                <= threshold
            )

            impostor_accept = (
                impostor_distances
                <= threshold
            )

            gar = float(
                np.mean(genuine_accept)
            )

            frr = float(
                np.mean(~genuine_accept)
            )

            far = float(
                np.mean(impostor_accept)
            )

            # ------------------------------------------------
            # Template identity
            # ------------------------------------------------

            template_comparison = (
                compare_template_to_users(
                    sequence_features,
                    raw_y,
                    template_data,
                    users,
                )
            )

            closest_user = (
                template_comparison[0]["user"]
            )

            closest_distance = float(
                template_comparison[0][
                    "mean_distance"
                ]
            )

            genuine_mean = float(
                np.mean(genuine_distances)
            )

            impostor_mean = float(
                np.mean(impostor_distances)
            )

            all_rows.append(
                {
                    "split": split_index,
                    "user": user_id,
                    "threshold": threshold,
                    "gar": gar,
                    "frr": frr,
                    "far": far,
                    "genuine_mean": genuine_mean,
                    "impostor_mean": impostor_mean,
                    "closest_user": closest_user,
                    "closest_distance": closest_distance,
                }
            )

    return all_rows


# ============================================================
# Few-Shot Benchmark
# ============================================================

def run_few_shot_benchmark(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    users: Iterable[str] | None = None,
    registration_counts: Iterable[int] = (
        5,
        10,
        20,
    ),
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
    k_value: float = DEFAULT_K_VALUE,
) -> list[dict[str, object]]:

    if users is None:
        users = DEFAULT_USERS

    users = tuple(users)

    rows: list[dict[str, object]] = []

    for registration_count in registration_counts:

        for split_index in range(
            1,
            n_splits + 1,
        ):

            rng = np.random.default_rng(
                seed
                + 1000
                + registration_count * 100
                + split_index
            )

            for user_id in users:

                user_indices = np.where(
                    np.asarray(raw_y) == user_id
                )[0]

                # We always keep 30 unseen genuine
                # samples whenever possible.
                test_count = min(
                    FORMAL_TEST_COUNT,
                    len(user_indices)
                    - registration_count,
                )

                if test_count <= 0:
                    continue

                shuffled = rng.permutation(
                    user_indices
                )

                registration_indices = (
                    shuffled[
                        :registration_count
                    ]
                )

                test_indices = shuffled[
                    registration_count:
                    registration_count
                    + test_count
                ]

                registration_features = (
                    sequence_features[
                        registration_indices
                    ]
                )

                genuine_test_features = (
                    sequence_features[
                        test_indices
                    ]
                )

                template_data = (
                    build_sequence_template(
                        registration_features,
                        k_value=k_value,
                    )
                )

                # ------------------------------------------------
                # Genuine
                # ------------------------------------------------

                genuine_distances = []

                for sample_features in (
                    genuine_test_features
                ):

                    distance, _ = (
                        authenticate_sample(
                            template_data,
                            sample_features[
                                np.newaxis,
                                :,
                            ],
                        )
                    )

                    genuine_distances.append(
                        distance
                    )

                # ------------------------------------------------
                # Impostors
                # ------------------------------------------------

                impostor_features, _ = (
                    make_impostor_test_set(
                        sequence_features,
                        raw_y,
                        user_id,
                        users,
                        IMPOSTOR_TEST_PER_USER,
                        rng,
                    )
                )

                impostor_distances = []

                for sample_features in (
                    impostor_features
                ):

                    distance, _ = (
                        authenticate_sample(
                            template_data,
                            sample_features[
                                np.newaxis,
                                :,
                            ],
                        )
                    )

                    impostor_distances.append(
                        distance
                    )

                genuine_distances = np.asarray(
                    genuine_distances,
                    dtype=np.float32,
                )

                impostor_distances = np.asarray(
                    impostor_distances,
                    dtype=np.float32,
                )

                threshold = float(
                    template_data["threshold"]
                )

                genuine_accept = (
                    genuine_distances
                    <= threshold
                )

                impostor_accept = (
                    impostor_distances
                    <= threshold
                )

                gar = float(
                    np.mean(genuine_accept)
                )

                frr = float(
                    np.mean(~genuine_accept)
                )

                far = float(
                    np.mean(impostor_accept)
                )

                rows.append(
                    {
                        "split": split_index,
                        "user": user_id,
                        "registration_samples":
                            registration_count,
                        "test_samples":
                            test_count,
                        "threshold":
                            threshold,
                        "gar":
                            gar,
                        "frr":
                            frr,
                        "far":
                            far,
                    }
                )

    return rows


# ============================================================
# Printing: Template Identity
# ============================================================

def print_template_identity(
    rows: list[dict[str, object]],
) -> None:

    print()
    print(
        "============================================================"
    )
    print(
        " Template Identity Benchmark"
    )
    print(
        "============================================================"
    )

    print(
        f"{'User':<14}"
        f"{'Closest User':<18}"
        f"{'Closest Distance':<18}"
        f"{'Genuine Mean':<16}"
        f"{'Impostor Mean':<16}"
    )

    print("-" * 90)

    users = sorted(
        set(
            str(row["user"])
            for row in rows
        )
    )

    for user_id in users:

        user_rows = [
            row
            for row in rows
            if row["user"] == user_id
        ]

        # Most common closest identity
        closest_users = [
            str(row["closest_user"])
            for row in user_rows
        ]

        counts = {
            name: closest_users.count(name)
            for name in set(closest_users)
        }

        most_common = max(
            counts,
            key=counts.get,
        )

        matching_distances = [
            float(row["closest_distance"])
            for row in user_rows
        ]

        genuine_means = [
            float(row["genuine_mean"])
            for row in user_rows
        ]

        impostor_means = [
            float(row["impostor_mean"])
            for row in user_rows
        ]

        print(
            f"{user_id:<14}"
            f"{most_common:<18}"
            f"{np.mean(matching_distances):<18.4f}"
            f"{np.mean(genuine_means):<16.4f}"
            f"{np.mean(impostor_means):<16.4f}"
        )


# ============================================================
# Printing: Formal Benchmark
# ============================================================

def print_formal_results(
    rows: list[dict[str, object]],
) -> None:

    print()
    print(
        "============================================================"
    )
    print(
        " Formal Access-Control Benchmark"
    )
    print(
        "============================================================"
    )

    print(
        f"Registration samples : "
        f"{FORMAL_REGISTRATION_COUNT}"
    )

    print(
        f"Genuine test samples : "
        f"{FORMAL_TEST_COUNT}"
    )

    print(
        f"Impostor test samples: "
        f"{FORMAL_TEST_COUNT}"
    )

    print(
        f"Random splits        : "
        f"{N_SPLITS}"
    )

    print()

    print(
        f"{'Split':<8}"
        f"{'User':<14}"
        f"{'Threshold':<12}"
        f"{'GAR':<10}"
        f"{'FRR':<10}"
        f"{'FAR':<10}"
    )

    print("-" * 70)

    for row in rows:

        print(
            f"{int(row['split']):<8}"
            f"{str(row['user']):<14}"
            f"{float(row['threshold']):<12.4f}"
            f"{float(row['gar']) * 100:<10.2f}"
            f"{float(row['frr']) * 100:<10.2f}"
            f"{float(row['far']) * 100:<10.2f}"
        )


# ============================================================
# Printing: Formal Summary
# ============================================================

def print_formal_summary(
    rows: list[dict[str, object]],
) -> None:

    print()
    print(
        "============================================================"
    )
    print(
        " Formal Benchmark Summary"
    )
    print(
        "============================================================"
    )

    print(
        f"{'User':<14}"
        f"{'GAR Mean':<12}"
        f"{'GAR Std':<12}"
        f"{'FRR Mean':<12}"
        f"{'FRR Std':<12}"
        f"{'FAR Mean':<12}"
        f"{'FAR Std':<12}"
        f"{'Threshold':<12}"
    )

    print("-" * 100)

    users = sorted(
        set(
            str(row["user"])
            for row in rows
        )
    )

    all_gar = []
    all_frr = []
    all_far = []

    for user_id in users:

        user_rows = [
            row
            for row in rows
            if row["user"] == user_id
        ]

        gar = np.asarray(
            [
                float(row["gar"])
                for row in user_rows
            ]
        )

        frr = np.asarray(
            [
                float(row["frr"])
                for row in user_rows
            ]
        )

        far = np.asarray(
            [
                float(row["far"])
                for row in user_rows
            ]
        )

        threshold = np.asarray(
            [
                float(row["threshold"])
                for row in user_rows
            ]
        )

        all_gar.extend(gar)
        all_frr.extend(frr)
        all_far.extend(far)

        print(
            f"{user_id:<14}"
            f"{np.mean(gar) * 100:<12.2f}"
            f"{np.std(gar) * 100:<12.2f}"
            f"{np.mean(frr) * 100:<12.2f}"
            f"{np.std(frr) * 100:<12.2f}"
            f"{np.mean(far) * 100:<12.2f}"
            f"{np.std(far) * 100:<12.2f}"
            f"{np.mean(threshold):<12.4f}"
        )

    print()
    print(
        "---------------- Overall ----------------"
    )

    print(
        f"Average GAR : "
        f"{np.mean(all_gar) * 100:.2f}%"
    )

    print(
        f"Average FRR : "
        f"{np.mean(all_frr) * 100:.2f}%"
    )

    print(
        f"Average FAR : "
        f"{np.mean(all_far) * 100:.2f}%"
    )

    print()
    print(
        f"GAR Std     : "
        f"{np.std(all_gar) * 100:.2f}%"
    )

    print(
        f"FRR Std     : "
        f"{np.std(all_frr) * 100:.2f}%"
    )

    print(
        f"FAR Std     : "
        f"{np.std(all_far) * 100:.2f}%"
    )


# ============================================================
# Printing: Few-Shot
# ============================================================

def print_few_shot_summary(
    rows: list[dict[str, object]],
) -> None:

    print()
    print(
        "============================================================"
    )
    print(
        " Few-Shot Registration Benchmark"
    )
    print(
        "============================================================"
    )

    print(
        f"{'Reg Samples':<14}"
        f"{'User':<14}"
        f"{'GAR Mean':<12}"
        f"{'GAR Std':<12}"
        f"{'FRR Mean':<12}"
        f"{'FRR Std':<12}"
        f"{'FAR Mean':<12}"
        f"{'FAR Std':<12}"
        f"{'Threshold':<12}"
    )

    print("-" * 115)

    registration_counts = sorted(
        set(
            int(row["registration_samples"])
            for row in rows
        )
    )

    for count in registration_counts:

        for user_id in DEFAULT_USERS:

            user_rows = [
                row
                for row in rows
                if int(
                    row["registration_samples"]
                ) == count
                and row["user"] == user_id
            ]

            if not user_rows:
                continue

            gar = np.asarray(
                [
                    float(row["gar"])
                    for row in user_rows
                ]
            )

            frr = np.asarray(
                [
                    float(row["frr"])
                    for row in user_rows
                ]
            )

            far = np.asarray(
                [
                    float(row["far"])
                    for row in user_rows
                ]
            )

            threshold = np.asarray(
                [
                    float(row["threshold"])
                    for row in user_rows
                ]
            )

            print(
                f"{count:<14}"
                f"{user_id:<14}"
                f"{np.mean(gar) * 100:<12.2f}"
                f"{np.std(gar) * 100:<12.2f}"
                f"{np.mean(frr) * 100:<12.2f}"
                f"{np.std(frr) * 100:<12.2f}"
                f"{np.mean(far) * 100:<12.2f}"
                f"{np.std(far) * 100:<12.2f}"
                f"{np.mean(threshold):<12.4f}"
            )


# ============================================================
# Printing: Dataset
# ============================================================

def print_dataset_information(
    X: np.ndarray,
    raw_y: np.ndarray,
) -> None:

    print()
    print(
        "========== Dataset Information =========="
    )

    print(
        f"Original X shape: {X.shape}"
    )

    labels, counts = np.unique(
        raw_y,
        return_counts=True,
    )

    for label, count in zip(
        labels,
        counts,
    ):

        print(
            f"{str(label):<14}: "
            f"{int(count)} samples"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "============================================================"
    )
    print(
        " Sequence-Level Statistical Authentication Benchmark"
    )
    print(
        "============================================================"
    )

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

    print()

    print(
        "========== Sequence Feature Information =========="
    )

    print(
        f"Base features          : "
        f"{BASE_FEATURE_COUNT}"
    )

    print(
        f"Statistics per feature : "
        f"{SEQUENCE_STATISTIC_COUNT}"
    )

    print(
        f"Final feature dimension: "
        f"{SEQUENCE_FEATURE_COUNT}"
    )

    print()

    print(
        "Statistics:"
    )

    for name in SEQUENCE_STATISTIC_NAMES:
        print(
            f"  - {name}"
        )

    # ========================================================
    # Formal 20-registration / 30-test benchmark
    # ========================================================

    formal_rows = run_formal_benchmark(
        sequence_features,
        raw_y,
        users=DEFAULT_USERS,
        registration_count=20,
        test_count=30,
        n_splits=20,
        seed=RANDOM_SEED,
        k_value=DEFAULT_K_VALUE,
    )

    # --------------------------------------------------------
    # Show split-by-split results
    # --------------------------------------------------------

    print_formal_results(
        formal_rows
    )

    # --------------------------------------------------------
    # Show "Template looks like who?"
    # --------------------------------------------------------

    print_template_identity(
        formal_rows
    )

    # --------------------------------------------------------
    # Formal summary
    # --------------------------------------------------------

    print_formal_summary(
        formal_rows
    )

    # ========================================================
    # Few-Shot
    # ========================================================

    few_shot_rows = run_few_shot_benchmark(
        sequence_features,
        raw_y,
        users=DEFAULT_USERS,
        registration_counts=(
            5,
            10,
            20,
        ),
        n_splits=20,
        seed=RANDOM_SEED,
        k_value=DEFAULT_K_VALUE,
    )

    print_few_shot_summary(
        few_shot_rows
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        " Benchmark Finished"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()