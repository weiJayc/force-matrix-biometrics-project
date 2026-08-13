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

BASE_FEATURE_COUNT = (
    RAW_SENSOR_COUNT + len(ENGINEERED_FEATURE_ORDER)
)

# Sequence statistics
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

SEQUENCE_STATISTIC_COUNT = len(
    SEQUENCE_STATISTIC_NAMES
)

SEQUENCE_FEATURE_COUNT = (
    BASE_FEATURE_COUNT
    * SEQUENCE_STATISTIC_COUNT
)

# Formal authentication protocol
REGISTRATION_COUNT = 20
TEST_COUNT = 30

# Number of random registration/test splits
NUM_SPLITS = 20

# Reproducible random seed
RANDOM_SEED = 42

# Threshold = mean + k * std
DEFAULT_K_VALUE = 2.0


# ============================================================
# Dataset helpers
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the original dataset."""
    return load_dataset(DATA_DIR)


def get_user_indices(
    raw_y: np.ndarray,
    user_id: str,
) -> np.ndarray:
    """Return dataset indices belonging to one user."""
    raw_y = np.asarray(raw_y)

    return np.flatnonzero(raw_y == user_id)


# ============================================================
# Sequence-level statistics
# ============================================================

def _safe_auc(values: np.ndarray) -> float:
    """
    Calculate normalized area under the curve.

    The time axis is normalized to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size == 0:
        return 0.0

    if values.size == 1:
        return float(values[0])

    time_axis = np.linspace(
        0.0,
        1.0,
        values.size,
        dtype=np.float32,
    )

    return float(
        np.trapezoid(
            values,
            time_axis,
        )
    )


def _safe_peak_position(
    values: np.ndarray,
) -> float:
    """
    Return normalized position of maximum value.

    0.0 = beginning
    1.0 = end
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size <= 1:
        return 0.0

    peak_index = int(
        np.argmax(values)
    )

    return float(
        peak_index / (values.size - 1)
    )


def _safe_max_abs_change(
    values: np.ndarray,
) -> float:
    """
    Return largest absolute frame-to-frame change.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size <= 1:
        return 0.0

    differences = np.diff(values)

    return float(
        np.max(
            np.abs(differences)
        )
    )


def extract_sequence_statistics(
    sample: np.ndarray,
) -> np.ndarray:
    """
    Convert one complete grip recording into
    sequence-level statistical features.

    Input:
        (50, 16)

    Output:
        (230,)
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
            f"got {sample.shape}"
        )

    if sample.shape[1] != RAW_SENSOR_COUNT:
        raise ValueError(
            f"Expected {RAW_SENSOR_COUNT} sensors, "
            f"got {sample.shape}"
        )

    # --------------------------------------------------------
    # Convert each frame:
    #
    # 16 raw sensor values
    # +
    # 7 engineered features
    #
    # = 23 base features
    # --------------------------------------------------------

    base_features = extract_feature_combination(
        sample[np.newaxis, :, :],
        feature_names=ENGINEERED_FEATURE_ORDER,
    )[0]

    if base_features.shape != (
        FRAME_COUNT,
        BASE_FEATURE_COUNT,
    ):
        raise ValueError(
            "Unexpected base feature shape: "
            f"{base_features.shape}"
        )

    sequence_features: list[float] = []

    # --------------------------------------------------------
    # Calculate sequence statistics
    # for every base feature
    # --------------------------------------------------------

    for feature_index in range(
        BASE_FEATURE_COUNT
    ):

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

        auc_value = _safe_auc(
            values
        )

        peak_position = (
            _safe_peak_position(values)
        )

        first_last_delta = float(
            values[-1] - values[0]
        )

        max_abs_change = (
            _safe_max_abs_change(values)
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
            "Unexpected sequence feature shape: "
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
        (samples, 230)
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

    for sample_index in range(
        X.shape[0]
    ):
        sequence_features[
            sample_index
        ] = extract_sequence_statistics(
            X[sample_index]
        )

    return sequence_features


# ============================================================
# User-specific normalization
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
    Fit min/max ONLY from registration data.

    Returns:
        normalized_registration
        feature_min
        feature_max
        usable_mask
    """

    registration_features = np.asarray(
        registration_features,
        dtype=np.float32,
    )

    if registration_features.ndim != 2:
        raise ValueError(
            "Expected 2D registration features, "
            f"got {registration_features.shape}"
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

    usable_mask = (
        feature_range != 0
    )

    normalized = np.zeros_like(
        registration_features,
        dtype=np.float32,
    )

    normalized[
        :,
        usable_mask,
    ] = (
        registration_features[
            :,
            usable_mask,
        ]
        - feature_min[
            usable_mask
        ]
    ) / feature_range[
        usable_mask
    ]

    # Constant features contain no
    # discrimination information.
    normalized[
        :,
        ~usable_mask,
    ] = 0.0

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
    to new authentication samples.
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

    usable_mask = (
        feature_range != 0
    )

    normalized = np.zeros_like(
        features,
        dtype=np.float32,
    )

    normalized[
        :,
        usable_mask,
    ] = (
        features[
            :,
            usable_mask,
        ]
        - feature_min[
            usable_mask
        ]
    ) / feature_range[
        usable_mask
    ]

    normalized[
        :,
        ~usable_mask,
    ] = 0.0

    return normalized


# ============================================================
# Template / Threshold
# ============================================================

def compute_threshold(
    distances: np.ndarray,
    k_value: float = DEFAULT_K_VALUE,
) -> float:
    """
    Threshold = mean + k * std
    """

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
    Build a template using ONLY
    registration samples.
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
        normalized_registration[
            :,
            usable_mask,
        ]
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
    """
    Calculate distance from one sample
    to the registered user's template.
    """

    normalized = apply_user_normalization(
        sample_features,
        template_data["feature_min"],
        template_data["feature_max"],
    )

    usable_mask = template_data[
        "usable_mask"
    ]

    normalized_sample = (
        normalized[
            :,
            usable_mask,
        ]
    )

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

    accepted = (
        distance <= threshold
    )

    return distance, accepted


# ============================================================
# Formal split generation
# ============================================================

def create_random_split(
    raw_y: np.ndarray,
    user_id: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create ONE formal authentication split.

    Exactly:
        20 registration samples
        30 unknown genuine test samples

    Any remaining samples are unused.

    This is especially important for background,
    which currently contains 51 samples.
    """

    indices = get_user_indices(
        raw_y,
        user_id,
    )

    required_count = (
        REGISTRATION_COUNT
        + TEST_COUNT
    )

    if len(indices) < required_count:
        raise ValueError(
            f"User {user_id} has only "
            f"{len(indices)} samples, "
            f"but {required_count} are required."
        )

    shuffled = rng.permutation(
        indices
    )

    registration_indices = (
        shuffled[
            :REGISTRATION_COUNT
        ]
    )

    test_indices = (
        shuffled[
            REGISTRATION_COUNT:
            REGISTRATION_COUNT
            + TEST_COUNT
        ]
    )

    return (
        registration_indices,
        test_indices,
    )


# ============================================================
# One-user / one-split evaluation
# ============================================================

def evaluate_user_on_split(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
    registration_indices: np.ndarray,
    genuine_test_indices: np.ndarray,
    k_value: float = DEFAULT_K_VALUE,
) -> dict[str, object]:
    """
    Evaluate one registered user
    on one random split.
    """

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    registration_features = (
        sequence_features[
            registration_indices
        ]
    )

    template_data = build_sequence_template(
        registration_features,
        k_value=k_value,
    )

    threshold = float(
        template_data["threshold"]
    )

    # --------------------------------------------------------
    # Genuine authentication
    #
    # ONLY the 30 samples that were NOT
    # used during registration.
    # --------------------------------------------------------

    genuine_distances: list[float] = []

    genuine_accept = 0
    genuine_reject = 0

    for index in genuine_test_indices:

        sample_features = (
            sequence_features[
                index
            ]
        )

        distance, accepted = (
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

        if accepted:
            genuine_accept += 1
        else:
            genuine_reject += 1

    # --------------------------------------------------------
    # Impostor authentication
    #
    # All samples belonging to other users.
    # They never participate in registration.
    # --------------------------------------------------------

    impostor_mask = (
        np.asarray(raw_y)
        != user_id
    )

    impostor_indices = np.flatnonzero(
        impostor_mask
    )

    # Remove nothing here:
    # impostor users are completely different
    # from the registered user.
    impostor_distances: list[float] = []

    impostor_accept = 0
    impostor_reject = 0

    for index in impostor_indices:

        sample_features = (
            sequence_features[
                index
            ]
        )

        distance, accepted = (
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

        if accepted:
            impostor_accept += 1
        else:
            impostor_reject += 1

    genuine_count = len(
        genuine_distances
    )

    impostor_count = len(
        impostor_distances
    )

    genuine_array = np.asarray(
        genuine_distances,
        dtype=np.float32,
    )

    impostor_array = np.asarray(
        impostor_distances,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    frr = (
        genuine_reject
        / genuine_count
        if genuine_count
        else 0.0
    )

    far = (
        impostor_accept
        / impostor_count
        if impostor_count
        else 0.0
    )

    genuine_acceptance_rate = (
        genuine_accept
        / genuine_count
        if genuine_count
        else 0.0
    )

    return {
        "user": user_id,

        "registration_count":
            REGISTRATION_COUNT,

        "genuine_test_count":
            genuine_count,

        "impostor_test_count":
            impostor_count,

        "threshold":
            threshold,

        "genuine_accept":
            genuine_accept,

        "genuine_reject":
            genuine_reject,

        "impostor_accept":
            impostor_accept,

        "impostor_reject":
            impostor_reject,

        "far":
            far,

        "frr":
            frr,

        "genuine_acceptance_rate":
            genuine_acceptance_rate,

        "genuine_mean_distance":
            float(
                np.mean(genuine_array)
            ),

        "genuine_std_distance":
            float(
                np.std(genuine_array)
            ),

        "impostor_mean_distance":
            float(
                np.mean(impostor_array)
            ),

        "impostor_std_distance":
            float(
                np.std(impostor_array)
            ),
    }


# ============================================================
# Formal multi-split experiment
# ============================================================

def run_formal_validation(
    sequence_features: np.ndarray,
    raw_y: np.ndarray,
    users: Iterable[str] | None = None,
    num_splits: int = NUM_SPLITS,
    random_seed: int = RANDOM_SEED,
    k_value: float = DEFAULT_K_VALUE,
) -> list[dict[str, object]]:
    """
    Run the formal door-access experiment.

    For every split:

        20 samples -> registration
        30 samples -> genuine authentication

    Impostors:
        all samples from other users.

    Each split independently rebuilds:
        - min/max
        - template
        - threshold
    """

    if users is None:
        users = DEFAULT_USERS

    users = tuple(users)

    rng = np.random.default_rng(
        random_seed
    )

    all_results: list[dict[str, object]] = []

    for split_index in range(
        num_splits
    ):

        print(
            f"\nRunning split "
            f"{split_index + 1}/"
            f"{num_splits}..."
        )

        # ----------------------------------------------------
        # Generate registration/test split
        # for every user
        # ----------------------------------------------------

        split_indices: dict[
            str,
            tuple[np.ndarray, np.ndarray],
        ] = {}

        for user_id in users:

            registration_indices, test_indices = (
                create_random_split(
                    raw_y,
                    user_id,
                    rng,
                )
            )

            split_indices[
                user_id
            ] = (
                registration_indices,
                test_indices,
            )

        # ----------------------------------------------------
        # Evaluate every registered user
        # ----------------------------------------------------

        for user_id in users:

            (
                registration_indices,
                genuine_test_indices,
            ) = split_indices[
                user_id
            ]

            result = evaluate_user_on_split(
                sequence_features,
                raw_y,
                user_id,
                registration_indices,
                genuine_test_indices,
                k_value=k_value,
            )

            result[
                "split"
            ] = split_index + 1

            all_results.append(
                result
            )

    return all_results


# ============================================================
# Printing
# ============================================================

def print_protocol_information() -> None:

    print()
    print(
        "============================================================"
    )
    print(
        " Formal Door Access Validation Protocol"
    )
    print(
        "============================================================"
    )

    print(
        f"Registration samples : "
        f"{REGISTRATION_COUNT}"
    )

    print(
        f"Genuine test samples : "
        f"{TEST_COUNT}"
    )

    print(
        f"Random splits        : "
        f"{NUM_SPLITS}"
    )

    print(
        f"Random seed          : "
        f"{RANDOM_SEED}"
    )

    print(
        f"k value              : "
        f"{DEFAULT_K_VALUE}"
    )

    print(
        "Threshold             : "
        "mean + k * std"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Registration min/max are calculated"
    )
    print(
        "  ONLY from the 20 registration samples."
    )

    print(
        "  Genuine test samples are completely"
    )
    print(
        "  unseen during registration."
    )

    print(
        "============================================================"
    )


def print_split_results(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "========== Per-Split Results =========="
    )

    print(
        f"{'Split':<8}"
        f"{'User':<14}"
        f"{'Threshold':<12}"
        f"{'Genuine Accept':<16}"
        f"{'Genuine Reject':<16}"
        f"{'FAR':<10}"
        f"{'FRR':<10}"
    )

    print("-" * 100)

    for row in results:

        print(
            f"{int(row['split']):<8}"
            f"{str(row['user']):<14}"
            f"{float(row['threshold']):<12.4f}"
            f"{int(row['genuine_accept']):<16}"
            f"{int(row['genuine_reject']):<16}"
            f"{float(row['far']) * 100:<10.2f}%"
            f"{float(row['frr']) * 100:<10.2f}%"
        )


def print_summary(
    results: list[dict[str, object]],
    users: Iterable[str],
) -> None:

    users = tuple(users)

    print()
    print(
        "============================================================"
    )
    print(
        " Formal Door Access Validation Summary"
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
        f"{'Threshold':<14}"
    )

    print("-" * 100)

    for user_id in users:

        user_rows = [
            row
            for row in results
            if row["user"] == user_id
        ]

        gar_values = np.asarray(
            [
                float(
                    row[
                        "genuine_acceptance_rate"
                    ]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        frr_values = np.asarray(
            [
                float(
                    row["frr"]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        far_values = np.asarray(
            [
                float(
                    row["far"]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        threshold_values = np.asarray(
            [
                float(
                    row["threshold"]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        print(
            f"{user_id:<14}"
            f"{np.mean(gar_values) * 100:<12.2f}"
            f"{np.std(gar_values) * 100:<12.2f}"
            f"{np.mean(frr_values) * 100:<12.2f}"
            f"{np.std(frr_values) * 100:<12.2f}"
            f"{np.mean(far_values) * 100:<12.2f}"
            f"{np.std(far_values) * 100:<12.2f}"
            f"{np.mean(threshold_values):<14.4f}"
        )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    all_gar = np.asarray(
        [
            float(
                row[
                    "genuine_acceptance_rate"
                ]
            )
            for row in results
        ],
        dtype=np.float32,
    )

    all_frr = np.asarray(
        [
            float(row["frr"])
            for row in results
        ],
        dtype=np.float32,
    )

    all_far = np.asarray(
        [
            float(row["far"])
            for row in results
        ],
        dtype=np.float32,
    )

    print()
    print(
        "========== Overall Mean =========="
    )

    print(
        f"Average Genuine Acceptance Rate : "
        f"{np.mean(all_gar) * 100:.2f}%"
    )

    print(
        f"Average FRR                     : "
        f"{np.mean(all_frr) * 100:.2f}%"
    )

    print(
        f"Average FAR                     : "
        f"{np.mean(all_far) * 100:.2f}%"
    )

    print()
    print(
        "========== Overall Std =========="
    )

    print(
        f"Genuine Acceptance Rate Std : "
        f"{np.std(all_gar) * 100:.2f}%"
    )

    print(
        f"FRR Std                    : "
        f"{np.std(all_frr) * 100:.2f}%"
    )

    print(
        f"FAR Std                    : "
        f"{np.std(all_far) * 100:.2f}%"
    )


def print_distance_summary(
    results: list[dict[str, object]],
    users: Iterable[str],
) -> None:

    print()
    print(
        "========== Distance Summary =========="
    )

    print(
        f"{'User':<14}"
        f"{'Genuine Mean':<16}"
        f"{'Genuine Std':<16}"
        f"{'Impostor Mean':<16}"
        f"{'Impostor Std':<16}"
    )

    print("-" * 85)

    for user_id in users:

        user_rows = [
            row
            for row in results
            if row["user"] == user_id
        ]

        genuine_means = np.asarray(
            [
                float(
                    row[
                        "genuine_mean_distance"
                    ]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        genuine_stds = np.asarray(
            [
                float(
                    row[
                        "genuine_std_distance"
                    ]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        impostor_means = np.asarray(
            [
                float(
                    row[
                        "impostor_mean_distance"
                    ]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        impostor_stds = np.asarray(
            [
                float(
                    row[
                        "impostor_std_distance"
                    ]
                )
                for row in user_rows
            ],
            dtype=np.float32,
        )

        print(
            f"{user_id:<14}"
            f"{np.mean(genuine_means):<16.4f}"
            f"{np.mean(genuine_stds):<16.4f}"
            f"{np.mean(impostor_means):<16.4f}"
            f"{np.mean(impostor_stds):<16.4f}"
        )


# ============================================================
# Dataset information
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
        f"Original X shape: "
        f"{X.shape}"
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


def print_feature_information() -> None:

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


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "============================================================"
    )
    print(
        " Sequence Statistics - Formal Door Access Validation"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, raw_y = load_data()

    print_dataset_information(
        X,
        raw_y,
    )

    # --------------------------------------------------------
    # Extract sequence-level features
    #
    # This part is exactly the same idea as your
    # previous Sequence Statistics experiment.
    # --------------------------------------------------------

    print()
    print(
        "Extracting sequence-level statistics..."
    )

    sequence_features = (
        extract_dataset_sequence_statistics(
            X
        )
    )

    print(
        f"Sequence feature shape: "
        f"{sequence_features.shape}"
    )

    if np.isnan(
        sequence_features
    ).any():

        raise ValueError(
            "NaN detected in sequence features"
        )

    print_feature_information()

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    print_protocol_information()

    # --------------------------------------------------------
    # Formal multi-split validation
    # --------------------------------------------------------

    results = run_formal_validation(
        sequence_features,
        raw_y,
        users=DEFAULT_USERS,
        num_splits=NUM_SPLITS,
        random_seed=RANDOM_SEED,
        k_value=DEFAULT_K_VALUE,
    )

    # --------------------------------------------------------
    # Print detailed split results
    # --------------------------------------------------------

    print_split_results(
        results
    )

    # --------------------------------------------------------
    # Print final user summary
    # --------------------------------------------------------

    print_summary(
        results,
        DEFAULT_USERS,
    )

    # --------------------------------------------------------
    # Distance summary
    # --------------------------------------------------------

    print_distance_summary(
        results,
        DEFAULT_USERS,
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        " Formal Validation Finished"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()