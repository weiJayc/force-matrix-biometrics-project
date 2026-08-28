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

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "dataset"
)


# ============================================================
# Imports
# ============================================================

from data_loader import load_dataset

from contact_gate import (
    calculate_contact_score,
    calibrate_contact_gate,
)


# ============================================================
# Configuration
# ============================================================

USERS = (
    "amber",
    "jay",
    "666",
)

NO_CONTACT_USER = "background"

REGISTRATION_COUNTS = (
    5,
    10,
    20,
)

TEST_COUNT = 30

REPEATS = 20

RANDOM_SEED = 42


# ------------------------------------------------------------
# Contact Gate calibration
#
# Background:
#   20 -> calibration
#   30 -> formal no-contact test
# ------------------------------------------------------------

BACKGROUND_CALIBRATION_COUNT = 20


# ------------------------------------------------------------
# Sequence Statistics threshold
#
# threshold =
#     mean(registration distance)
#     +
#     K * std(registration distance)
# ------------------------------------------------------------

SEQUENCE_THRESHOLD_K = 2.0


# ------------------------------------------------------------
# Contact Area threshold
# ------------------------------------------------------------

CONTACT_AREA_THRESHOLD = 1000.0


# ============================================================
# Dataset
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load original dataset.

    Expected:
        X.shape == (samples, 50, 16)
    """

    X, y = load_dataset(DATA_DIR)

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(y)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape "
            f"(samples, frames, sensors), "
            f"got {X.shape}"
        )

    if X.shape[1:] != (50, 16):
        raise ValueError(
            f"Expected each sample to be "
            f"(50, 16), got {X.shape[1:]}"
        )

    return X, y


# ============================================================
# Dataset Information
# ============================================================

def print_dataset_information(
    X: np.ndarray,
    labels: np.ndarray,
) -> None:

    print()
    print("=" * 110)
    print("DATASET INFORMATION")
    print("=" * 110)

    print(
        f"Dataset             : {DATA_DIR}"
    )

    print(
        f"X shape             : {X.shape}"
    )

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    print()

    for label, count in zip(
        unique_labels,
        counts,
    ):
        print(
            f"{str(label):<15}"
            f": {int(count)} samples"
        )


# ============================================================
# Sequence Statistics
# ============================================================

def extract_sequence_statistics(
    sample: np.ndarray,
) -> np.ndarray:
    """
    Convert one raw pressure sequence:

        (50, 16)

    into one sequence-level feature vector.

    Frame-level features:

        1. pressure_sum
        2. max_pressure
        3. contact_area
        4. cop_x
        5. cop_y
        6. left_right_ratio
        7. top_bottom_ratio

    For each feature:

        mean
        std
        min
        max
        mean absolute temporal difference
        std temporal difference
        max absolute temporal difference
        start
        end

    Additional global features:

        argmax pressure_sum
        argmax max_pressure
        total pressure sum
        std pressure_sum
    """

    sample = np.asarray(
        sample,
        dtype=np.float32,
    )

    if sample.ndim != 2:
        raise ValueError(
            f"Expected sample shape "
            f"(frames, sensors), "
            f"got {sample.shape}"
        )

    if sample.shape[1] != 16:
        raise ValueError(
            f"Expected 16 sensors, "
            f"got {sample.shape[1]}"
        )

    if sample.shape[0] != 50:
        raise ValueError(
            f"Expected 50 frames, "
            f"got {sample.shape[0]}"
        )

    # ========================================================
    # Basic frame-level pressure
    # ========================================================

    pressure_sum = np.sum(
        sample,
        axis=1,
    )

    max_pressure = np.max(
        sample,
        axis=1,
    )

    contact_area = np.sum(
        sample > CONTACT_AREA_THRESHOLD,
        axis=1,
    )

    # ========================================================
    # 4 x 4 pressure matrix
    # ========================================================

    matrix = sample.reshape(
        sample.shape[0],
        4,
        4,
    )

    x_positions = np.array(
        [0, 1, 2, 3],
        dtype=np.float32,
    )

    y_positions = np.array(
        [0, 1, 2, 3],
        dtype=np.float32,
    )

    x_coords = np.tile(
        x_positions,
        (4, 1),
    )

    y_coords = np.tile(
        y_positions.reshape(-1, 1),
        (1, 4),
    )

    pressure_sum_safe = np.where(
        pressure_sum > 0,
        pressure_sum,
        1.0,
    )

    # ========================================================
    # COP
    # ========================================================

    cop_x = (
        np.sum(
            matrix * x_coords,
            axis=(1, 2),
        )
        / pressure_sum_safe
    )

    cop_y = (
        np.sum(
            matrix * y_coords,
            axis=(1, 2),
        )
        / pressure_sum_safe
    )

    # ========================================================
    # Left / Right
    # ========================================================

    left_pressure = np.sum(
        matrix[:, :, :2],
        axis=(1, 2),
    )

    right_pressure = np.sum(
        matrix[:, :, 2:],
        axis=(1, 2),
    )

    left_right_ratio = (
        left_pressure
        /
        np.where(
            right_pressure > 0,
            right_pressure,
            1.0,
        )
    )

    # ========================================================
    # Top / Bottom
    # ========================================================

    top_pressure = np.sum(
        matrix[:, :2, :],
        axis=(1, 2),
    )

    bottom_pressure = np.sum(
        matrix[:, 2:, :],
        axis=(1, 2),
    )

    top_bottom_ratio = (
        top_pressure
        /
        np.where(
            bottom_pressure > 0,
            bottom_pressure,
            1.0,
        )
    )

    # ========================================================
    # Sequence-level statistics
    # ========================================================

    features: list[float] = []

    frame_features = {
        "pressure_sum": pressure_sum,
        "max_pressure": max_pressure,
        "contact_area": contact_area,
        "cop_x": cop_x,
        "cop_y": cop_y,
        "left_right_ratio": left_right_ratio,
        "top_bottom_ratio": top_bottom_ratio,
    }

    for _, values in frame_features.items():

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Level
        # ----------------------------------------------------

        features.append(
            float(np.mean(values))
        )

        features.append(
            float(np.std(values))
        )

        features.append(
            float(np.min(values))
        )

        features.append(
            float(np.max(values))
        )

        # ----------------------------------------------------
        # Temporal change
        # ----------------------------------------------------

        if len(values) > 1:

            diff = np.diff(values)

            features.append(
                float(
                    np.mean(
                        np.abs(diff)
                    )
                )
            )

            features.append(
                float(
                    np.std(diff)
                )
            )

            features.append(
                float(
                    np.max(
                        np.abs(diff)
                    )
                )
            )

        else:

            features.extend(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            )

        # ----------------------------------------------------
        # Start / End
        # ----------------------------------------------------

        features.append(
            float(values[0])
        )

        features.append(
            float(values[-1])
        )

    # ========================================================
    # Whole-sequence global features
    # ========================================================

    features.append(
        float(
            np.argmax(
                pressure_sum
            )
        )
    )

    features.append(
        float(
            np.argmax(
                max_pressure
            )
        )
    )

    features.append(
        float(
            np.sum(
                pressure_sum
            )
        )
    )

    features.append(
        float(
            np.std(
                pressure_sum
            )
        )
    )

    result = np.asarray(
        features,
        dtype=np.float32,
    )

    if np.isnan(result).any():
        raise ValueError(
            "NaN detected in sequence statistics."
        )

    return result


# ============================================================
# Dataset Sequence Statistics
# ============================================================

def extract_dataset_statistics(
    X: np.ndarray,
) -> np.ndarray:
    """
    Convert:

        (samples, 50, 16)

    into:

        (samples, feature_dimension)
    """

    return np.asarray(
        [
            extract_sequence_statistics(
                sample
            )
            for sample in X
        ],
        dtype=np.float32,
    )


# ============================================================
# Normalization
# ============================================================

def fit_normalization(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Min-Max normalization using
    REGISTRATION DATA ONLY.
    """

    minimum = np.min(
        X,
        axis=0,
    )

    maximum = np.max(
        X,
        axis=0,
    )

    return (
        minimum,
        maximum,
    )


def normalize(
    X: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    """
    Apply previously fitted Min-Max normalization.
    """

    denominator = (
        maximum - minimum
    )

    denominator = np.where(
        denominator == 0,
        1.0,
        denominator,
    )

    return (
        (X - minimum)
        / denominator
    ).astype(np.float32)


# ============================================================
# Sequence Statistics Template
# ============================================================

def build_sequence_template(
    registration_features: np.ndarray,
) -> dict:
    """
    Build Sequence Statistics identity template.

    Registration:
        raw sequence statistics
        ->
        Min-Max normalization
        ->
        mean template

    Registration distance:
        Euclidean distance between
        each registration vector
        and the template.

    Threshold:

        mean(distance)
        +
        K * std(distance)
    """

    registration_features = np.asarray(
        registration_features,
        dtype=np.float32,
    )

    if registration_features.ndim != 2:
        raise ValueError(
            "registration_features must be 2D."
        )

    # ========================================================
    # Fit normalization ONLY on registration
    # ========================================================

    minimum, maximum = (
        fit_normalization(
            registration_features
        )
    )

    normalized_registration = (
        normalize(
            registration_features,
            minimum,
            maximum,
        )
    )

    # ========================================================
    # Template
    # ========================================================

    template = np.mean(
        normalized_registration,
        axis=0,
    )

    # ========================================================
    # Registration distances
    # ========================================================

    registration_distances = np.linalg.norm(
        normalized_registration - template,
        axis=1,
    )

    # ========================================================
    # Calibrated identity threshold
    # ========================================================

    threshold = (
        float(
            np.mean(
                registration_distances
            )
        )
        +
        SEQUENCE_THRESHOLD_K
        *
        float(
            np.std(
                registration_distances
            )
        )
    )

    return {
        "template": template,
        "minimum": minimum,
        "maximum": maximum,
        "threshold": threshold,
        "registration_distances":
            registration_distances,
    }


# ============================================================
# Sequence Statistics Authentication
# ============================================================

def calculate_sequence_distance(
    template: dict,
    test_features: np.ndarray,
) -> np.ndarray:
    """
    Calculate Euclidean distance from
    Sequence Statistics template.
    """

    normalized = normalize(
        test_features,
        template["minimum"],
        template["maximum"],
    )

    return np.linalg.norm(
        normalized - template["template"],
        axis=1,
    )


def authenticate_sequence_statistics(
    template: dict,
    sample_features: np.ndarray,
) -> tuple[float, bool]:
    """
    Authenticate one sequence.

    Accept when:

        distance <= threshold
    """

    distance = calculate_sequence_distance(
        template,
        sample_features.reshape(1, -1),
    )[0]

    accepted = (
        distance
        <=
        template["threshold"]
    )

    return (
        float(distance),
        bool(accepted),
    )


# ============================================================
# Split User Data
# ============================================================

def create_user_split(
    user_indices: np.ndarray,
    registration_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split one user's data into:

        registration_count
        +
        30 test samples

    No overlap.
    """

    required = (
        registration_count
        + TEST_COUNT
    )

    if len(user_indices) < required:
        raise ValueError(
            f"Need {required} samples, "
            f"but only {len(user_indices)} available."
        )

    shuffled = rng.permutation(
        user_indices
    )

    registration_indices = (
        shuffled[
            :registration_count
        ]
    )

    test_indices = (
        shuffled[
            registration_count:
            registration_count
            + TEST_COUNT
        ]
    )

    return (
        registration_indices,
        test_indices,
    )


# ============================================================
# Split Background
# ============================================================

def create_background_split(
    background_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Background:

        20 -> Contact Gate calibration

        30 -> unseen no-contact test
    """

    required = (
        BACKGROUND_CALIBRATION_COUNT
        + TEST_COUNT
    )

    if len(background_indices) < required:
        raise ValueError(
            f"Background needs "
            f"{required} samples, "
            f"but only "
            f"{len(background_indices)} available."
        )

    shuffled = rng.permutation(
        background_indices
    )

    calibration_indices = (
        shuffled[
            :BACKGROUND_CALIBRATION_COUNT
        ]
    )

    test_indices = (
        shuffled[
            BACKGROUND_CALIBRATION_COUNT:
            BACKGROUND_CALIBRATION_COUNT
            + TEST_COUNT
        ]
    )

    return (
        calibration_indices,
        test_indices,
    )


# ============================================================
# Contact Gate
# ============================================================

def calibrate_gate(
    template_user: str,
    registration_count: int,
    repeat: int,
    registration_samples: np.ndarray,
    background_calibration: np.ndarray,
):
    """
    Contact Gate calibration.

    Genuine:
        registration samples

    No-contact:
        background calibration samples

    Formal test samples are NOT used.
    """

    return calibrate_contact_gate(
        user_id=template_user,
        registration_count=registration_count,
        repeat=repeat,
        registration_samples=registration_samples,
        background_calibration_samples=(
            background_calibration
        ),
    )


def contact_decision(
    gate,
    sample: np.ndarray,
) -> tuple[float, bool]:

    score = float(
        calculate_contact_score(
            sample
        )
    )

    contact = (
        score
        >=
        float(gate.threshold)
    )

    return (
        score,
        contact,
    )


# ============================================================
# Contact Gate Evaluation
# ============================================================

def evaluate_contact_gate(
    gate,
    genuine_test: np.ndarray,
    no_contact_test: np.ndarray,
) -> dict:
    """
    Evaluate Contact Gate on unseen data.
    """

    genuine_contact_count = 0
    no_contact_contact_count = 0

    genuine_scores = []
    no_contact_scores = []

    # ========================================================
    # Genuine
    # ========================================================

    for sample in genuine_test:

        score, contact = (
            contact_decision(
                gate,
                sample,
            )
        )

        genuine_scores.append(
            score
        )

        if contact:
            genuine_contact_count += 1

    # ========================================================
    # No-contact
    # ========================================================

    for sample in no_contact_test:

        score, contact = (
            contact_decision(
                gate,
                sample,
            )
        )

        no_contact_scores.append(
            score
        )

        if contact:
            no_contact_contact_count += 1

    genuine_total = len(
        genuine_test
    )

    no_contact_total = len(
        no_contact_test
    )

    contact_rate = (
        genuine_contact_count
        /
        genuine_total
        if genuine_total > 0
        else 0.0
    )

    miss_rate = (
        1.0
        -
        contact_rate
    )

    no_contact_far = (
        no_contact_contact_count
        /
        no_contact_total
        if no_contact_total > 0
        else 0.0
    )

    no_contact_reject = (
        1.0
        -
        no_contact_far
    )

    return {
        "contact_rate":
            contact_rate,

        "miss_rate":
            miss_rate,

        "no_contact_far":
            no_contact_far,

        "no_contact_reject":
            no_contact_reject,

        "balanced_accuracy":
            (
                contact_rate
                +
                no_contact_reject
            )
            / 2.0,

        "genuine_scores":
            np.asarray(
                genuine_scores,
                dtype=np.float64,
            ),

        "no_contact_scores":
            np.asarray(
                no_contact_scores,
                dtype=np.float64,
            ),
    }


# ============================================================
# One E2E Experiment
# ============================================================

def run_one_experiment(
    X: np.ndarray,
    X_features: np.ndarray,
    labels: np.ndarray,
    template_user: str,
    registration_count: int,
    repeat: int,
    registration_indices_by_user: dict,
    test_indices_by_user: dict,
    background_calibration_indices: np.ndarray,
    no_contact_test_indices: np.ndarray,
) -> dict:
    """
    Complete two-stage pipeline:

        Raw 50x16
            ↓
        Contact Gate
            ↓
        Sequence Statistics
            ↓
        Euclidean Identity Authentication
    """

    # ========================================================
    # Data
    # ========================================================

    registration_indices = (
        registration_indices_by_user[
            template_user
        ]
    )

    genuine_test_indices = (
        test_indices_by_user[
            template_user
        ]
    )

    registration_samples = (
        X[
            registration_indices
        ]
    )

    genuine_test = (
        X[
            genuine_test_indices
        ]
    )

    background_calibration = (
        X[
            background_calibration_indices
        ]
    )

    no_contact_test = (
        X[
            no_contact_test_indices
        ]
    )

    # ========================================================
    # 1. Contact Gate Calibration
    # ========================================================

    gate = calibrate_gate(
        template_user=template_user,
        registration_count=registration_count,
        repeat=repeat,
        registration_samples=registration_samples,
        background_calibration=(
            background_calibration
        ),
    )

    # ========================================================
    # 2. Contact Gate Formal Evaluation
    # ========================================================

    gate_result = (
        evaluate_contact_gate(
            gate=gate,
            genuine_test=genuine_test,
            no_contact_test=no_contact_test,
        )
    )

    # ========================================================
    # 3. Sequence Statistics Identity Template
    # ========================================================

    registration_features = (
        X_features[
            registration_indices
        ]
    )

    sequence_template = (
        build_sequence_template(
            registration_features
        )
    )

    # ========================================================
    # 4. Genuine
    #
    # Contact Gate
    #       ↓
    # Sequence Statistics
    # ========================================================

    genuine_contact_count = 0
    genuine_accept_count = 0

    genuine_distances = []

    for index in genuine_test_indices:

        sample = X[index]

        _, contact = (
            contact_decision(
                gate,
                sample,
            )
        )

        if not contact:
            continue

        genuine_contact_count += 1

        sample_features = (
            X_features[index]
        )

        distance, accepted = (
            authenticate_sequence_statistics(
                sequence_template,
                sample_features,
            )
        )

        genuine_distances.append(
            distance
        )

        if accepted:
            genuine_accept_count += 1

    # ========================================================
    # 5. Impostor
    #
    # Contact Gate FIRST
    #
    # FAR denominator:
    #
    #     2 users × 30 samples
    #
    # Gate rejected samples remain
    # inside denominator.
    # ========================================================

    impostor_total = 0
    impostor_contact_count = 0
    impostor_accept_count = 0

    impostor_distances = []

    impostor_details = {}

    for impostor_user in USERS:

        if impostor_user == template_user:
            continue

        impostor_indices = (
            test_indices_by_user[
                impostor_user
            ]
        )

        accepted_count = 0
        contact_count = 0

        distances = []

        for index in impostor_indices:

            impostor_total += 1

            sample = X[index]

            _, contact = (
                contact_decision(
                    gate,
                    sample,
                )
            )

            # --------------------------------------------
            # Contact Gate rejects
            #
            # Correct rejection.
            # Do NOT run identity authentication.
            # --------------------------------------------

            if not contact:
                continue

            contact_count += 1
            impostor_contact_count += 1

            sample_features = (
                X_features[index]
            )

            distance, accepted = (
                authenticate_sequence_statistics(
                    sequence_template,
                    sample_features,
                )
            )

            distances.append(
                distance
            )

            if accepted:
                accepted_count += 1
                impostor_accept_count += 1

        impostor_details[
            impostor_user
        ] = {
            "accept":
                accepted_count,

            "reject":
                TEST_COUNT
                - accepted_count,

            "contact":
                contact_count,

            "contact_rate":
                (
                    contact_count
                    /
                    TEST_COUNT
                ),

            "average_distance":
                (
                    float(
                        np.mean(
                            distances
                        )
                    )
                    if distances
                    else np.nan
                ),
        }

        impostor_distances.extend(
            distances
        )

    # ========================================================
    # 6. Final E2E Identity Metrics
    # ========================================================

    gar = (
        genuine_accept_count
        /
        TEST_COUNT
    )

    frr = (
        1.0
        -
        gar
    )

    far = (
        impostor_accept_count
        /
        impostor_total
        if impostor_total > 0
        else 0.0
    )

    return {
        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        "template_user":
            template_user,

        "registration_count":
            registration_count,

        "repeat":
            repeat,

        # ----------------------------------------------------
        # Contact Gate
        # ----------------------------------------------------

        "contact_threshold":
            float(
                gate.threshold
            ),

        "contact_rate":
            gate_result[
                "contact_rate"
            ],

        "contact_miss_rate":
            gate_result[
                "miss_rate"
            ],

        "no_contact_far":
            gate_result[
                "no_contact_far"
            ],

        "no_contact_reject":
            gate_result[
                "no_contact_reject"
            ],

        # ----------------------------------------------------
        # Sequence Statistics
        # ----------------------------------------------------

        "sequence_threshold":
            float(
                sequence_template[
                    "threshold"
                ]
            ),

        "gar":
            gar,

        "frr":
            frr,

        "far":
            far,

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        "genuine_contact_count":
            genuine_contact_count,

        "genuine_accept":
            genuine_accept_count,

        "genuine_reject":
            TEST_COUNT
            -
            genuine_accept_count,

        "genuine_distance":
            (
                float(
                    np.mean(
                        genuine_distances
                    )
                )
                if genuine_distances
                else np.nan
            ),

        "impostor_contact_count":
            impostor_contact_count,

        "impostor_accept":
            impostor_accept_count,

        "impostor_total":
            impostor_total,

        "impostor_distance":
            (
                float(
                    np.mean(
                        impostor_distances
                    )
                )
                if impostor_distances
                else np.nan
            ),

        # ----------------------------------------------------
        # Details
        # ----------------------------------------------------

        "impostor_details":
            impostor_details,
    }


# ============================================================
# Prepare One Repeat
# ============================================================

def prepare_repeat_split(
    X: np.ndarray,
    labels: np.ndarray,
    registration_count: int,
    rng: np.random.Generator,
):
    """
    IMPORTANT:

    One repeat creates one fixed split.

    The SAME split is then used by:

        Contact Gate standalone
        Sequence Statistics standalone
        E2E

    This makes comparison fair.
    """

    user_indices = {
        user:
            np.where(
                labels == user
            )[0]
        for user in USERS
    }

    background_indices = np.where(
        labels == NO_CONTACT_USER
    )[0]

    registration_indices_by_user = {}
    test_indices_by_user = {}

    # ========================================================
    # User splits
    # ========================================================

    for user in USERS:

        (
            registration_indices,
            test_indices,
        ) = create_user_split(
            user_indices[user],
            registration_count,
            rng,
        )

        registration_indices_by_user[
            user
        ] = registration_indices

        test_indices_by_user[
            user
        ] = test_indices

    # ========================================================
    # Background split
    # ========================================================

    (
        background_calibration_indices,
        no_contact_test_indices,
    ) = create_background_split(
        background_indices,
        rng,
    )

    return (
        registration_indices_by_user,
        test_indices_by_user,
        background_calibration_indices,
        no_contact_test_indices,
    )


# ============================================================
# Contact Gate Standalone Summary
# ============================================================

def print_contact_gate_summary(
    rows: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "CONTACT GATE STANDALONE SUMMARY"
    )
    print("=" * 110)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Contact Rate':>17}"
        f"{'Miss Rate':>16}"
        f"{'No-contact FAR':>20}"
        f"{'No-contact Reject':>22}"
        f"{'Threshold':>18}"
    )

    print("-" * 110)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            selected = [
                row
                for row in rows
                if row[
                    "registration_count"
                ]
                == registration_count
                and row[
                    "template_user"
                ]
                == user
            ]

            if not selected:
                continue

            contact_rates = np.asarray(
                [
                    row["contact_rate"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            miss_rates = np.asarray(
                [
                    row["contact_miss_rate"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            no_contact_fars = np.asarray(
                [
                    row["no_contact_far"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            no_contact_rejects = (
                1.0
                -
                no_contact_fars
            )

            thresholds = np.asarray(
                [
                    row["contact_threshold"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{np.mean(contact_rates) * 100:>14.2f}%"
                f"{np.mean(miss_rates) * 100:>13.2f}%"
                f"{np.mean(no_contact_fars) * 100:>17.2f}%"
                f"{np.mean(no_contact_rejects) * 100:>19.2f}%"
                f"{np.mean(thresholds):>18.4f}"
            )


# ============================================================
# Sequence Statistics Standalone Summary
# ============================================================

def print_sequence_statistics_summary(
    rows: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "SEQUENCE STATISTICS STANDALONE SUMMARY"
    )
    print("=" * 110)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'GAR Mean':>14}"
        f"{'GAR Std':>14}"
        f"{'FRR Mean':>14}"
        f"{'FAR Mean':>14}"
        f"{'FAR Std':>14}"
        f"{'Threshold':>18}"
    )

    print("-" * 110)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            selected = [
                row
                for row in rows
                if row[
                    "registration_count"
                ]
                == registration_count
                and row[
                    "template_user"
                ]
                == user
            ]

            if not selected:
                continue

            gar_values = np.asarray(
                [
                    row["standalone_gar"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            frr_values = (
                1.0
                -
                gar_values
            )

            far_values = np.asarray(
                [
                    row["standalone_far"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            thresholds = np.asarray(
                [
                    row["sequence_threshold"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{np.mean(gar_values) * 100:>10.2f}%"
                f"{np.std(gar_values) * 100:>10.2f}%"
                f"{np.mean(frr_values) * 100:>10.2f}%"
                f"{np.mean(far_values) * 100:>10.2f}%"
                f"{np.std(far_values) * 100:>10.2f}%"
                f"{np.mean(thresholds):>18.4f}"
            )


# ============================================================
# Complete E2E Summary
# ============================================================

def print_e2e_summary(
    rows: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "COMPLETE TWO-STAGE END-TO-END BENCHMARK"
    )
    print("=" * 110)

    print(
        "Pipeline:"
    )

    print(
        "    Raw Pressure 50 x 16"
    )

    print(
        "          -> Contact Gate"
    )

    print(
        "          -> Sequence Statistics"
    )

    print(
        "          -> Euclidean Identity Authentication"
    )

    print()

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'GAR Mean':>14}"
        f"{'GAR Std':>14}"
        f"{'FRR Mean':>14}"
        f"{'FAR Mean':>14}"
        f"{'FAR Std':>14}"
        f"{'No-contact FAR':>18}"
        f"{'Gate Contact':>17}"
        f"{'Gate Threshold':>18}"
        f"{'SS Threshold':>18}"
    )

    print("-" * 155)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            selected = [
                row
                for row in rows
                if row[
                    "registration_count"
                ]
                == registration_count
                and row[
                    "template_user"
                ]
                == user
            ]

            if not selected:
                continue

            gar_values = np.asarray(
                [
                    row["gar"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            frr_values = (
                1.0
                -
                gar_values
            )

            far_values = np.asarray(
                [
                    row["far"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            no_contact_fars = np.asarray(
                [
                    row["no_contact_far"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            contact_rates = np.asarray(
                [
                    row["contact_rate"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            gate_thresholds = np.asarray(
                [
                    row["contact_threshold"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            sequence_thresholds = np.asarray(
                [
                    row["sequence_threshold"]
                    for row in selected
                ],
                dtype=np.float64,
            )

            print(
                f"{registration_count:<6}"
                f"{user:<12}"
                f"{np.mean(gar_values) * 100:>10.2f}%"
                f"{np.std(gar_values) * 100:>10.2f}%"
                f"{np.mean(frr_values) * 100:>10.2f}%"
                f"{np.mean(far_values) * 100:>10.2f}%"
                f"{np.std(far_values) * 100:>10.2f}%"
                f"{np.mean(no_contact_fars) * 100:>15.2f}%"
                f"{np.mean(contact_rates) * 100:>14.2f}%"
                f"{np.mean(gate_thresholds):>18.4f}"
                f"{np.mean(sequence_thresholds):>18.4f}"
            )


# ============================================================
# Standalone vs E2E
# ============================================================

def print_standalone_vs_e2e(
    rows: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "STANDALONE vs END-TO-END COMPARISON"
    )
    print("=" * 110)

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'SS GAR':>13}"
        f"{'E2E GAR':>13}"
        f"{'GAR Change':>15}"
        f"{'SS FAR':>13}"
        f"{'E2E FAR':>13}"
        f"{'FAR Change':>15}"
        f"{'Gate Miss':>15}"
        f"{'NC FAR':>13}"
    )

    print("-" * 110)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            selected = [
                row
                for row in rows
                if row[
                    "registration_count"
                ]
                == registration_count
                and row[
                    "template_user"
                ]
                == user
            ]

            if not selected:
                continue

            standalone_gar = np.mean(
                [
                    row[
                        "standalone_gar"
                    ]
                    for row in selected
                ]
            )

            e2e_gar = np.mean(
                [
                    row["gar"]
                    for row in selected
                ]
            )

            standalone_far = np.mean(
                [
                    row[
                        "standalone_far"
                    ]
                    for row in selected
                ]
            )

            e2e_far = np.mean(
                [
                    row["far"]
                    for row in selected
                ]
            )

            gate_miss = np.mean(
                [
                    row[
                        "contact_miss_rate"
                    ]
                    for row in selected
                ]
            )

            no_contact_far = np.mean(
                [
                    row[
                        "no_contact_far"
                    ]
                    for row in selected
                ]
            )

            gar_change = (
                e2e_gar
                -
                standalone_gar
            )

            far_change = (
                e2e_far
                -
                standalone_far
            )

            print(
                f"{registration_count:<6}"
                f"{user:<12}"
                f"{standalone_gar * 100:>9.2f}%"
                f"{e2e_gar * 100:>10.2f}%"
                f"{gar_change * 100:>12.2f}%"
                f"{standalone_far * 100:>9.2f}%"
                f"{e2e_far * 100:>10.2f}%"
                f"{far_change * 100:>12.2f}%"
                f"{gate_miss * 100:>12.2f}%"
                f"{no_contact_far * 100:>10.2f}%"
            )


# ============================================================
# Contact Gate Threshold Debug
# ============================================================

def print_threshold_debug(
    rows: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "CALIBRATED THRESHOLDS"
    )
    print("=" * 110)

    print(
        f"{'Reg':<8}"
        f"{'Repeat':<10}"
        f"{'User':<12}"
        f"{'Gate Threshold':>20}"
        f"{'Sequence Threshold':>22}"
    )

    print("-" * 110)

    for row in rows:

        print(
            f"{row['registration_count']:<8}"
            f"{row['repeat']:<10}"
            f"{row['template_user']:<12}"
            f"{row['contact_threshold']:>20.4f}"
            f"{row['sequence_threshold']:>22.4f}"
        )


# ============================================================
# Main Benchmark
# ============================================================

def run_benchmark(
    X: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:

    # ========================================================
    # Extract Sequence Statistics ONCE
    #
    # Feature extraction itself does not use test labels.
    # ========================================================

    print()
    print(
        "Extracting Sequence Statistics..."
    )

    X_features = (
        extract_dataset_statistics(
            X
        )
    )

    print(
        f"Sequence feature shape : "
        f"{X_features.shape}"
    )

    if np.isnan(X_features).any():
        raise ValueError(
            "NaN detected in sequence feature dataset."
        )

    all_rows = []

    # ========================================================
    # Registration Count
    # ========================================================

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        print()
        print("=" * 110)
        print(
            f"REGISTRATION = "
            f"{registration_count}"
        )
        print("=" * 110)

        # ====================================================
        # Random generator
        #
        # Each registration count gets its own deterministic
        # sequence of 20 repeats.
        # ====================================================

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        # ====================================================
        # Repeat
        # ====================================================

        for repeat in range(
            1,
            REPEATS + 1,
        ):

            print()
            print(
                f"Running Reg="
                f"{registration_count}, "
                f"Repeat={repeat:02d}"
            )

            # =================================================
            # ONE split for this repeat
            #
            # Same data is used by:
            #
            #   Contact Gate
            #   Sequence Statistics
            #   E2E
            # =================================================

            (
                registration_indices_by_user,
                test_indices_by_user,
                background_calibration_indices,
                no_contact_test_indices,
            ) = prepare_repeat_split(
                X,
                labels,
                registration_count,
                rng,
            )

            # =================================================
            # Each user becomes registration identity
            # =================================================

            for template_user in USERS:

                print(
                    f"    Template="
                    f"{template_user}"
                )

                result = (
                    run_one_experiment(
                        X=X,
                        X_features=X_features,
                        labels=labels,
                        template_user=template_user,
                        registration_count=(
                            registration_count
                        ),
                        repeat=repeat,
                        registration_indices_by_user=(
                            registration_indices_by_user
                        ),
                        test_indices_by_user=(
                            test_indices_by_user
                        ),
                        background_calibration_indices=(
                            background_calibration_indices
                        ),
                        no_contact_test_indices=(
                            no_contact_test_indices
                        ),
                    )
                )

                # =================================================
                # Sequence Statistics STANDALONE
                #
                # Here we intentionally do NOT use Contact Gate.
                #
                # This answers:
                #
                # "如果所有 samples 都直接進入
                #  Sequence Statistics，
                #  身份辨識效果如何？"
                # =================================================

                registration_indices = (
                    registration_indices_by_user[
                        template_user
                    ]
                )

                registration_features = (
                    X_features[
                        registration_indices
                    ]
                )

                sequence_template = (
                    build_sequence_template(
                        registration_features
                    )
                )

                # -------------------------------------------------
                # Genuine standalone
                # -------------------------------------------------

                genuine_indices = (
                    test_indices_by_user[
                        template_user
                    ]
                )

                genuine_distances = (
                    calculate_sequence_distance(
                        sequence_template,
                        X_features[
                            genuine_indices
                        ],
                    )
                )

                genuine_accept = (
                    genuine_distances
                    <=
                    sequence_template[
                        "threshold"
                    ]
                )

                standalone_genuine_accept = (
                    int(
                        np.sum(
                            genuine_accept
                        )
                    )
                )

                standalone_gar = (
                    standalone_genuine_accept
                    /
                    TEST_COUNT
                )

                # -------------------------------------------------
                # Impostor standalone
                # -------------------------------------------------

                standalone_impostor_accept = 0
                standalone_impostor_total = 0

                for impostor_user in USERS:

                    if (
                        impostor_user
                        ==
                        template_user
                    ):
                        continue

                    impostor_indices = (
                        test_indices_by_user[
                            impostor_user
                        ]
                    )

                    distances = (
                        calculate_sequence_distance(
                            sequence_template,
                            X_features[
                                impostor_indices
                            ],
                        )
                    )

                    accepted = (
                        distances
                        <=
                        sequence_template[
                            "threshold"
                        ]
                    )

                    standalone_impostor_accept += (
                        int(
                            np.sum(
                                accepted
                            )
                        )
                    )

                    standalone_impostor_total += (
                        len(
                            impostor_indices
                        )
                    )

                standalone_far = (
                    standalone_impostor_accept
                    /
                    standalone_impostor_total
                    if standalone_impostor_total
                    > 0
                    else 0.0
                )

                # =================================================
                # Add standalone values
                # =================================================

                result[
                    "standalone_gar"
                ] = standalone_gar

                result[
                    "standalone_far"
                ] = standalone_far

                result[
                    "standalone_frr"
                ] = (
                    1.0
                    -
                    standalone_gar
                )

                all_rows.append(
                    result
                )

    return all_rows


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 110)
    print(
        "SEQUENCE STATISTICS + CONTACT GATE "
        "FORMAL DOOR ACCESS BENCHMARK"
    )
    print("=" * 110)

    print()
    print(
        "Pipeline:"
    )

    print(
        "    Raw Pressure 50 x 16"
    )

    print(
        "          ↓"
    )

    print(
        "    Contact Gate"
    )

    print(
        "          ↓"
    )

    print(
        "    Sequence Statistics"
    )

    print(
        "          ↓"
    )

    print(
        "    Euclidean Identity Authentication"
    )

    print()
    print(
        f"Dataset                     : "
        f"{DATA_DIR}"
    )

    print(
        f"Users                       : "
        f"{USERS}"
    )

    print(
        f"No-contact                  : "
        f"{NO_CONTACT_USER}"
    )

    print(
        f"Registration counts         : "
        f"{REGISTRATION_COUNTS}"
    )

    print(
        f"Test count                  : "
        f"{TEST_COUNT}"
    )

    print(
        f"Repeats                     : "
        f"{REPEATS}"
    )

    print(
        f"Background calibration     : "
        f"{BACKGROUND_CALIBRATION_COUNT}"
    )

    print(
        f"Sequence threshold K        : "
        f"{SEQUENCE_THRESHOLD_K}"
    )

    # ========================================================
    # Load
    # ========================================================

    X, labels = load_data()

    print_dataset_information(
        X,
        labels,
    )

    # ========================================================
    # Run
    # ========================================================

    results = run_benchmark(
        X,
        labels,
    )

    # ========================================================
    # Results
    # ========================================================

    print_contact_gate_summary(
        results
    )

    print_sequence_statistics_summary(
        results
    )

    print_e2e_summary(
        results
    )

    print_standalone_vs_e2e(
        results
    )

    print_threshold_debug(
        results
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 110)
    print(
        "Benchmark Finished"
    )
    print("=" * 110)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()