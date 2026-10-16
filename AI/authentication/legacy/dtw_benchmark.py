from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
from authentication.contact_gate import (
    ContactGate,
)

# ============================================================
# Project Path
# ============================================================

AI_ROOT = Path(__file__).resolve().parents[1]

if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


# ============================================================
# Import
# ============================================================

from data_loader import load_dataset

from AI.authentication.legacy.authentication import AuthenticationSystem


# ============================================================
# Configuration
# ============================================================

USERS = (
    "amber",
    "jay",
    "666",
)

# Background is NOT a registered user.
NO_CONTACT_USER = "background"

# Same formal benchmark setting as Sequence Statistics
REGISTRATION_COUNTS = (
    5,
    10,
    20,
)

TEST_COUNT = 30
REPEATS = 20

RANDOM_SEED = 42

# ------------------------------------------------------------
# DTW configuration
# ------------------------------------------------------------
CONTACT_GATE_THRESHOLD = 2_830_340.0
# Sakoe-Chiba warping window.
#
# 5 means that frame i can only match frames around
# i +/- 5.
#
# This prevents DTW from making extremely unrealistic
# time alignments.
DTW_WINDOW = 5

# ------------------------------------------------------------
# Threshold configuration
# ------------------------------------------------------------

THRESHOLD_K = 2.0


# ============================================================
# Data Loading
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:

    X, y = load_dataset(DATA_DIR)

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to be 3D, got shape {X.shape}"
        )

    if X.shape[1:] != (50, 16):
        raise ValueError(
            "This benchmark expects every sample to have "
            f"shape (50, 16), got {X.shape[1:]}"
        )

    return X, np.asarray(y)


# ============================================================
# User Data
# ============================================================

def get_user_samples(
    X: np.ndarray,
    y: np.ndarray,
    user_id: str,
) -> np.ndarray:

    mask = np.asarray(y) == user_id

    samples = X[mask]

    if len(samples) == 0:
        raise ValueError(
            f"No samples found for user '{user_id}'."
        )

    return samples


# ============================================================
# Registration-only Min/Max Normalization
# ============================================================

def calculate_registration_minmax(
    registration_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate sensor-wise Min/Max ONLY from registration data.

    registration_samples:
        shape = (N, 50, 16)

    Returns:
        sensor_min: shape (16,)
        sensor_max: shape (16,)

    IMPORTANT:
        Test / impostor / background data are NOT used here.
    """

    samples = np.asarray(
        registration_samples,
        dtype=np.float32,
    )

    if samples.ndim != 3:
        raise ValueError(
            f"Expected 3D registration data, "
            f"got {samples.shape}"
        )

    sensor_min = np.min(
        samples,
        axis=(0, 1),
    )

    sensor_max = np.max(
        samples,
        axis=(0, 1),
    )

    return sensor_min, sensor_max


def normalize_sequence(
    sample: np.ndarray,
    sensor_min: np.ndarray,
    sensor_max: np.ndarray,
) -> np.ndarray:
    """
    Normalize one 50x16 sequence.

    Each sensor/channel is normalized independently.

    IMPORTANT:
        Min/Max come ONLY from registration data.
    """

    sample = np.asarray(
        sample,
        dtype=np.float32,
    )

    if sample.shape != (50, 16):
        raise ValueError(
            f"Expected sample shape (50, 16), "
            f"got {sample.shape}"
        )

    denominator = sensor_max - sensor_min

    # Prevent division by zero for sensors that never changed
    denominator = np.where(
        denominator == 0,
        1.0,
        denominator,
    )

    normalized = (
        sample - sensor_min
    ) / denominator

    return normalized.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Multivariate DTW
# ============================================================

def dtw_distance(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
    window: int = DTW_WINDOW,
) -> float:
    """
    Multivariate DTW for 50x16 raw pressure sequences.

    Each frame is a 16-dimensional pressure vector.

    Local cost:
        Euclidean distance between two 16-D frames.

    DTW:
        Finds the minimum-cost temporal alignment.

    Returns:
        Average DTW path cost.

    Using average path cost instead of total path cost makes
    the result less sensitive to the exact length of the
    warping path.
    """

    a = np.asarray(
        sequence_a,
        dtype=np.float32,
    )

    b = np.asarray(
        sequence_b,
        dtype=np.float32,
    )

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            "DTW inputs must be 2D sequences."
        )

    if a.shape[1] != 16 or b.shape[1] != 16:
        raise ValueError(
            "DTW expects 16 pressure channels."
        )

    n = a.shape[0]
    m = b.shape[0]

    # Sakoe-Chiba window
    window = max(
        int(window),
        abs(n - m),
    )

    infinity = np.float64(np.inf)

    # Cost matrix
    cost = np.full(
        (n + 1, m + 1),
        infinity,
        dtype=np.float64,
    )

    # Path length matrix
    path_length = np.zeros(
        (n + 1, m + 1),
        dtype=np.int32,
    )

    cost[0, 0] = 0.0

    for i in range(1, n + 1):

        start_j = max(
            1,
            i - window,
        )

        end_j = min(
            m,
            i + window,
        )

        frame_a = a[i - 1]

        for j in range(start_j, end_j + 1):

            frame_b = b[j - 1]

            # ------------------------------------------------
            # Local distance between two 16-D pressure frames
            # ------------------------------------------------

            local_cost = float(
                np.linalg.norm(
                    frame_a - frame_b
                )
            )

            # ------------------------------------------------
            # Three possible DTW transitions
            #
            # 1. diagonal
            # 2. vertical
            # 3. horizontal
            # ------------------------------------------------

            previous_costs = (
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

            best_index = int(
                np.argmin(previous_costs)
            )

            best_previous_cost = (
                previous_costs[best_index]
            )

            if not np.isfinite(
                best_previous_cost
            ):
                continue

            cost[i, j] = (
                local_cost
                + best_previous_cost
            )

            if best_index == 0:
                previous_length = (
                    path_length[i - 1, j]
                )

            elif best_index == 1:
                previous_length = (
                    path_length[i, j - 1]
                )

            else:
                previous_length = (
                    path_length[i - 1, j - 1]
                )

            path_length[i, j] = (
                previous_length + 1
            )

    total_cost = cost[n, m]
    total_length = path_length[n, m]

    if not np.isfinite(total_cost):
        return float("inf")

    if total_length <= 0:
        return float("inf")

    # --------------------------------------------------------
    # Average path cost
    # --------------------------------------------------------

    return float(
        total_cost / total_length
    )


# ============================================================
# Prepare Registration Sequences
# ============================================================

def prepare_registration_sequences(
    registration_samples: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:

    sensor_min, sensor_max = (
        calculate_registration_minmax(
            registration_samples
        )
    )

    normalized_sequences = []

    for sample in registration_samples:

        normalized = normalize_sequence(
            sample,
            sensor_min,
            sensor_max,
        )

        normalized_sequences.append(
            normalized
        )

    return (
        normalized_sequences,
        sensor_min,
        sensor_max,
    )


# ============================================================
# Registration Pairwise DTW
# ============================================================

def calculate_registration_distances(
    registration_sequences: list[np.ndarray],
) -> np.ndarray:
    """
    Calculate pairwise DTW distances among registration
    sequences.

    Only registration data are used.

    These distances are later used to calculate threshold.
    """

    distances = []

    count = len(
        registration_sequences
    )

    for i in range(count):

        for j in range(
            i + 1,
            count,
        ):

            distance = dtw_distance(
                registration_sequences[i],
                registration_sequences[j],
                window=DTW_WINDOW,
            )

            distances.append(
                distance
            )

    if not distances:
        raise ValueError(
            "At least two registration samples "
            "are required to calculate pairwise DTW."
        )

    return np.asarray(
        distances,
        dtype=np.float64,
    )


# ============================================================
# Threshold
# ============================================================

def calculate_threshold(
    registration_distances: np.ndarray,
) -> float:

    distances = np.asarray(
        registration_distances,
        dtype=np.float64,
    )

    mean_distance = float(
        np.mean(distances)
    )

    std_distance = float(
        np.std(distances)
    )

    threshold = (
        mean_distance
        + THRESHOLD_K * std_distance
    )

    return float(threshold)


# ============================================================
# Authenticate One Sequence
# ============================================================

def authenticate_sequence(
    template_sequences: list[np.ndarray],
    threshold: float,
    sample: np.ndarray,
    sensor_min: np.ndarray,
    sensor_max: np.ndarray,
) -> tuple[float, bool]:
    """
    Compare one unknown sequence against the registration
    template.

    Template representation:
        all normalized registration sequences.

    Unknown sequence:
        normalized using registration-only Min/Max.

    Distance:
        minimum DTW distance to the registration templates.

    This is a template-set approach:
        distance(sample, template)
        =
        min DTW(sample, each registration sequence)
    """

    normalized_sample = normalize_sequence(
        sample,
        sensor_min,
        sensor_max,
    )

    distances = []

    for template_sequence in template_sequences:

        distance = dtw_distance(
            template_sequence,
            normalized_sample,
            window=DTW_WINDOW,
        )

        distances.append(
            distance
        )

    distance = float(
        np.min(distances)
    )

    accept = (
        distance <= threshold
    )

    return (
        distance,
        accept,
    )


# ============================================================
# Evaluate Samples
# ============================================================

def evaluate_samples(
    template_sequences: list[np.ndarray],
    threshold: float,
    samples: np.ndarray,
    sensor_min: np.ndarray,
    sensor_max: np.ndarray,
) -> dict[str, float]:

    distances = []

    accept_count = 0

    for sample in samples:

        distance, accept = (
            authenticate_sequence(
                template_sequences,
                threshold,
                sample,
                sensor_min,
                sensor_max,
            )
        )

        distances.append(
            distance
        )

        if accept:
            accept_count += 1

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    total = len(samples)

    reject_count = (
        total - accept_count
    )

    rate = (
        accept_count / total
        if total > 0
        else 0.0
    )

    average_distance = (
        float(np.mean(distances))
        if len(distances) > 0
        else 0.0
    )

    return {
        "accept": float(
            accept_count
        ),

        "reject": float(
            reject_count
        ),

        "rate": float(
            rate
        ),

        "average_distance": (
            average_distance
        ),
    }


# ============================================================
# One Experiment
# ============================================================

def run_one_experiment(
    X: np.ndarray,
    y: np.ndarray,
    template_user: str,
    registration_count: int,
    repeat_index: int,
    rng: np.random.Generator,
) -> dict:

    # ========================================================
    # Genuine pool
    # ========================================================

    genuine_pool = get_user_samples(
        X,
        y,
        template_user,
    )

    required = (
        registration_count
        + TEST_COUNT
    )

    if len(genuine_pool) < required:

        raise ValueError(
            f"{template_user} has "
            f"{len(genuine_pool)} samples, "
            f"but needs {required}."
        )

    # ========================================================
    # SAME split logic as formal benchmark
    # ========================================================

    shuffled_indices = rng.permutation(
        len(genuine_pool)
    )

    registration_indices = (
        shuffled_indices[
            :registration_count
        ]
    )

    genuine_indices = (
        shuffled_indices[
            registration_count:
            registration_count + TEST_COUNT
        ]
    )

    registration_samples = (
        genuine_pool[
            registration_indices
        ]
    )

    genuine_samples = (
        genuine_pool[
            genuine_indices
        ]
    )

    # ========================================================
    # Registration-only normalization
    # ========================================================

    (
        template_sequences,
        sensor_min,
        sensor_max,
    ) = prepare_registration_sequences(
        registration_samples
    )

    # ========================================================
    # Registration pairwise distances
    #
    # IMPORTANT:
    # threshold is calculated ONLY from registration data.
    # ========================================================

    registration_distances = (
        calculate_registration_distances(
            template_sequences
        )
    )

    threshold = calculate_threshold(
        registration_distances
    )

    # ========================================================
    # Genuine test
    # ========================================================

    genuine_result = evaluate_samples(
        template_sequences,
        threshold,
        genuine_samples,
        sensor_min,
        sensor_max,
    )

    # ========================================================
    # Impostor test
    #
    # Each other registered user:
    # randomly select 30 samples.
    # ========================================================

    impostor_results = {}

    total_impostor_accept = 0
    total_impostor_samples = 0

    impostor_distances = []

    for impostor_user in USERS:

        if impostor_user == template_user:
            continue

        impostor_pool = get_user_samples(
            X,
            y,
            impostor_user,
        )

        if len(impostor_pool) < TEST_COUNT:

            raise ValueError(
                f"{impostor_user} has "
                f"{len(impostor_pool)} samples, "
                f"but needs {TEST_COUNT}."
            )

        impostor_indices = rng.choice(
            len(impostor_pool),
            size=TEST_COUNT,
            replace=False,
        )

        impostor_samples = (
            impostor_pool[
                impostor_indices
            ]
        )

        result = evaluate_samples(
            template_sequences,
            threshold,
            impostor_samples,
            sensor_min,
            sensor_max,
        )

        impostor_results[
            impostor_user
        ] = result

        total_impostor_accept += int(
            result["accept"]
        )

        total_impostor_samples += (
            TEST_COUNT
        )

        # IMPORTANT:
        # Store the weighted sample-level
        # distance information.
        #
        # Average distance is only for display.
        impostor_distances.append(
            result["average_distance"]
        )

    # ========================================================
    # FAR
    #
    # All impostor samples are pooled.
    # ========================================================

    far = (
        total_impostor_accept
        / total_impostor_samples
        if total_impostor_samples > 0
        else 0.0
    )

    # ========================================================
    # No-contact
    #
    # Background is NOT a user.
    # ========================================================

    no_contact_pool = get_user_samples(
        X,
        y,
        NO_CONTACT_USER,
    )

    if len(no_contact_pool) < TEST_COUNT:

        raise ValueError(
            f"{NO_CONTACT_USER} has "
            f"{len(no_contact_pool)} samples, "
            f"but needs {TEST_COUNT}."
        )

    no_contact_indices = rng.choice(
        len(no_contact_pool),
        size=TEST_COUNT,
        replace=False,
    )

    no_contact_samples = (
        no_contact_pool[
            no_contact_indices
        ]
    )

    no_contact_result = evaluate_samples(
        template_sequences,
        threshold,
        no_contact_samples,
        sensor_min,
        sensor_max,
    )

    # ========================================================
    # Final metrics
    # ========================================================

    gar = genuine_result["rate"]

    frr = 1.0 - gar

    no_contact_far = (
        no_contact_result["rate"]
    )

    return {

        "template_user":
            template_user,

        "registration_count":
            registration_count,

        "repeat":
            repeat_index,

        "threshold":
            float(threshold),

        "gar":
            float(gar),

        "frr":
            float(frr),

        "genuine_accept":
            genuine_result["accept"],

        "genuine_reject":
            genuine_result["reject"],

        "genuine_distance":
            genuine_result[
                "average_distance"
            ],

        "far":
            float(far),

        "impostor_distance":
            float(
                np.mean(
                    impostor_distances
                )
            ),

        "no_contact_far":
            float(no_contact_far),

        "no_contact_distance":
            no_contact_result[
                "average_distance"
            ],

        "impostor_results":
            impostor_results,
    }


# ============================================================
# Run Benchmark
# ============================================================

def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict]:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        print()
        print("=" * 100)
        print(
            "DTW Benchmark | "
            f"Registration = {registration_count} | "
            f"Test = {TEST_COUNT} | "
            f"Repeats = {REPEATS}"
        )
        print("=" * 100)

        for repeat in range(
            1,
            REPEATS + 1,
        ):

            for template_user in USERS:

                print(
                    f"Running Reg="
                    f"{registration_count}, "
                    f"Repeat={repeat:02d}, "
                    f"Template={template_user}"
                )

                result = run_one_experiment(
                    X=X,
                    y=y,
                    template_user=template_user,
                    registration_count=(
                        registration_count
                    ),
                    repeat_index=repeat,
                    rng=rng,
                )

                results.append(
                    result
                )

    return results


# ============================================================
# Detailed Template -> Test User
# ============================================================

def print_template_results(
    results: list[dict],
) -> None:

    print()
    print("=" * 115)
    print(
        "DTW TEMPLATE -> TEST USER"
    )
    print("=" * 115)

    print(
        f"{'Reg':<6}"
        f"{'Repeat':<8}"
        f"{'Template':<15}"
        f"{'Test User':<15}"
        f"{'Accept':>10}"
        f"{'Reject':>10}"
        f"{'GAR':>12}"
        f"{'Avg Distance':>18}"
    )

    print("-" * 115)

    for row in results:

        template_user = (
            row["template_user"]
        )

        # ----------------------------------------------------
        # Genuine
        # ----------------------------------------------------

        print(
            f"{row['registration_count']:<6}"
            f"{row['repeat']:<8}"
            f"{template_user:<15}"
            f"{template_user:<15}"
            f"{int(row['genuine_accept']):>10}"
            f"{int(row['genuine_reject']):>10}"
            f"{row['gar'] * 100:>11.2f}%"
            f"{row['genuine_distance']:>18.4f}"
        )

        # ----------------------------------------------------
        # Impostors
        # ----------------------------------------------------

        for (
            test_user,
            test_result,
        ) in row[
            "impostor_results"
        ].items():

            print(
                f"{row['registration_count']:<6}"
                f"{row['repeat']:<8}"
                f"{template_user:<15}"
                f"{test_user:<15}"
                f"{int(test_result['accept']):>10}"
                f"{int(test_result['reject']):>10}"
                f"{test_result['rate'] * 100:>11.2f}%"
                f"{test_result['average_distance']:>18.4f}"
            )

        # ----------------------------------------------------
        # No-contact
        # ----------------------------------------------------

        print(
            f"{row['registration_count']:<6}"
            f"{row['repeat']:<8}"
            f"{template_user:<15}"
            f"{NO_CONTACT_USER:<15}"
            f"{'N/A':>10}"
            f"{'N/A':>10}"
            f"{row['no_contact_far'] * 100:>11.2f}%"
            f"{row['no_contact_distance']:>18.4f}"
        )


# ============================================================
# Formal Summary
# ============================================================

def print_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 125)
    print(
        "DTW RAW 50x16 FORMAL DOOR ACCESS SUMMARY"
    )
    print("=" * 125)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'GAR Mean':>12}"
        f"{'GAR Std':>12}"
        f"{'FRR Mean':>12}"
        f"{'FRR Std':>12}"
        f"{'FAR Mean':>12}"
        f"{'FAR Std':>12}"
        f"{'No-contact FAR':>18}"
        f"{'Threshold':>14}"
    )

    print("-" * 125)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = [
                row
                for row in results
                if (
                    row[
                        "registration_count"
                    ]
                    == registration_count
                    and
                    row[
                        "template_user"
                    ]
                    == user
                )
            ]

            gar = np.asarray([
                row["gar"]
                for row in rows
            ])

            frr = np.asarray([
                row["frr"]
                for row in rows
            ])

            far = np.asarray([
                row["far"]
                for row in rows
            ])

            no_contact_far = np.asarray([
                row["no_contact_far"]
                for row in rows
            ])

            thresholds = np.asarray([
                row["threshold"]
                for row in rows
            ])

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{np.mean(gar) * 100:>11.2f}%"
                f"{np.std(gar) * 100:>11.2f}%"
                f"{np.mean(frr) * 100:>11.2f}%"
                f"{np.std(frr) * 100:>11.2f}%"
                f"{np.mean(far) * 100:>11.2f}%"
                f"{np.std(far) * 100:>11.2f}%"
                f"{np.mean(no_contact_far) * 100:>17.2f}%"
                f"{np.mean(thresholds):>14.4f}"
            )


# ============================================================
# Distance Summary
# ============================================================

def print_distance_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 105)
    print(
        "DTW RAW 50x16 DISTANCE SUMMARY"
    )
    print("=" * 105)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Genuine Distance':>20}"
        f"{'Impostor Distance':>20}"
        f"{'No-contact Distance':>22}"
    )

    print("-" * 105)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = [
                row
                for row in results
                if (
                    row[
                        "registration_count"
                    ]
                    == registration_count
                    and
                    row[
                        "template_user"
                    ]
                    == user
                )
            ]

            genuine_distance = (
                np.mean([
                    row[
                        "genuine_distance"
                    ]
                    for row in rows
                ])
            )

            impostor_distance = (
                np.mean([
                    row[
                        "impostor_distance"
                    ]
                    for row in rows
                ])
            )

            no_contact_distance = (
                np.mean([
                    row[
                        "no_contact_distance"
                    ]
                    for row in rows
                ])
            )

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{genuine_distance:>20.4f}"
                f"{impostor_distance:>20.4f}"
                f"{no_contact_distance:>22.4f}"
            )


# ============================================================
# Distance Separation
# ============================================================

def print_distance_separation(
    results: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "DTW RAW 50x16 DISTANCE SEPARATION"
    )
    print("=" * 110)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Genuine':>14}"
        f"{'Threshold':>14}"
        f"{'No-contact':>14}"
        f"{'Impostor':>14}"
        f"{'Imp-Genuine':>16}"
        f"{'NC-Genuine':>16}"
    )

    print("-" * 110)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = [
                row
                for row in results
                if (
                    row[
                        "registration_count"
                    ]
                    == registration_count
                    and
                    row[
                        "template_user"
                    ]
                    == user
                )
            ]

            genuine = np.mean([
                row["genuine_distance"]
                for row in rows
            ])

            threshold = np.mean([
                row["threshold"]
                for row in rows
            ])

            no_contact = np.mean([
                row[
                    "no_contact_distance"
                ]
                for row in rows
            ])

            impostor = np.mean([
                row[
                    "impostor_distance"
                ]
                for row in rows
            ])

            imp_minus_genuine = (
                impostor - genuine
            )

            nc_minus_genuine = (
                no_contact - genuine
            )

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{genuine:>14.4f}"
                f"{threshold:>14.4f}"
                f"{no_contact:>14.4f}"
                f"{impostor:>14.4f}"
                f"{imp_minus_genuine:>16.4f}"
                f"{nc_minus_genuine:>16.4f}"
            )


# ============================================================
# Registration Effect
# ============================================================

def print_registration_effect(
    results: list[dict],
) -> None:

    print()
    print("=" * 100)
    print(
        "DTW REGISTRATION SIZE EFFECT"
    )
    print("=" * 100)

    print(
        f"{'User':<15}"
        f"{'Reg':>8}"
        f"{'GAR':>12}"
        f"{'FRR':>12}"
        f"{'FAR':>12}"
        f"{'No-contact FAR':>18}"
        f"{'Threshold':>14}"
    )

    print("-" * 100)

    for user in USERS:

        for registration_count in (
            REGISTRATION_COUNTS
        ):

            rows = [
                row
                for row in results
                if (
                    row[
                        "template_user"
                    ]
                    == user
                    and
                    row[
                        "registration_count"
                    ]
                    == registration_count
                )
            ]

            gar = np.mean([
                row["gar"]
                for row in rows
            ])

            frr = np.mean([
                row["frr"]
                for row in rows
            ])

            far = np.mean([
                row["far"]
                for row in rows
            ])

            no_contact_far = np.mean([
                row[
                    "no_contact_far"
                ]
                for row in rows
            ])

            threshold = np.mean([
                row["threshold"]
                for row in rows
            ])

            print(
                f"{user:<15}"
                f"{registration_count:>8}"
                f"{gar * 100:>11.2f}%"
                f"{frr * 100:>11.2f}%"
                f"{far * 100:>11.2f}%"
                f"{no_contact_far * 100:>17.2f}%"
                f"{threshold:>14.4f}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    X, y = load_data()

    print()
    print("=" * 110)
    print(
        "DTW RAW 50x16 FORMAL DOOR ACCESS BENCHMARK"
    )
    print("=" * 110)

    print(
        f"Dataset shape       : {X.shape}"
    )

    print(
        f"Users               : {USERS}"
    )

    print(
        f"No-contact          : {NO_CONTACT_USER}"
    )

    print(
        f"Registration counts : "
        f"{REGISTRATION_COUNTS}"
    )

    print(
        f"Unknown test count  : "
        f"{TEST_COUNT}"
    )

    print(
        f"Repeats             : "
        f"{REPEATS}"
    )

    print(
        f"DTW window          : "
        f"{DTW_WINDOW}"
    )

    print()
    print(
        "Input representation:"
    )

    print(
        "50 frames × 16 raw pressure sensors"
    )

    print()
    print(
        "Normalization:"
    )

    print(
        "Sensor-wise Min/Max calculated "
        "ONLY from registration samples."
    )

    print()
    print(
        "DTW:"
    )

    print(
        "Multivariate DTW using 16-D frame "
        "Euclidean local distance."
    )

    print()
    print(
        "Template:"
    )

    print(
        "Registration sequences are kept as "
        "individual normalized sequences."
    )

    print(
        "Unknown distance = minimum DTW distance "
        "to registration sequences."
    )

    print()
    print(
        "Threshold:"
    )

    print(
        "Pairwise registration DTW mean "
        f"+ {THRESHOLD_K:.1f} × std"
    )

    print()
    print(
        "Evaluation:"
    )

    print(
        "5 / 10 / 20 registration "
        "+ 30 unknown genuine "
        "+ 30 impostor per user "
        "+ 30 no-contact "
        "+ 20 repeats"
    )

    print("=" * 110)

    # ========================================================
    # Run
    # ========================================================

    results = run_benchmark(
        X,
        y,
    )

    # ========================================================
    # Outputs
    # ========================================================

    print_template_results(
        results
    )

    print_summary(
        results
    )

    print_distance_summary(
        results
    )

    print_distance_separation(
        results
    )

    print_registration_effect(
        results
    )

    print()
    print("=" * 110)
    print(
        "DTW RAW 50x16 Benchmark Finished"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()