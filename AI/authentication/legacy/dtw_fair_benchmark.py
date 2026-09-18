from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


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

# DTW Sakoe-Chiba window
DTW_WINDOW = 5

# Threshold:
# registration pairwise DTW mean + K * std
THRESHOLD_K = 2.0


# ============================================================
# Data
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:

    X, y = load_dataset(DATA_DIR)

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to be 3D, got {X.shape}"
        )

    return X, y


def get_user_samples(
    X: np.ndarray,
    y: np.ndarray,
    user_id: str,
) -> np.ndarray:

    mask = np.asarray(y) == user_id

    return X[mask]


# ============================================================
# DTW
# ============================================================

def frame_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    """
    Euclidean distance between two frames.

    a shape = (16,)
    b shape = (16,)

    Raw 16 pressure sensors are compared directly.
    """

    diff = a.astype(np.float64) - b.astype(np.float64)

    return float(np.linalg.norm(diff))


def dtw_distance(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
    window: int = DTW_WINDOW,
) -> float:

    """
    Raw 50 x 16 DTW.

    sequence_a:
        (50, 16)

    sequence_b:
        (50, 16)

    DTW aligns the temporal dimension.
    Each frame contains the 16 raw pressure values.

    No engineered features are used.
    """

    a = np.asarray(
        sequence_a,
        dtype=np.float64,
    )

    b = np.asarray(
        sequence_b,
        dtype=np.float64,
    )

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected 2D sequences, got "
            f"{a.shape} and {b.shape}"
        )

    if a.shape[1] != 16 or b.shape[1] != 16:
        raise ValueError(
            f"Expected 16 pressure dimensions, got "
            f"{a.shape[1]} and {b.shape[1]}"
        )

    n = a.shape[0]
    m = b.shape[0]

    window = max(
        window,
        abs(n - m),
    )

    inf = np.inf

    dp = np.full(
        (n + 1, m + 1),
        inf,
        dtype=np.float64,
    )

    dp[0, 0] = 0.0

    for i in range(1, n + 1):

        start = max(
            1,
            i - window,
        )

        end = min(
            m,
            i + window,
        )

        for j in range(start, end + 1):

            cost = frame_distance(
                a[i - 1],
                b[j - 1],
            )

            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    # --------------------------------------------------------
    # Important:
    #
    # Normalize by path length.
    #
    # This makes the DTW distance more comparable between
    # sequences and avoids the raw accumulated path cost
    # becoming unnecessarily huge.
    # --------------------------------------------------------

    return float(
        dp[n, m] / max(n, m)
    )


# ============================================================
# Registration Template
# ============================================================

def build_registration_template(
    registration_samples: np.ndarray,
) -> np.ndarray:

    """
    DTW does not create a mean 50x16 template.

    Instead, all registration sequences are retained.

    During authentication:
        test sequence
        is compared against
        every registration sequence

    and the minimum DTW distance is used.

    This is the DTW analogue of a template.
    """

    if registration_samples.ndim != 3:
        raise ValueError(
            "Expected registration samples to be 3D"
        )

    return np.asarray(
        registration_samples,
        dtype=np.float32,
    )


# ============================================================
# Distance to Template
# ============================================================

def distance_to_template(
    template_sequences: np.ndarray,
    sample: np.ndarray,
) -> float:

    """
    Compare one unknown sequence against the
    complete registration template.

    Distance =
        minimum DTW distance between the sample
        and any registration sequence.
    """

    distances = []

    for registration_sequence in template_sequences:

        distance = dtw_distance(
            registration_sequence,
            sample,
            window=DTW_WINDOW,
        )

        distances.append(distance)

    return float(
        np.min(distances)
    )


# ============================================================
# Build Threshold
# ============================================================

def build_threshold(
    registration_samples: np.ndarray,
) -> tuple[float, np.ndarray]:

    """
    Threshold is calculated ONLY from registration data.

    For every registration sequence:

        compare against all other registration sequences

    Then:

        threshold =
            mean(pairwise distances)
            + 2 * std(pairwise distances)

    IMPORTANT:
        no genuine test data
        no impostor data
        no background data
        are used here.
    """

    count = len(registration_samples)

    if count < 2:
        raise ValueError(
            "At least 2 registration samples "
            "are required to build threshold."
        )

    distances = []

    for i in range(count):

        for j in range(i + 1, count):

            distance = dtw_distance(
                registration_samples[i],
                registration_samples[j],
                window=DTW_WINDOW,
            )

            distances.append(distance)

    distances = np.asarray(
        distances,
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

    return (
        float(threshold),
        distances,
    )


# ============================================================
# Evaluate
# ============================================================

def evaluate_samples(
    template_sequences: np.ndarray,
    threshold: float,
    samples: np.ndarray,
) -> dict[str, float]:

    distances = []

    accept_count = 0

    for sample in samples:

        distance = distance_to_template(
            template_sequences,
            sample,
        )

        distances.append(distance)

        if distance <= threshold:
            accept_count += 1

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    total = len(samples)

    reject_count = (
        total - accept_count
    )

    acceptance_rate = (
        accept_count / total
        if total > 0
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
            acceptance_rate
        ),

        "average_distance": (
            float(np.mean(distances))
            if len(distances) > 0
            else 0.0
        ),

        "distances": distances,
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
    # 1. Genuine pool
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
            f"but needs {required}"
        )

    # ========================================================
    # 2. Split registration / unknown genuine
    # ========================================================

    indices = rng.permutation(
        len(genuine_pool)
    )

    registration_indices = indices[
        :registration_count
    ]

    genuine_test_indices = indices[
        registration_count:
        registration_count + TEST_COUNT
    ]

    registration_samples = (
        genuine_pool[
            registration_indices
        ]
    )

    genuine_test_samples = (
        genuine_pool[
            genuine_test_indices
        ]
    )

    # ========================================================
    # 3. Build DTW template
    # ========================================================

    template_sequences = (
        build_registration_template(
            registration_samples
        )
    )

    # ========================================================
    # 4. Build threshold
    #
    # ONLY registration data
    # ========================================================

    threshold, registration_distances = (
        build_threshold(
            registration_samples
        )
    )

    # ========================================================
    # 5. Genuine
    # ========================================================

    genuine_result = evaluate_samples(
        template_sequences,
        threshold,
        genuine_test_samples,
    )

    # ========================================================
    # 6. Impostors
    # ========================================================

    impostor_results = {}

    total_impostor_accept = 0
    total_impostor_samples = 0

    all_impostor_distances = []

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
                f"{impostor_user} has only "
                f"{len(impostor_pool)} samples"
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

        all_impostor_distances.extend(
            result["distances"].tolist()
        )

    # ========================================================
    # 7. FAR
    #
    # IMPORTANT:
    # sample-level aggregation
    # ========================================================

    far = (
        total_impostor_accept
        / total_impostor_samples
    )

    # ========================================================
    # 8. No-contact
    # ========================================================

    no_contact_pool = get_user_samples(
        X,
        y,
        NO_CONTACT_USER,
    )

    if len(no_contact_pool) < TEST_COUNT:

        raise ValueError(
            f"{NO_CONTACT_USER} has only "
            f"{len(no_contact_pool)} samples"
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
    )

    # ========================================================
    # 9. Metrics
    # ========================================================

    gar = genuine_result["rate"]

    frr = 1.0 - gar

    no_contact_far = (
        no_contact_result["rate"]
    )

    # ========================================================
    # 10. Distance separation
    # ========================================================

    genuine_distance = (
        genuine_result[
            "average_distance"
        ]
    )

    impostor_distance = float(
        np.mean(all_impostor_distances)
    )

    no_contact_distance = (
        no_contact_result[
            "average_distance"
        ]
    )

    return {

        "template_user":
            template_user,

        "registration_count":
            registration_count,

        "repeat":
            repeat_index,

        "threshold":
            threshold,

        "gar":
            gar,

        "frr":
            frr,

        "far":
            far,

        "no_contact_far":
            no_contact_far,

        "genuine_accept":
            genuine_result["accept"],

        "genuine_reject":
            genuine_result["reject"],

        "genuine_distance":
            genuine_distance,

        "impostor_distance":
            impostor_distance,

        "no_contact_distance":
            no_contact_distance,

        "registration_distance_mean":
            float(
                np.mean(
                    registration_distances
                )
            ),

        "registration_distance_std":
            float(
                np.std(
                    registration_distances
                )
            ),

        "impostor_results":
            impostor_results,

        "no_contact_result":
            no_contact_result,
    }


# ============================================================
# Benchmark
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
        print("=" * 110)

        print(
            f"DTW RAW 50x16 | "
            f"Registration={registration_count} | "
            f"Test={TEST_COUNT} | "
            f"Repeats={REPEATS}"
        )

        print("=" * 110)

        for repeat in range(
            1,
            REPEATS + 1,
        ):

            for template_user in USERS:

                print(
                    f"Running "
                    f"Reg={registration_count}, "
                    f"Repeat={repeat:02d}, "
                    f"Template={template_user}"
                )

                result = run_one_experiment(
                    X=X,
                    y=y,
                    template_user=template_user,
                    registration_count=registration_count,
                    repeat_index=repeat,
                    rng=rng,
                )

                results.append(result)

    return results


# ============================================================
# Summary
# ============================================================

def print_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 120)
    print(
        "DTW RAW 50x16 FORMAL DOOR ACCESS SUMMARY"
    )
    print("=" * 120)

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

    print("-" * 120)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = [
                r
                for r in results
                if (
                    r[
                        "registration_count"
                    ]
                    == registration_count
                    and
                    r[
                        "template_user"
                    ]
                    == user
                )
            ]

            gar = np.array([
                r["gar"]
                for r in rows
            ])

            frr = np.array([
                r["frr"]
                for r in rows
            ])

            far = np.array([
                r["far"]
                for r in rows
            ])

            no_contact_far = np.array([
                r["no_contact_far"]
                for r in rows
            ])

            thresholds = np.array([
                r["threshold"]
                for r in rows
            ])

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{np.mean(gar)*100:>11.2f}%"
                f"{np.std(gar)*100:>11.2f}%"
                f"{np.mean(frr)*100:>11.2f}%"
                f"{np.std(frr)*100:>11.2f}%"
                f"{np.mean(far)*100:>11.2f}%"
                f"{np.std(far)*100:>11.2f}%"
                f"{np.mean(no_contact_far)*100:>17.2f}%"
                f"{np.mean(thresholds):>14.4f}"
            )


# ============================================================
# Distance Summary
# ============================================================

def print_distance_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "DTW RAW 50x16 DISTANCE SUMMARY"
    )
    print("=" * 110)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Genuine Distance':>22}"
        f"{'Impostor Distance':>22}"
        f"{'No-contact Distance':>22}"
    )

    print("-" * 110)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = [
                r
                for r in results
                if (
                    r[
                        "registration_count"
                    ]
                    == registration_count
                    and
                    r[
                        "template_user"
                    ]
                    == user
                )
            ]

            genuine = np.mean([
                r["genuine_distance"]
                for r in rows
            ])

            impostor = np.mean([
                r["impostor_distance"]
                for r in rows
            ])

            no_contact = np.mean([
                r["no_contact_distance"]
                for r in rows
            ])

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{genuine:>22.4f}"
                f"{impostor:>22.4f}"
                f"{no_contact:>22.4f}"
            )


# ============================================================
# Distance Separation
# ============================================================

def print_distance_separation(
    results: list[dict],
) -> None:

    print()
    print("=" * 120)
    print(
        "DTW RAW 50x16 DISTANCE SEPARATION"
    )
    print("=" * 120)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Genuine':>16}"
        f"{'Threshold':>16}"
        f"{'No-contact':>16}"
        f"{'Impostor':>16}"
        f"{'Imp-Genuine':>18}"
        f"{'NC-Genuine':>18}"
    )

    print("-" * 120)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = [
                r
                for r in results
                if (
                    r[
                        "registration_count"
                    ]
                    == registration_count
                    and
                    r[
                        "template_user"
                    ]
                    == user
                )
            ]

            genuine = np.mean([
                r["genuine_distance"]
                for r in rows
            ])

            threshold = np.mean([
                r["threshold"]
                for r in rows
            ])

            no_contact = np.mean([
                r["no_contact_distance"]
                for r in rows
            ])

            impostor = np.mean([
                r["impostor_distance"]
                for r in rows
            ])

            imp_gap = (
                impostor
                - genuine
            )

            nc_gap = (
                no_contact
                - genuine
            )

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{genuine:>16.4f}"
                f"{threshold:>16.4f}"
                f"{no_contact:>16.4f}"
                f"{impostor:>16.4f}"
                f"{imp_gap:>18.4f}"
                f"{nc_gap:>18.4f}"
            )


# ============================================================
# Registration Size Effect
# ============================================================

def print_registration_effect(
    results: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "DTW RAW 50x16 REGISTRATION SIZE EFFECT"
    )
    print("=" * 110)

    print(
        f"{'User':<15}"
        f"{'Reg':>8}"
        f"{'GAR':>12}"
        f"{'FRR':>12}"
        f"{'FAR':>12}"
        f"{'No-contact FAR':>20}"
        f"{'Threshold':>16}"
    )

    print("-" * 110)

    for user in USERS:

        for registration_count in (
            REGISTRATION_COUNTS
        ):

            rows = [
                r
                for r in results
                if (
                    r[
                        "template_user"
                    ]
                    == user
                    and
                    r[
                        "registration_count"
                    ]
                    == registration_count
                )
            ]

            gar = np.mean([
                r["gar"]
                for r in rows
            ])

            frr = np.mean([
                r["frr"]
                for r in rows
            ])

            far = np.mean([
                r["far"]
                for r in rows
            ])

            nc_far = np.mean([
                r["no_contact_far"]
                for r in rows
            ])

            threshold = np.mean([
                r["threshold"]
                for r in rows
            ])

            print(
                f"{user:<15}"
                f"{registration_count:>8}"
                f"{gar*100:>11.2f}%"
                f"{frr*100:>11.2f}%"
                f"{far*100:>11.2f}%"
                f"{nc_far*100:>19.2f}%"
                f"{threshold:>16.4f}"
            )


# ============================================================
# Overall Mean
# ============================================================

def print_overall_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 100)
    print(
        "DTW RAW 50x16 OVERALL PERFORMANCE"
    )
    print("=" * 100)

    gar = np.mean([
        r["gar"]
        for r in results
    ])

    frr = np.mean([
        r["frr"]
        for r in results
    ])

    far = np.mean([
        r["far"]
        for r in results
    ])

    nc_far = np.mean([
        r["no_contact_far"]
        for r in results
    ])

    print(
        f"Average GAR           : "
        f"{gar*100:.2f}%"
    )

    print(
        f"Average FRR           : "
        f"{frr*100:.2f}%"
    )

    print(
        f"Average FAR           : "
        f"{far*100:.2f}%"
    )

    print(
        f"Average No-contact FAR: "
        f"{nc_far*100:.2f}%"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    X, y = load_data()

    print()
    print("=" * 110)

    print(
        "DTW RAW 50x16 "
        "FAIR FORMAL DOOR ACCESS BENCHMARK"
    )

    print("=" * 110)

    print(
        f"Dataset shape       : {X.shape}"
    )

    print(
        f"Users               : {USERS}"
    )

    print(
        f"No-contact          : "
        f"{NO_CONTACT_USER}"
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

    print(
        "Sequence representation:"
    )

    print(
        "Raw pressure sequence = "
        "50 frames × 16 sensors"
    )

    print(
        "No engineered features"
    )

    print(
        "Threshold:"
    )

    print(
        "registration pairwise DTW "
        f"mean + {THRESHOLD_K} * std"
    )

    print(
        "Template:"
    )

    print(
        "all registration sequences; "
        "authentication uses minimum DTW distance"
    )

    # ========================================================
    # Run
    # ========================================================

    results = run_benchmark(
        X,
        y,
    )

    # ========================================================
    # Output
    # ========================================================

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

    print_overall_summary(
        results
    )

    print()
    print("=" * 110)
    print(
        "DTW RAW 50x16 FAIR BENCHMARK FINISHED"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()