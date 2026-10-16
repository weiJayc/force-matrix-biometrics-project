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


# ============================================================
# Contact Gate Configuration
# ============================================================

# Threshold 的建立方式：
#
# registration contact score
#        mean + K * std
#
# 因為 contact score 越大代表越有可能真的接觸，
# 所以：
#
# score >= threshold -> CONTACT
# score <  threshold -> NO CONTACT
#
CONTACT_K = 2.0


# ============================================================
# Data
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_dataset(DATA_DIR)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X shape = (samples, frames, sensors), "
            f"got {X.shape}"
        )

    if X.shape[1:] != (50, 16):
        raise ValueError(
            f"Expected each sample to have shape (50, 16), "
            f"got {X.shape[1:]}"
        )

    return X, y


def get_user_samples(
    X: np.ndarray,
    y: np.ndarray,
    user_id: str,
) -> np.ndarray:

    mask = y == user_id

    return X[mask]


# ============================================================
# Contact Score
# ============================================================

def contact_score(sample: np.ndarray) -> float:
    """
    Calculate contact score from raw 50x16 pressure data.

    This version intentionally uses RAW pressure only.

    Main idea:
        A genuine grip should generate pressure.
        A no-contact sample should have much smaller pressure.

    Score:
        mean of frame-wise total pressure.

    sample shape:
        (50, 16)
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

    # 50 frames
    # each frame has 16 pressure sensors
    pressure_sum = np.sum(
        sample,
        axis=1,
    )

    # Whole-sequence average pressure
    score = np.mean(
        pressure_sum
    )

    return float(score)


# ============================================================
# Optional alternative scores
# ============================================================

def contact_score_max(sample: np.ndarray) -> float:
    """
    Maximum pressure_sum during the whole sequence.
    """

    pressure_sum = np.sum(
        sample,
        axis=1,
    )

    return float(
        np.max(pressure_sum)
    )


def contact_score_area(sample: np.ndarray) -> float:
    """
    Average number of active sensors.

    A sensor is considered active when its value > 0.
    """

    active = sample > 0

    area = np.sum(
        active,
        axis=1,
    )

    return float(
        np.mean(area)
    )


# ============================================================
# Contact score function
# ============================================================

def calculate_score(
    sample: np.ndarray,
) -> float:

    return contact_score(sample)


# ============================================================
# Build Contact Gate
# ============================================================

def build_contact_gate(
    registration_samples: np.ndarray,
) -> dict[str, float]:
    """
    Build contact threshold using ONLY registration samples.

    IMPORTANT:
        No genuine test samples are used.
        No background samples are used.

    Therefore this does not leak test information.
    """

    scores = np.asarray(
        [
            calculate_score(sample)
            for sample in registration_samples
        ],
        dtype=np.float64,
    )

    mean_score = float(
        np.mean(scores)
    )

    std_score = float(
        np.std(scores)
    )

    threshold = (
        mean_score
        - CONTACT_K * std_score
    )

    # --------------------------------------------------------
    # Why MINUS?
    #
    # Genuine contact:
    #       score should be HIGH
    #
    # No contact:
    #       score should be LOW
    #
    # We therefore accept contact when:
    #
    #       score >= threshold
    #
    # The threshold is placed below the genuine distribution.
    # --------------------------------------------------------

    return {
        "threshold": float(threshold),
        "mean": mean_score,
        "std": std_score,
    }


# ============================================================
# Evaluate Contact
# ============================================================

def evaluate_contact(
    samples: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    scores = np.asarray(
        [
            calculate_score(sample)
            for sample in samples
        ],
        dtype=np.float64,
    )

    contact_mask = scores >= threshold

    contact_count = int(
        np.sum(contact_mask)
    )

    no_contact_count = (
        len(samples) - contact_count
    )

    contact_rate = (
        contact_count / len(samples)
        if len(samples) > 0
        else 0.0
    )

    no_contact_rate = (
        no_contact_count / len(samples)
        if len(samples) > 0
        else 0.0
    )

    return {
        "contact": float(contact_count),
        "no_contact": float(no_contact_count),
        "contact_rate": float(contact_rate),
        "no_contact_rate": float(no_contact_rate),
        "mean_score": (
            float(np.mean(scores))
            if len(scores) > 0
            else 0.0
        ),
        "std_score": (
            float(np.std(scores))
            if len(scores) > 0
            else 0.0
        ),
        "min_score": (
            float(np.min(scores))
            if len(scores) > 0
            else 0.0
        ),
        "max_score": (
            float(np.max(scores))
            if len(scores) > 0
            else 0.0
        ),
    }


# ============================================================
# One Experiment
# ============================================================

def run_one_experiment(
    X: np.ndarray,
    y: np.ndarray,
    user_id: str,
    registration_count: int,
    repeat: int,
    rng: np.random.Generator,
) -> dict:

    # --------------------------------------------------------
    # Genuine pool
    # --------------------------------------------------------

    genuine_pool = get_user_samples(
        X,
        y,
        user_id,
    )

    required = (
        registration_count
        + TEST_COUNT
    )

    if len(genuine_pool) < required:
        raise ValueError(
            f"{user_id}: "
            f"need {required} samples, "
            f"but only {len(genuine_pool)} available."
        )

    # --------------------------------------------------------
    # Random split
    #
    # Registration and genuine test NEVER overlap.
    # --------------------------------------------------------

    indices = rng.permutation(
        len(genuine_pool)
    )

    registration_indices = (
        indices[:registration_count]
    )

    genuine_test_indices = (
        indices[
            registration_count:
            registration_count + TEST_COUNT
        ]
    )

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

    # --------------------------------------------------------
    # Build gate
    # --------------------------------------------------------

    gate = build_contact_gate(
        registration_samples
    )

    threshold = gate["threshold"]

    # --------------------------------------------------------
    # Genuine test
    # --------------------------------------------------------

    genuine_result = evaluate_contact(
        genuine_test_samples,
        threshold,
    )

    # --------------------------------------------------------
    # No-contact test
    # --------------------------------------------------------

    no_contact_pool = get_user_samples(
        X,
        y,
        NO_CONTACT_USER,
    )

    if len(no_contact_pool) < TEST_COUNT:
        raise ValueError(
            f"{NO_CONTACT_USER}: "
            f"need {TEST_COUNT} samples, "
            f"but only {len(no_contact_pool)} available."
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

    no_contact_result = evaluate_contact(
        no_contact_samples,
        threshold,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    genuine_contact_rate = (
        genuine_result["contact_rate"]
    )

    genuine_miss_rate = (
        1.0 - genuine_contact_rate
    )

    # VERY IMPORTANT:
    #
    # No-contact FAR means:
    #
    #     no-contact samples
    #     incorrectly classified as CONTACT
    #
    no_contact_far = (
        no_contact_result["contact_rate"]
    )

    no_contact_rejection_rate = (
        1.0 - no_contact_far
    )

    return {
        "user": user_id,
        "registration_count": registration_count,
        "repeat": repeat,

        "threshold": threshold,

        "registration_mean": gate["mean"],
        "registration_std": gate["std"],

        "genuine_contact_rate":
            genuine_contact_rate,

        "genuine_miss_rate":
            genuine_miss_rate,

        "no_contact_far":
            no_contact_far,

        "no_contact_rejection_rate":
            no_contact_rejection_rate,

        "genuine_mean_score":
            genuine_result["mean_score"],

        "genuine_std_score":
            genuine_result["std_score"],

        "no_contact_mean_score":
            no_contact_result["mean_score"],

        "no_contact_std_score":
            no_contact_result["std_score"],

        "genuine_min_score":
            genuine_result["min_score"],

        "genuine_max_score":
            genuine_result["max_score"],

        "no_contact_min_score":
            no_contact_result["min_score"],

        "no_contact_max_score":
            no_contact_result["max_score"],
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
        print("=" * 100)
        print(
            f"Contact Gate Benchmark | "
            f"Registration = {registration_count} | "
            f"Test = {TEST_COUNT} | "
            f"Repeats = {REPEATS}"
        )
        print("=" * 100)

        for repeat in range(
            1,
            REPEATS + 1,
        ):

            for user_id in USERS:

                print(
                    f"Running "
                    f"Reg={registration_count}, "
                    f"Repeat={repeat:02d}, "
                    f"User={user_id}"
                )

                result = run_one_experiment(
                    X=X,
                    y=y,
                    user_id=user_id,
                    registration_count=registration_count,
                    repeat=repeat,
                    rng=rng,
                )

                results.append(result)

    return results


# ============================================================
# Detailed Results
# ============================================================

def print_detailed_results(
    results: list[dict],
) -> None:

    print()
    print("=" * 125)
    print(
        "CONTACT GATE DETAILED RESULTS"
    )
    print("=" * 125)

    print(
        f"{'Reg':<6}"
        f"{'Repeat':<8}"
        f"{'User':<12}"
        f"{'Threshold':>12}"
        f"{'Genuine CR':>14}"
        f"{'Genuine Miss':>16}"
        f"{'No-contact FAR':>18}"
        f"{'No-contact Reject':>20}"
    )

    print("-" * 125)

    for r in results:

        print(
            f"{r['registration_count']:<6}"
            f"{r['repeat']:<8}"
            f"{r['user']:<12}"
            f"{r['threshold']:>12.4f}"
            f"{r['genuine_contact_rate'] * 100:>13.2f}%"
            f"{r['genuine_miss_rate'] * 100:>15.2f}%"
            f"{r['no_contact_far'] * 100:>17.2f}%"
            f"{r['no_contact_rejection_rate'] * 100:>19.2f}%"
        )


# ============================================================
# Main Summary
# ============================================================

def print_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 125)
    print(
        "CONTACT GATE FORMAL SUMMARY"
    )
    print("=" * 125)

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'Contact Rate':>16}"
        f"{'Miss Rate':>16}"
        f"{'No-contact FAR':>20}"
        f"{'No-contact Reject':>22}"
        f"{'Threshold':>14}"
    )

    print("-" * 125)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user_id in USERS:

            rows = [
                r
                for r in results
                if (
                    r["registration_count"]
                    == registration_count
                    and
                    r["user"]
                    == user_id
                )
            ]

            contact_rates = np.asarray([
                r["genuine_contact_rate"]
                for r in rows
            ])

            miss_rates = np.asarray([
                r["genuine_miss_rate"]
                for r in rows
            ])

            no_contact_fars = np.asarray([
                r["no_contact_far"]
                for r in rows
            ])

            no_contact_rejections = np.asarray([
                r["no_contact_rejection_rate"]
                for r in rows
            ])

            thresholds = np.asarray([
                r["threshold"]
                for r in rows
            ])

            print(
                f"{registration_count:<6}"
                f"{user_id:<12}"
                f"{np.mean(contact_rates) * 100:>15.2f}%"
                f"{np.mean(miss_rates) * 100:>15.2f}%"
                f"{np.mean(no_contact_fars) * 100:>19.2f}%"
                f"{np.mean(no_contact_rejections) * 100:>21.2f}%"
                f"{np.mean(thresholds):>14.4f}"
            )


# ============================================================
# Score Distribution Summary
# ============================================================

def print_score_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 125)
    print(
        "CONTACT SCORE SEPARATION"
    )
    print("=" * 125)

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'Registration Mean':>20}"
        f"{'Genuine Mean':>18}"
        f"{'No-contact Mean':>20}"
        f"{'Genuine Min':>18}"
        f"{'No-contact Max':>20}"
    )

    print("-" * 125)

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user_id in USERS:

            rows = [
                r
                for r in results
                if (
                    r["registration_count"]
                    == registration_count
                    and
                    r["user"]
                    == user_id
                )
            ]

            registration_mean = np.mean([
                r["registration_mean"]
                for r in rows
            ])

            genuine_mean = np.mean([
                r["genuine_mean_score"]
                for r in rows
            ])

            no_contact_mean = np.mean([
                r["no_contact_mean_score"]
                for r in rows
            ])

            genuine_min = np.mean([
                r["genuine_min_score"]
                for r in rows
            ])

            no_contact_max = np.mean([
                r["no_contact_max_score"]
                for r in rows
            ])

            print(
                f"{registration_count:<6}"
                f"{user_id:<12}"
                f"{registration_mean:>20.4f}"
                f"{genuine_mean:>18.4f}"
                f"{no_contact_mean:>20.4f}"
                f"{genuine_min:>18.4f}"
                f"{no_contact_max:>20.4f}"
            )


# ============================================================
# Overall Summary
# ============================================================

def print_overall_summary(
    results: list[dict],
) -> None:

    contact_rates = np.asarray([
        r["genuine_contact_rate"]
        for r in results
    ])

    miss_rates = np.asarray([
        r["genuine_miss_rate"]
        for r in results
    ])

    no_contact_fars = np.asarray([
        r["no_contact_far"]
        for r in results
    ])

    no_contact_rejections = np.asarray([
        r["no_contact_rejection_rate"]
        for r in results
    ])

    print()
    print("=" * 100)
    print(
        "OVERALL CONTACT GATE PERFORMANCE"
    )
    print("=" * 100)

    print(
        f"Average Genuine Contact Rate : "
        f"{np.mean(contact_rates) * 100:.2f}%"
    )

    print(
        f"Average Genuine Miss Rate    : "
        f"{np.mean(miss_rates) * 100:.2f}%"
    )

    print(
        f"Average No-contact FAR       : "
        f"{np.mean(no_contact_fars) * 100:.2f}%"
    )

    print(
        f"Average No-contact Reject    : "
        f"{np.mean(no_contact_rejections) * 100:.2f}%"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  Genuine Contact Rate "
        "-> real grip successfully detected"
    )

    print(
        "  Genuine Miss Rate "
        "-> real grip incorrectly treated as no-contact"
    )

    print(
        "  No-contact FAR "
        "-> no-contact incorrectly treated as contact"
    )

    print(
        "  No-contact Reject "
        "-> no-contact correctly rejected"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    X, y = load_data()

    print()
    print("=" * 100)
    print(
        "RAW 50x16 CONTACT GATE BENCHMARK"
    )
    print("=" * 100)

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
        f"Registration counts : {REGISTRATION_COUNTS}"
    )

    print(
        f"Unknown test count  : {TEST_COUNT}"
    )

    print(
        f"Repeats             : {REPEATS}"
    )

    print(
        f"Contact K           : {CONTACT_K}"
    )

    print()
    print(
        "Contact score:"
    )

    print(
        "  mean(frame pressure_sum)"
    )

    print()
    print(
        "Threshold:"
    )

    print(
        "  registration mean - "
        f"{CONTACT_K} * registration std"
    )

    results = run_benchmark(
        X,
        y,
    )

    print_detailed_results(
        results
    )

    print_summary(
        results
    )

    print_score_summary(
        results
    )

    print_overall_summary(
        results
    )

    print()
    print("=" * 100)
    print(
        "Contact Gate Benchmark Finished"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()