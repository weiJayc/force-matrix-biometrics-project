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
# Imports
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
# Contact Score
# ============================================================

def calculate_contact_score(
    sample: np.ndarray,
) -> float:
    """
    Calculate one scalar contact score from a complete
    50 x 16 pressure sequence.

    score = sum of all pressure values

    Higher score -> stronger contact
    Lower score  -> weaker / no contact
    """

    sample = np.asarray(
        sample,
        dtype=np.float64,
    )

    if sample.ndim != 2:
        raise ValueError(
            f"Expected sample shape (frames, sensors), "
            f"got {sample.shape}"
        )

    return float(np.sum(sample))


# ============================================================
# Load Data
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:

    X, y = load_dataset(DATA_DIR)

    X = np.asarray(X)

    y = np.asarray(y)

    print(
        f"Dataset shape : {X.shape}"
    )

    print(
        f"Users         : {USERS}"
    )

    print(
        f"No-contact    : {NO_CONTACT_USER}"
    )

    return X, y


# ============================================================
# Get User Samples
# ============================================================

def get_user_samples(
    X: np.ndarray,
    y: np.ndarray,
    user_id: str,
) -> np.ndarray:

    mask = y == user_id

    samples = X[mask]

    if len(samples) == 0:
        raise ValueError(
            f"No samples found for user: {user_id}"
        )

    return samples


# ============================================================
# Calculate Scores
# ============================================================

def calculate_scores(
    samples: np.ndarray,
) -> np.ndarray:

    return np.asarray(
        [
            calculate_contact_score(sample)
            for sample in samples
        ],
        dtype=np.float64,
    )


# ============================================================
# Evaluate Threshold
# ============================================================

def evaluate_threshold(
    threshold: float,
    genuine_scores: np.ndarray,
    no_contact_scores: np.ndarray,
) -> dict[str, float]:

    # --------------------------------------------------------
    # Genuine contact
    #
    # score >= threshold -> contact
    # --------------------------------------------------------

    genuine_contact = (
        genuine_scores >= threshold
    )

    genuine_contact_rate = float(
        np.mean(genuine_contact)
    )

    genuine_miss_rate = (
        1.0 - genuine_contact_rate
    )

    # --------------------------------------------------------
    # No-contact
    #
    # score >= threshold -> incorrectly regarded as contact
    # --------------------------------------------------------

    no_contact_as_contact = (
        no_contact_scores >= threshold
    )

    no_contact_far = float(
        np.mean(no_contact_as_contact)
    )

    no_contact_reject = (
        1.0 - no_contact_far
    )

    # --------------------------------------------------------
    # Balanced Accuracy
    #
    # TPR = genuine contact rate
    # TNR = no-contact reject rate
    # --------------------------------------------------------

    balanced_accuracy = (
        genuine_contact_rate
        + no_contact_reject
    ) / 2.0

    # --------------------------------------------------------
    # Youden's J
    # --------------------------------------------------------

    youden_j = (
        genuine_contact_rate
        + no_contact_reject
        - 1.0
    )

    return {
        "threshold": float(threshold),

        "contact_rate": genuine_contact_rate,

        "miss_rate": genuine_miss_rate,

        "no_contact_far": no_contact_far,

        "no_contact_reject": no_contact_reject,

        "balanced_accuracy": balanced_accuracy,

        "youden_j": youden_j,
    }


# ============================================================
# Find Best Threshold
# ============================================================

def find_best_threshold(
    genuine_scores: np.ndarray,
    no_contact_scores: np.ndarray,
) -> dict[str, float]:

    all_scores = np.concatenate(
        [
            genuine_scores,
            no_contact_scores,
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Do not print every threshold.
    #
    # We still evaluate every possible threshold.
    # --------------------------------------------------------

    thresholds = np.unique(
        all_scores
    )

    # Add boundaries
    thresholds = np.concatenate(
        [
            [float(np.min(thresholds)) - 1.0],
            thresholds,
            [float(np.max(thresholds)) + 1.0],
        ]
    )

    best_result = None

    for threshold in thresholds:

        result = evaluate_threshold(
            threshold,
            genuine_scores,
            no_contact_scores,
        )

        if (
            best_result is None
            or
            result["youden_j"]
            > best_result["youden_j"]
        ):
            best_result = result

    return best_result


# ============================================================
# Find Threshold Under Target FAR
# ============================================================

def find_threshold_for_target_far(
    genuine_scores: np.ndarray,
    no_contact_scores: np.ndarray,
    target_far: float,
) -> dict[str, float] | None:

    all_scores = np.concatenate(
        [
            genuine_scores,
            no_contact_scores,
        ]
    )

    thresholds = np.unique(
        all_scores
    )

    candidates = []

    for threshold in thresholds:

        result = evaluate_threshold(
            threshold,
            genuine_scores,
            no_contact_scores,
        )

        if (
            result["no_contact_far"]
            <= target_far
        ):
            candidates.append(result)

    if not candidates:
        return None

    # Among thresholds satisfying FAR target,
    # choose the one with highest genuine contact rate.

    candidates.sort(
        key=lambda x: (
            x["contact_rate"],
            -x["threshold"],
        ),
        reverse=True,
    )

    return candidates[0]


# ============================================================
# Print One Result
# ============================================================

def print_result(
    label: str,
    result: dict[str, float] | None,
) -> None:

    if result is None:

        print(
            f"{label:<22} No threshold satisfies target"
        )

        return

    print(
        f"{label:<22}"
        f"{result['threshold']:>14.4f}"
        f"{result['contact_rate'] * 100:>14.2f}%"
        f"{result['miss_rate'] * 100:>14.2f}%"
        f"{result['no_contact_far'] * 100:>16.2f}%"
        f"{result['no_contact_reject'] * 100:>18.2f}%"
        f"{result['balanced_accuracy'] * 100:>18.2f}%"
    )


# ============================================================
# Global Threshold Test
# ============================================================

def run_global_test(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    print()
    print("=" * 110)
    print(
        "GLOBAL CONTACT GATE THRESHOLD ANALYSIS"
    )
    print("=" * 110)

    # --------------------------------------------------------
    # Genuine = all real users
    # No-contact = background
    # --------------------------------------------------------

    genuine_samples = np.concatenate(
        [
            get_user_samples(
                X,
                y,
                user,
            )
            for user in USERS
        ],
        axis=0,
    )

    no_contact_samples = get_user_samples(
        X,
        y,
        NO_CONTACT_USER,
    )

    genuine_scores = calculate_scores(
        genuine_samples
    )

    no_contact_scores = calculate_scores(
        no_contact_samples
    )

    print()
    print(
        "Score statistics"
    )

    print("-" * 110)

    print(
        f"Genuine      : "
        f"mean={np.mean(genuine_scores):.4f}, "
        f"std={np.std(genuine_scores):.4f}, "
        f"min={np.min(genuine_scores):.4f}, "
        f"max={np.max(genuine_scores):.4f}"
    )

    print(
        f"No-contact   : "
        f"mean={np.mean(no_contact_scores):.4f}, "
        f"std={np.std(no_contact_scores):.4f}, "
        f"min={np.min(no_contact_scores):.4f}, "
        f"max={np.max(no_contact_scores):.4f}"
    )

    # --------------------------------------------------------
    # Best threshold
    # --------------------------------------------------------

    best = find_best_threshold(
        genuine_scores,
        no_contact_scores,
    )

    print()
    print(
        "BEST THRESHOLD"
    )

    print("-" * 110)

    print(
        f"Threshold             : "
        f"{best['threshold']:.4f}"
    )

    print(
        f"Contact Rate          : "
        f"{best['contact_rate'] * 100:.2f}%"
    )

    print(
        f"Genuine Miss Rate     : "
        f"{best['miss_rate'] * 100:.2f}%"
    )

    print(
        f"No-contact FAR        : "
        f"{best['no_contact_far'] * 100:.2f}%"
    )

    print(
        f"No-contact Reject     : "
        f"{best['no_contact_reject'] * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy     : "
        f"{best['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"Youden J              : "
        f"{best['youden_j']:.4f}"
    )

    # --------------------------------------------------------
    # Target FAR operating points
    # --------------------------------------------------------

    print()
    print(
        "TARGET NO-CONTACT FAR OPERATING POINTS"
    )

    print("-" * 110)

    print(
        f"{'Target':<22}"
        f"{'Threshold':>14}"
        f"{'Contact Rate':>14}"
        f"{'Miss Rate':>14}"
        f"{'No-contact FAR':>16}"
        f"{'No-contact Reject':>18}"
        f"{'Balanced Acc.':>18}"
    )

    print("-" * 110)

    for target_far in (
        0.01,
        0.05,
        0.10,
        0.20,
    ):

        result = find_threshold_for_target_far(
            genuine_scores,
            no_contact_scores,
            target_far,
        )

        if result is None:

            print(
                f"FAR <= {target_far * 100:.0f}%"
                f"{'NO SOLUTION':>88}"
            )

        else:

            print(
                f"FAR <= {target_far * 100:.0f}%"
                f"{result['threshold']:>14.4f}"
                f"{result['contact_rate'] * 100:>14.2f}%"
                f"{result['miss_rate'] * 100:>14.2f}%"
                f"{result['no_contact_far'] * 100:>16.2f}%"
                f"{result['no_contact_reject'] * 100:>18.2f}%"
                f"{result['balanced_accuracy'] * 100:>18.2f}%"
            )


# ============================================================
# Per-user Threshold Analysis
# ============================================================

def run_per_user_test(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    print()
    print("=" * 110)
    print(
        "PER-USER CONTACT GATE THRESHOLD ANALYSIS"
    )
    print("=" * 110)

    no_contact_samples = get_user_samples(
        X,
        y,
        NO_CONTACT_USER,
    )

    no_contact_scores = calculate_scores(
        no_contact_samples
    )

    print()
    print(
        f"{'User':<12}"
        f"{'Threshold':>14}"
        f"{'Contact Rate':>15}"
        f"{'Miss Rate':>15}"
        f"{'No-contact FAR':>18}"
        f"{'No-contact Reject':>20}"
        f"{'Balanced Acc.':>18}"
    )

    print("-" * 110)

    for user in USERS:

        genuine_samples = get_user_samples(
            X,
            y,
            user,
        )

        genuine_scores = calculate_scores(
            genuine_samples
        )

        best = find_best_threshold(
            genuine_scores,
            no_contact_scores,
        )

        print(
            f"{user:<12}"
            f"{best['threshold']:>14.4f}"
            f"{best['contact_rate'] * 100:>14.2f}%"
            f"{best['miss_rate'] * 100:>14.2f}%"
            f"{best['no_contact_far'] * 100:>17.2f}%"
            f"{best['no_contact_reject'] * 100:>19.2f}%"
            f"{best['balanced_accuracy'] * 100:>17.2f}%"
        )


# ============================================================
# Registration Size Analysis
# ============================================================

def run_registration_size_analysis(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    """
    This section checks whether registration size changes
    the contact-score distribution.

    IMPORTANT:
    The contact gate itself does NOT require a template.

    Therefore registration_count is only used to report
    how many samples would normally be available during
    registration. The actual contact score distribution is
    independent from identity registration.
    """

    print()
    print("=" * 110)
    print(
        "REGISTRATION SIZE INFORMATION"
    )
    print("=" * 110)

    print()
    print(
        "Contact Gate uses raw pressure score only."
    )

    print(
        "Therefore the threshold itself does not need "
        "5/10/20 registration samples."
    )

    print()

    for count in REGISTRATION_COUNTS:

        print(
            f"Registration = {count:>2} samples"
            f" -> Contact Gate threshold is independent "
            f"of identity template."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 110)
    print(
        "CONTACT GATE THRESHOLD SWEEP"
    )
    print("=" * 110)

    print()
    print(
        "Purpose:"
    )

    print(
        "  Determine whether a physical grip/contact exists "
        "before identity authentication."
    )

    print()
    print(
        "Contact decision:"
    )

    print(
        "  score >= threshold -> CONTACT"
    )

    print(
        "  score <  threshold -> NO-CONTACT"
    )

    print()
    print(
        "Contact score:"
    )

    print(
        "  sum of all 50 x 16 raw pressure values"
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    run_global_test(
        X,
        y,
    )

    # --------------------------------------------------------
    # Per user
    # --------------------------------------------------------

    run_per_user_test(
        X,
        y,
    )

    # --------------------------------------------------------
    # Registration information
    # --------------------------------------------------------

    run_registration_size_analysis(
        X,
        y,
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "Contact Gate Threshold Sweep Finished"
    )
    print("=" * 110)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()