from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# Contact Gate
# ============================================================

@dataclass
class ContactGate:
    """
    One calibrated Contact Gate.

    score >= threshold
        -> CONTACT

    score < threshold
        -> NO-CONTACT
    """

    user_id: str
    registration_count: int
    repeat: int

    threshold: float

    genuine_mean: float
    genuine_std: float

    no_contact_mean: float
    no_contact_std: float

    calibration_balanced_accuracy: float
    calibration_youden_j: float


# ============================================================
# Contact Score
# ============================================================

def calculate_contact_score(
    sample: np.ndarray,
) -> float:
    """
    Raw pressure contact score.

    Input:
        sample: (50, 16)

    Score:
        maximum frame-level pressure sum
    """

    sample = np.asarray(
        sample,
        dtype=np.float32,
    )

    if sample.ndim != 2:
        raise ValueError(
            f"Expected (frames, sensors), "
            f"got {sample.shape}"
        )

    if sample.shape[1] != 16:
        raise ValueError(
            f"Expected 16 sensors, "
            f"got {sample.shape[1]}"
        )

    pressure_sum = np.sum(
        sample,
        axis=1,
    )

    return float(
        np.max(pressure_sum)
    )


def calculate_contact_scores(
    X: np.ndarray,
) -> np.ndarray:

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 3:
        raise ValueError(
            f"Expected (samples, frames, sensors), "
            f"got {X.shape}"
        )

    return np.asarray(
        [
            calculate_contact_score(sample)
            for sample in X
        ],
        dtype=np.float64,
    )


# ============================================================
# Threshold Search
# ============================================================

def find_best_threshold(
    genuine_scores: np.ndarray,
    no_contact_scores: np.ndarray,
):
    """
    Find threshold using calibration data ONLY.

    Decision:

        score >= threshold
            -> CONTACT

        score < threshold
            -> NO-CONTACT

    Objective:
        maximize Balanced Accuracy

    This is calibration only.
    Formal test samples must NOT enter here.
    """

    genuine_scores = np.asarray(
        genuine_scores,
        dtype=np.float64,
    )

    no_contact_scores = np.asarray(
        no_contact_scores,
        dtype=np.float64,
    )

    all_scores = np.concatenate(
        [
            genuine_scores,
            no_contact_scores,
        ]
    )

    thresholds = np.unique(
        all_scores
    )

    best_threshold = None
    best_balanced_accuracy = -np.inf
    best_youden_j = -np.inf

    for threshold in thresholds:

        # ----------------------------------------------------
        # Genuine
        # ----------------------------------------------------

        genuine_contact = (
            genuine_scores >= threshold
        )

        tpr = float(
            np.mean(
                genuine_contact
            )
        )

        # ----------------------------------------------------
        # No-contact
        # ----------------------------------------------------

        no_contact_contact = (
            no_contact_scores >= threshold
        )

        fpr = float(
            np.mean(
                no_contact_contact
            )
        )

        tnr = 1.0 - fpr

        balanced_accuracy = (
            tpr + tnr
        ) / 2.0

        youden_j = (
            tpr - fpr
        )

        # ----------------------------------------------------
        # Best threshold
        # ----------------------------------------------------

        if (
            balanced_accuracy
            > best_balanced_accuracy
        ):

            best_balanced_accuracy = (
                balanced_accuracy
            )

            best_youden_j = youden_j

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_balanced_accuracy,
        best_youden_j,
    )


# ============================================================
# Calibrate
# ============================================================

def calibrate_contact_gate(
    user_id: str,
    registration_count: int,
    repeat: int,
    registration_samples: np.ndarray,
    background_calibration_samples: np.ndarray,
) -> ContactGate:
    """
    Build one user's Contact Gate.

    Genuine calibration data:
        registration_samples

    No-contact calibration data:
        background_calibration_samples

    IMPORTANT:
        Both datasets must be calibration data only.
    """

    genuine_scores = (
        calculate_contact_scores(
            registration_samples
        )
    )

    no_contact_scores = (
        calculate_contact_scores(
            background_calibration_samples
        )
    )

    (
        threshold,
        balanced_accuracy,
        youden_j,
    ) = find_best_threshold(
        genuine_scores,
        no_contact_scores,
    )

    return ContactGate(
        user_id=user_id,
        registration_count=registration_count,
        repeat=repeat,

        threshold=threshold,

        genuine_mean=float(
            np.mean(genuine_scores)
        ),

        genuine_std=float(
            np.std(genuine_scores)
        ),

        no_contact_mean=float(
            np.mean(no_contact_scores)
        ),

        no_contact_std=float(
            np.std(no_contact_scores)
        ),

        calibration_balanced_accuracy=(
            balanced_accuracy
        ),

        calibration_youden_j=(
            youden_j
        ),
    )


# ============================================================
# Prediction
# ============================================================

def contact_gate_predict(
    gate: ContactGate,
    sample: np.ndarray,
):
    """
    Return:

        score
        is_contact
    """

    score = calculate_contact_score(
        sample
    )

    is_contact = (
        score >= gate.threshold
    )

    return (
        score,
        bool(is_contact),
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_contact_gate(
    gate: ContactGate,
    genuine_test: np.ndarray,
    no_contact_test: np.ndarray,
):
    """
    Evaluate Contact Gate on completely unseen
    formal test data.
    """

    genuine_scores = (
        calculate_contact_scores(
            genuine_test
        )
    )

    no_contact_scores = (
        calculate_contact_scores(
            no_contact_test
        )
    )

    genuine_contact = (
        genuine_scores >= gate.threshold
    )

    no_contact_contact = (
        no_contact_scores >= gate.threshold
    )

    contact_rate = float(
        np.mean(
            genuine_contact
        )
    )

    genuine_miss_rate = (
        1.0 - contact_rate
    )

    no_contact_far = float(
        np.mean(
            no_contact_contact
        )
    )

    no_contact_reject = (
        1.0 - no_contact_far
    )

    balanced_accuracy = (
        contact_rate
        + no_contact_reject
    ) / 2.0

    return {
        "contact_rate":
            contact_rate,

        "genuine_miss_rate":
            genuine_miss_rate,

        "no_contact_far":
            no_contact_far,

        "no_contact_reject":
            no_contact_reject,

        "balanced_accuracy":
            balanced_accuracy,

        "genuine_mean":
            float(
                np.mean(genuine_scores)
            ),

        "no_contact_mean":
            float(
                np.mean(no_contact_scores)
            ),

        "genuine_min":
            float(
                np.min(genuine_scores)
            ),

        "no_contact_max":
            float(
                np.max(no_contact_scores)
            ),
    }


# ============================================================
# Print Gate
# ============================================================

def print_gate(
    gate: ContactGate,
):
    print(
        f"{gate.user_id:<10}"
        f"{gate.registration_count:>5}"
        f"{gate.repeat:>7}"
        f"{gate.threshold:>16.4f}"
        f"{gate.genuine_mean:>18.4f}"
        f"{gate.no_contact_mean:>20.4f}"
        f"{gate.calibration_balanced_accuracy * 100:>18.2f}%"
    )