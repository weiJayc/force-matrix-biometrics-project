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


# Registration:
#
#   5 samples
#   10 samples
#   20 samples
#
REGISTRATION_COUNTS = (
    5,
    10,
    20,
)


# Formal test samples per user
TEST_COUNT = 30


# Number of repeated experiments
REPEATS = 20


# Reproducible random seed
RANDOM_SEED = 42


# ============================================================
# Background Split
# ============================================================

# Background has:
#
#   20 samples -> Contact Gate calibration
#   30 samples -> formal no-contact test
#
BACKGROUND_CALIBRATION_COUNT = 20


# ============================================================
# DTW Configuration
# ============================================================

# Sakoe-Chiba warping window
DTW_WINDOW = 5


# DTW threshold:
#
# threshold =
#       mean(registration scores)
#       +
#       K * std(registration scores)
#
DTW_THRESHOLD_K = 2.0


# ============================================================
# Dataset
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the original dataset.

    Expected shape:

        X = (samples, 50, 16)

    Labels:

        amber
        jay
        666
        background
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
            f"Expected every sample to be "
            f"(50, 16), got {X.shape[1:]}"
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
# DTW
# ============================================================

def dtw_distance(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
    window: int = DTW_WINDOW,
) -> float:
    """
    Multivariate DTW.

    Input:

        sequence_a: (T, 16)
        sequence_b: (T, 16)

    Each DTW cell uses Euclidean distance
    between two 16-dimensional pressure frames.

    Sakoe-Chiba window limits the warping.
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
            f"DTW expects 2D sequences, "
            f"got {a.shape} and {b.shape}"
        )

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dimensions do not match: "
            f"{a.shape[1]} vs {b.shape[1]}"
        )

    n = a.shape[0]
    m = b.shape[0]

    window = max(
        window,
        abs(n - m),
    )

    cost = np.full(
        (n + 1, m + 1),
        np.inf,
        dtype=np.float64,
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

        for j in range(
            start_j,
            end_j + 1,
        ):

            frame_distance = np.linalg.norm(
                a[i - 1] - b[j - 1]
            )

            cost[i, j] = (
                frame_distance
                +
                min(
                    cost[i - 1, j],
                    cost[i, j - 1],
                    cost[i - 1, j - 1],
                )
            )

    return float(cost[n, m])


# ============================================================
# Split Helpers
# ============================================================

def split_registration_test(
    samples: np.ndarray,
    registration_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split one user's original dataset into:

        registration_count
        +
        TEST_COUNT

    Example:

        Reg = 5

        5 registration
        30 formal test

    No overlap.
    """

    required = (
        registration_count
        + TEST_COUNT
    )

    if len(samples) < required:
        raise ValueError(
            f"Need {required} samples, "
            f"but only {len(samples)} available."
        )

    indices = rng.permutation(
        len(samples)
    )

    registration_indices = (
        indices[
            :registration_count
        ]
    )

    test_indices = (
        indices[
            registration_count:
            registration_count + TEST_COUNT
        ]
    )

    return (
        samples[registration_indices],
        samples[test_indices],
    )


def split_background(
    samples: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Background split:

        20 -> Contact Gate calibration

        30 -> unseen formal no-contact test
    """

    required = (
        BACKGROUND_CALIBRATION_COUNT
        + TEST_COUNT
    )

    if len(samples) < required:
        raise ValueError(
            f"Background needs at least "
            f"{required} samples, "
            f"but only {len(samples)} available."
        )

    indices = rng.permutation(
        len(samples)
    )

    calibration_indices = (
        indices[
            :BACKGROUND_CALIBRATION_COUNT
        ]
    )

    test_indices = (
        indices[
            BACKGROUND_CALIBRATION_COUNT:
            BACKGROUND_CALIBRATION_COUNT
            + TEST_COUNT
        ]
    )

    return (
        samples[calibration_indices],
        samples[test_indices],
    )


# ============================================================
# Contact Gate
# ============================================================

def contact_decision(
    gate,
    sample: np.ndarray,
) -> tuple[float, bool]:
    """
    Apply calibrated Contact Gate.

    score >= threshold
        -> CONTACT

    score < threshold
        -> NO-CONTACT
    """

    score = float(
        calculate_contact_score(sample)
    )

    contact = (
        score >= float(gate.threshold)
    )

    return (
        score,
        contact,
    )


def calibrate_gate(
    template_user: str,
    registration_count: int,
    repeat: int,
    registration_samples: np.ndarray,
    background_calibration: np.ndarray,
):
    """
    Calibrate Contact Gate using:

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


# ============================================================
# DTW Template
# ============================================================

def build_dtw_template(
    registration_samples: np.ndarray,
) -> dict:
    """
    Build DTW identity template.

    The registration sequences themselves
    are retained.

    Authentication:

        sample
           ↓
        DTW against every registration sample
           ↓
        minimum distance

    Threshold calibration uses the same scoring rule:

        registration sample
           ↓
        DTW against all OTHER registration samples
           ↓
        minimum distance
    """

    registration_samples = np.asarray(
        registration_samples,
        dtype=np.float32,
    )

    n = len(
        registration_samples
    )

    if n < 2:
        raise ValueError(
            "At least 2 registration samples "
            "are required for DTW threshold "
            "calibration."
        )

    registration_scores = []

    for i in range(n):

        distances = []

        for j in range(n):

            if i == j:
                continue

            distance = dtw_distance(
                registration_samples[i],
                registration_samples[j],
            )

            distances.append(
                distance
            )

        registration_scores.append(
            np.min(distances)
        )

    registration_scores = np.asarray(
        registration_scores,
        dtype=np.float64,
    )

    threshold = (
        float(
            np.mean(
                registration_scores
            )
        )
        +
        DTW_THRESHOLD_K
        *
        float(
            np.std(
                registration_scores
            )
        )
    )

    return {
        "registration_samples":
            registration_samples,

        "registration_scores":
            registration_scores,

        "threshold":
            threshold,
    }


# ============================================================
# DTW Authentication
# ============================================================

def authenticate_dtw(
    template: dict,
    sample: np.ndarray,
) -> tuple[float, bool]:
    """
    Authenticate one sample.

    Score:

        minimum DTW distance

    Decision:

        distance <= threshold
            -> ACCEPT

        distance > threshold
            -> REJECT
    """

    registration_samples = (
        template[
            "registration_samples"
        ]
    )

    distances = []

    for registration_sample in (
        registration_samples
    ):

        distance = dtw_distance(
            registration_sample,
            sample,
        )

        distances.append(
            distance
        )

    distance = float(
        np.min(distances)
    )

    accepted = (
        distance
        <=
        template["threshold"]
    )

    return (
        distance,
        accepted,
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
    Evaluate Contact Gate independently.

    Genuine:

        CONTACT = correct

    Background:

        NO-CONTACT = correct
    """

    genuine_scores = []
    no_contact_scores = []

    genuine_contact_count = 0
    no_contact_contact_count = 0

    # --------------------------------------------------------
    # Genuine
    # --------------------------------------------------------

    for sample in genuine_test:

        score, contact = contact_decision(
            gate,
            sample,
        )

        genuine_scores.append(
            score
        )

        if contact:
            genuine_contact_count += 1

    # --------------------------------------------------------
    # No-contact
    # --------------------------------------------------------

    for sample in no_contact_test:

        score, contact = contact_decision(
            gate,
            sample,
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
        / genuine_total
        if genuine_total > 0
        else 0.0
    )

    genuine_miss_rate = (
        1.0
        - contact_rate
    )

    no_contact_far = (
        no_contact_contact_count
        / no_contact_total
        if no_contact_total > 0
        else 0.0
    )

    no_contact_reject = (
        1.0
        - no_contact_far
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
# DTW Standalone
# ============================================================

def evaluate_dtw_standalone(
    template: dict,
    template_user: str,
    test_samples_by_user: dict,
    no_contact_test: np.ndarray,
) -> dict:
    """
    DTW-only benchmark.

    IMPORTANT:

    This does NOT use Contact Gate.

    Every formal test sample goes directly
    into DTW.

    This allows us to compare:

        DTW standalone

    against:

        Contact Gate -> DTW
    """

    threshold = float(
        template["threshold"]
    )

    # --------------------------------------------------------
    # Genuine
    # --------------------------------------------------------

    genuine_test = (
        test_samples_by_user[
            template_user
        ]
    )

    genuine_accept_count = 0
    genuine_distances = []

    for sample in genuine_test:

        distance, accepted = (
            authenticate_dtw(
                template,
                sample,
            )
        )

        genuine_distances.append(
            distance
        )

        if accepted:
            genuine_accept_count += 1

    genuine_total = len(
        genuine_test
    )

    gar = (
        genuine_accept_count
        / genuine_total
        if genuine_total > 0
        else 0.0
    )

    frr = (
        1.0
        - gar
    )

    # --------------------------------------------------------
    # Impostors
    # --------------------------------------------------------

    impostor_accept_count = 0
    impostor_total = 0

    impostor_distances = []

    for impostor_user in USERS:

        if impostor_user == template_user:
            continue

        impostor_test = (
            test_samples_by_user[
                impostor_user
            ]
        )

        for sample in impostor_test:

            impostor_total += 1

            distance, accepted = (
                authenticate_dtw(
                    template,
                    sample,
                )
            )

            impostor_distances.append(
                distance
            )

            if accepted:
                impostor_accept_count += 1

    far = (
        impostor_accept_count
        / impostor_total
        if impostor_total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # No-contact
    # --------------------------------------------------------

    no_contact_accept_count = 0

    no_contact_distances = []

    for sample in no_contact_test:

        distance, accepted = (
            authenticate_dtw(
                template,
                sample,
            )
        )

        no_contact_distances.append(
            distance
        )

        if accepted:
            no_contact_accept_count += 1

    no_contact_total = len(
        no_contact_test
    )

    no_contact_far = (
        no_contact_accept_count
        / no_contact_total
        if no_contact_total > 0
        else 0.0
    )

    return {
        "gar":
            gar,

        "frr":
            frr,

        "far":
            far,

        "no_contact_far":
            no_contact_far,

        "threshold":
            threshold,

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

        "no_contact_distance":
            (
                float(
                    np.mean(
                        no_contact_distances
                    )
                )
                if no_contact_distances
                else np.nan
            ),
    }


# ============================================================
# E2E Benchmark
# ============================================================

def evaluate_e2e(
    gate,
    dtw_template: dict,
    template_user: str,
    test_samples_by_user: dict,
    no_contact_test: np.ndarray,
) -> dict:
    """
    Complete two-stage authentication.

    Pipeline:

        Input
          ↓
        Contact Gate
          ↓
        if NO-CONTACT
            -> Reject
        if CONTACT
            ↓
        DTW
            ↓
        Accept / Reject

    IMPORTANT:

    FAR denominator includes ALL impostor
    test samples.

    Samples rejected by Contact Gate are
    therefore counted as correct rejections.
    """

    # ========================================================
    # Genuine
    # ========================================================

    genuine_test = (
        test_samples_by_user[
            template_user
        ]
    )

    genuine_contact_count = 0
    genuine_accept_count = 0

    genuine_gate_miss_count = 0

    genuine_distances = []

    for sample in genuine_test:

        _, contact = contact_decision(
            gate,
            sample,
        )

        if not contact:

            genuine_gate_miss_count += 1

            continue

        genuine_contact_count += 1

        distance, accepted = (
            authenticate_dtw(
                dtw_template,
                sample,
            )
        )

        genuine_distances.append(
            distance
        )

        if accepted:
            genuine_accept_count += 1

    genuine_total = len(
        genuine_test
    )

    # Final E2E GAR:
    #
    # successful authentication
    # /
    # ALL genuine test samples
    #
    gar = (
        genuine_accept_count
        / genuine_total
        if genuine_total > 0
        else 0.0
    )

    frr = (
        1.0
        - gar
    )

    gate_contact_rate = (
        genuine_contact_count
        / genuine_total
        if genuine_total > 0
        else 0.0
    )

    gate_miss_rate = (
        1.0
        - gate_contact_rate
    )

    # ========================================================
    # Impostor
    # ========================================================

    impostor_total = 0
    impostor_contact_count = 0
    impostor_accept_count = 0

    impostor_distances = []

    impostor_details = {}

    for impostor_user in USERS:

        if impostor_user == template_user:
            continue

        impostor_test = (
            test_samples_by_user[
                impostor_user
            ]
        )

        contact_count = 0
        accept_count = 0

        distances = []

        for sample in impostor_test:

            # ------------------------------------------------
            # FAR denominator ALWAYS includes this sample
            # ------------------------------------------------

            impostor_total += 1

            _, contact = contact_decision(
                gate,
                sample,
            )

            if not contact:
                continue

            contact_count += 1
            impostor_contact_count += 1

            distance, accepted = (
                authenticate_dtw(
                    dtw_template,
                    sample,
                )
            )

            distances.append(
                distance
            )

            if accepted:

                accept_count += 1
                impostor_accept_count += 1

        impostor_details[
            impostor_user
        ] = {
            "contact":
                contact_count,

            "contact_rate":
                (
                    contact_count
                    / TEST_COUNT
                ),

            "accept":
                accept_count,

            "reject":
                TEST_COUNT
                - accept_count,

            "average_distance":
                (
                    float(
                        np.mean(distances)
                    )
                    if distances
                    else np.nan
                ),
        }

        impostor_distances.extend(
            distances
        )

    far = (
        impostor_accept_count
        / impostor_total
        if impostor_total > 0
        else 0.0
    )

    # ========================================================
    # No-contact
    # ========================================================

    no_contact_gate_contact_count = 0
    no_contact_e2e_accept_count = 0

    no_contact_distances = []

    for sample in no_contact_test:

        _, contact = contact_decision(
            gate,
            sample,
        )

        if not contact:
            continue

        no_contact_gate_contact_count += 1

        distance, accepted = (
            authenticate_dtw(
                dtw_template,
                sample,
            )
        )

        no_contact_distances.append(
            distance
        )

        if accepted:
            no_contact_e2e_accept_count += 1

    no_contact_total = len(
        no_contact_test
    )

    no_contact_far = (
        no_contact_e2e_accept_count
        / no_contact_total
        if no_contact_total > 0
        else 0.0
    )

    return {
        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        "gar":
            gar,

        "frr":
            frr,

        "far":
            far,

        # ----------------------------------------------------
        # Contact Gate
        # ----------------------------------------------------

        "gate_contact_rate":
            gate_contact_rate,

        "gate_miss_rate":
            gate_miss_rate,

        "gate_contact_count":
            genuine_contact_count,

        "gate_miss_count":
            genuine_gate_miss_count,

        # ----------------------------------------------------
        # No-contact
        # ----------------------------------------------------

        "no_contact_far":
            no_contact_far,

        "no_contact_gate_contact_count":
            no_contact_gate_contact_count,

        "no_contact_e2e_accept_count":
            no_contact_e2e_accept_count,

        # ----------------------------------------------------
        # Distances
        # ----------------------------------------------------

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

        "no_contact_distance":
            (
                float(
                    np.mean(
                        no_contact_distances
                    )
                )
                if no_contact_distances
                else np.nan
            ),

        # ----------------------------------------------------
        # Details
        # ----------------------------------------------------

        "impostor_details":
            impostor_details,
    }


# ============================================================
# Prepare Fair Dataset Split
# ============================================================

def prepare_repeat_split(
    X: np.ndarray,
    y: np.ndarray,
    registration_count: int,
    rng: np.random.Generator,
):
    """
    IMPORTANT:

    For ONE repeat:

        each user gets ONE registration/test split.

    Then:

        amber
        jay
        666

    take turns becoming the registered user.

    Therefore all three templates use the
    SAME underlying formal test sets.

    This makes DTW standalone and E2E
    directly comparable.
    """

    registration_samples_by_user = {}
    test_samples_by_user = {}

    # ========================================================
    # Users
    # ========================================================

    for user in USERS:

        samples = get_user_samples(
            X,
            y,
            user,
        )

        (
            registration_samples,
            test_samples,
        ) = split_registration_test(
            samples,
            registration_count,
            rng,
        )

        registration_samples_by_user[
            user
        ] = registration_samples

        test_samples_by_user[
            user
        ] = test_samples

    # ========================================================
    # Background
    # ========================================================

    background_samples = (
        get_user_samples(
            X,
            y,
            NO_CONTACT_USER,
        )
    )

    (
        background_calibration,
        no_contact_test,
    ) = split_background(
        background_samples,
        rng,
    )

    return (
        registration_samples_by_user,
        test_samples_by_user,
        background_calibration,
        no_contact_test,
    )


# ============================================================
# One Experiment
# ============================================================

def run_one_experiment(
    template_user: str,
    registration_count: int,
    repeat: int,
    registration_samples_by_user: dict,
    test_samples_by_user: dict,
    background_calibration: np.ndarray,
    no_contact_test: np.ndarray,
) -> dict:
    """
    Run:

        1. Contact Gate calibration
        2. Contact Gate standalone
        3. DTW template
        4. DTW standalone
        5. Complete E2E
    """

    registration_samples = (
        registration_samples_by_user[
            template_user
        ]
    )

    genuine_test = (
        test_samples_by_user[
            template_user
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
    # 2. Contact Gate Standalone
    # ========================================================

    gate_result = evaluate_contact_gate(
        gate=gate,
        genuine_test=genuine_test,
        no_contact_test=no_contact_test,
    )

    # ========================================================
    # 3. Build DTW Template
    # ========================================================

    dtw_template = build_dtw_template(
        registration_samples
    )

    # ========================================================
    # 4. DTW Standalone
    #
    # No Contact Gate here.
    #
    # Same exact formal test samples.
    # ========================================================

    dtw_result = evaluate_dtw_standalone(
        template=dtw_template,
        template_user=template_user,
        test_samples_by_user=test_samples_by_user,
        no_contact_test=no_contact_test,
    )

    # ========================================================
    # 5. Complete E2E
    #
    # Contact Gate
    #       ↓
    # DTW
    # ========================================================

    e2e_result = evaluate_e2e(
        gate=gate,
        dtw_template=dtw_template,
        template_user=template_user,
        test_samples_by_user=test_samples_by_user,
        no_contact_test=no_contact_test,
    )

    # ========================================================
    # Return
    # ========================================================

    return {

        # ----------------------------------------------------
        # Identification
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
                "genuine_miss_rate"
            ],

        "no_contact_gate_far":
            gate_result[
                "no_contact_far"
            ],

        "no_contact_gate_reject":
            gate_result[
                "no_contact_reject"
            ],

        # ----------------------------------------------------
        # DTW Standalone
        # ----------------------------------------------------

        "dtw_gar":
            dtw_result[
                "gar"
            ],

        "dtw_frr":
            dtw_result[
                "frr"
            ],

        "dtw_far":
            dtw_result[
                "far"
            ],

        "dtw_no_contact_far":
            dtw_result[
                "no_contact_far"
            ],

        "dtw_threshold":
            dtw_result[
                "threshold"
            ],

        "dtw_genuine_distance":
            dtw_result[
                "genuine_distance"
            ],

        "dtw_impostor_distance":
            dtw_result[
                "impostor_distance"
            ],

        "dtw_no_contact_distance":
            dtw_result[
                "no_contact_distance"
            ],

        # ----------------------------------------------------
        # E2E
        # ----------------------------------------------------

        "e2e_gar":
            e2e_result[
                "gar"
            ],

        "e2e_frr":
            e2e_result[
                "frr"
            ],

        "e2e_far":
            e2e_result[
                "far"
            ],

        "e2e_no_contact_far":
            e2e_result[
                "no_contact_far"
            ],

        "e2e_gate_contact_rate":
            e2e_result[
                "gate_contact_rate"
            ],

        "e2e_gate_miss_rate":
            e2e_result[
                "gate_miss_rate"
            ],

        "e2e_genuine_distance":
            e2e_result[
                "genuine_distance"
            ],

        "e2e_impostor_distance":
            e2e_result[
                "impostor_distance"
            ],

        "e2e_no_contact_distance":
            e2e_result[
                "no_contact_distance"
            ],

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        "dtw_registration_scores":
            dtw_template[
                "registration_scores"
            ],
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

    all_results = []

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        print()
        print(
            "=" * 110
        )

        print(
            f"DTW + Contact Gate "
            f"| Registration = "
            f"{registration_count} "
            f"| Test = {TEST_COUNT} "
            f"| Repeats = {REPEATS}"
        )

        print(
            "=" * 110
        )

        for repeat in range(
            1,
            REPEATS + 1,
        ):

            (
                registration_samples_by_user,
                test_samples_by_user,
                background_calibration,
                no_contact_test,
            ) = prepare_repeat_split(
                X=X,
                y=y,
                registration_count=
                    registration_count,
                rng=rng,
            )

            # ------------------------------------------------
            # Same split for all users
            # ------------------------------------------------

            for template_user in USERS:

                print(
                    f"Running "
                    f"Reg={registration_count}, "
                    f"Repeat={repeat:02d}, "
                    f"Template={template_user}"
                )

                result = run_one_experiment(
                    template_user=
                        template_user,

                    registration_count=
                        registration_count,

                    repeat=
                        repeat,

                    registration_samples_by_user=
                        registration_samples_by_user,

                    test_samples_by_user=
                        test_samples_by_user,

                    background_calibration=
                        background_calibration,

                    no_contact_test=
                        no_contact_test,
                )

                all_results.append(
                    result
                )

    return all_results


# ============================================================
# Utility
# ============================================================

def get_rows(
    results: list[dict],
    registration_count: int,
    user: str,
) -> list[dict]:

    return [
        row
        for row in results
        if (
            row["registration_count"]
            == registration_count
            and
            row["template_user"]
            == user
        )
    ]


def mean_metric(
    rows: list[dict],
    key: str,
) -> float:

    values = np.asarray(
        [
            row[key]
            for row in rows
        ],
        dtype=np.float64,
    )

    return float(
        np.mean(values)
    )


def std_metric(
    rows: list[dict],
    key: str,
) -> float:

    values = np.asarray(
        [
            row[key]
            for row in rows
        ],
        dtype=np.float64,
    )

    return float(
        np.std(values)
    )


# ============================================================
# Contact Gate Standalone Summary
# ============================================================

def print_contact_gate_summary(
    results: list[dict],
) -> None:

    print()
    print(
        "=" * 110
    )

    print(
        "CONTACT GATE STANDALONE SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'Contact Rate':>16}"
        f"{'Miss Rate':>14}"
        f"{'No-contact FAR':>18}"
        f"{'No-contact Reject':>20}"
        f"{'Threshold':>18}"
    )

    print(
        "-" * 110
    )

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = get_rows(
                results,
                registration_count,
                user,
            )

            contact_rate = mean_metric(
                rows,
                "contact_rate",
            )

            miss_rate = mean_metric(
                rows,
                "contact_miss_rate",
            )

            no_contact_far = mean_metric(
                rows,
                "no_contact_gate_far",
            )

            no_contact_reject = (
                1.0
                - no_contact_far
            )

            threshold = mean_metric(
                rows,
                "contact_threshold",
            )

            print(
                f"{registration_count:<6}"
                f"{user:<12}"
                f"{contact_rate * 100:>14.2f}%"
                f"{miss_rate * 100:>12.2f}%"
                f"{no_contact_far * 100:>16.2f}%"
                f"{no_contact_reject * 100:>18.2f}%"
                f"{threshold:>18.4f}"
            )


# ============================================================
# DTW Standalone Summary
# ============================================================

def print_dtw_standalone_summary(
    results: list[dict],
) -> None:

    print()
    print(
        "=" * 120
    )

    print(
        "DTW STANDALONE SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'GAR Mean':>14}"
        f"{'GAR Std':>14}"
        f"{'FRR Mean':>14}"
        f"{'FAR Mean':>14}"
        f"{'FAR Std':>14}"
        f"{'No-contact FAR':>18}"
        f"{'Threshold':>18}"
    )

    print(
        "-" * 120
    )

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = get_rows(
                results,
                registration_count,
                user,
            )

            gar_mean = mean_metric(
                rows,
                "dtw_gar",
            )

            gar_std = std_metric(
                rows,
                "dtw_gar",
            )

            frr_mean = mean_metric(
                rows,
                "dtw_frr",
            )

            far_mean = mean_metric(
                rows,
                "dtw_far",
            )

            far_std = std_metric(
                rows,
                "dtw_far",
            )

            no_contact_far = mean_metric(
                rows,
                "dtw_no_contact_far",
            )

            threshold = mean_metric(
                rows,
                "dtw_threshold",
            )

            print(
                f"{registration_count:<6}"
                f"{user:<12}"
                f"{gar_mean * 100:>12.2f}%"
                f"{gar_std * 100:>12.2f}%"
                f"{frr_mean * 100:>12.2f}%"
                f"{far_mean * 100:>12.2f}%"
                f"{far_std * 100:>12.2f}%"
                f"{no_contact_far * 100:>16.2f}%"
                f"{threshold:>18.4f}"
            )


# ============================================================
# DTW Distance Summary
# ============================================================

def print_distance_summary(
    results: list[dict],
) -> None:

    print()
    print(
        "=" * 110
    )

    print(
        "DTW DISTANCE SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'Genuine Distance':>22}"
        f"{'Impostor Distance':>22}"
        f"{'No-contact Distance':>22}"
        f"{'DTW Threshold':>20}"
    )

    print(
        "-" * 110
    )

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = get_rows(
                results,
                registration_count,
                user,
            )

            genuine_distance = mean_metric(
                rows,
                "dtw_genuine_distance",
            )

            impostor_distance = mean_metric(
                rows,
                "dtw_impostor_distance",
            )

            no_contact_distance = mean_metric(
                rows,
                "dtw_no_contact_distance",
            )

            threshold = mean_metric(
                rows,
                "dtw_threshold",
            )

            print(
                f"{registration_count:<6}"
                f"{user:<12}"
                f"{genuine_distance:>22.4f}"
                f"{impostor_distance:>22.4f}"
                f"{no_contact_distance:>22.4f}"
                f"{threshold:>20.4f}"
            )


# ============================================================
# E2E Summary
# ============================================================

def print_e2e_summary(
    results: list[dict],
) -> None:

    print()
    print(
        "=" * 120
    )

    print(
        "COMPLETE TWO-STAGE END-TO-END BENCHMARK"
    )

    print(
        "=" * 120
    )

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
        "          -> DTW Identity Authentication"
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
        f"{'Gate Contact':>16}"
        f"{'Gate Threshold':>18}"
        f"{'DTW Threshold':>18}"
    )

    print(
        "-" * 160
    )

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = get_rows(
                results,
                registration_count,
                user,
            )

            gar_mean = mean_metric(
                rows,
                "e2e_gar",
            )

            gar_std = std_metric(
                rows,
                "e2e_gar",
            )

            frr_mean = mean_metric(
                rows,
                "e2e_frr",
            )

            far_mean = mean_metric(
                rows,
                "e2e_far",
            )

            far_std = std_metric(
                rows,
                "e2e_far",
            )

            no_contact_far = mean_metric(
                rows,
                "e2e_no_contact_far",
            )

            gate_contact = mean_metric(
                rows,
                "e2e_gate_contact_rate",
            )

            gate_threshold = mean_metric(
                rows,
                "contact_threshold",
            )

            dtw_threshold = mean_metric(
                rows,
                "dtw_threshold",
            )

            print(
                f"{registration_count:<6}"
                f"{user:<12}"
                f"{gar_mean * 100:>12.2f}%"
                f"{gar_std * 100:>12.2f}%"
                f"{frr_mean * 100:>12.2f}%"
                f"{far_mean * 100:>12.2f}%"
                f"{far_std * 100:>12.2f}%"
                f"{no_contact_far * 100:>16.2f}%"
                f"{gate_contact * 100:>14.2f}%"
                f"{gate_threshold:>18.4f}"
                f"{dtw_threshold:>18.4f}"
            )


# ============================================================
# Standalone vs E2E Comparison
# ============================================================

def print_comparison(
    results: list[dict],
) -> None:

    print()
    print(
        "=" * 125
    )

    print(
        "STANDALONE vs END-TO-END COMPARISON"
    )

    print(
        "=" * 125
    )

    print(
        f"{'Reg':<6}"
        f"{'User':<12}"
        f"{'DTW GAR':>12}"
        f"{'E2E GAR':>12}"
        f"{'GAR Change':>14}"
        f"{'DTW FAR':>12}"
        f"{'E2E FAR':>12}"
        f"{'FAR Change':>14}"
        f"{'Gate Miss':>14}"
        f"{'NC FAR':>12}"
    )

    print(
        "-" * 125
    )

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for user in USERS:

            rows = get_rows(
                results,
                registration_count,
                user,
            )

            dtw_gar = mean_metric(
                rows,
                "dtw_gar",
            )

            e2e_gar = mean_metric(
                rows,
                "e2e_gar",
            )

            dtw_far = mean_metric(
                rows,
                "dtw_far",
            )

            e2e_far = mean_metric(
                rows,
                "e2e_far",
            )

            gate_miss = mean_metric(
                rows,
                "e2e_gate_miss_rate",
            )

            no_contact_far = mean_metric(
                rows,
                "e2e_no_contact_far",
            )

            gar_change = (
                e2e_gar
                - dtw_gar
            )

            far_change = (
                e2e_far
                - dtw_far
            )

            print(
                f"{registration_count:<6}"
                f"{user:<12}"
                f"{dtw_gar * 100:>10.2f}%"
                f"{e2e_gar * 100:>10.2f}%"
                f"{gar_change * 100:>12.2f}%"
                f"{dtw_far * 100:>10.2f}%"
                f"{e2e_far * 100:>10.2f}%"
                f"{far_change * 100:>12.2f}%"
                f"{gate_miss * 100:>12.2f}%"
                f"{no_contact_far * 100:>10.2f}%"
            )


# ============================================================
# Registration Effect
# ============================================================

def print_registration_effect(
    results: list[dict],
) -> None:

    print()
    print(
        "=" * 120
    )

    print(
        "DTW + CONTACT GATE REGISTRATION SIZE EFFECT"
    )

    print(
        "=" * 120
    )

    print(
        f"{'User':<12}"
        f"{'Reg':>8}"
        f"{'Contact Rate':>16}"
        f"{'GAR':>12}"
        f"{'FRR':>12}"
        f"{'FAR':>12}"
        f"{'No-contact FAR':>18}"
        f"{'Gate Threshold':>18}"
    )

    print(
        "-" * 120
    )

    for user in USERS:

        for registration_count in (
            REGISTRATION_COUNTS
        ):

            rows = get_rows(
                results,
                registration_count,
                user,
            )

            contact_rate = mean_metric(
                rows,
                "contact_rate",
            )

            gar = mean_metric(
                rows,
                "e2e_gar",
            )

            frr = mean_metric(
                rows,
                "e2e_frr",
            )

            far = mean_metric(
                rows,
                "e2e_far",
            )

            no_contact_far = mean_metric(
                rows,
                "e2e_no_contact_far",
            )

            threshold = mean_metric(
                rows,
                "contact_threshold",
            )

            print(
                f"{user:<12}"
                f"{registration_count:>8}"
                f"{contact_rate * 100:>14.2f}%"
                f"{gar * 100:>10.2f}%"
                f"{frr * 100:>10.2f}%"
                f"{far * 100:>10.2f}%"
                f"{no_contact_far * 100:>16.2f}%"
                f"{threshold:>18.4f}"
            )


# ============================================================
# Threshold Debug
# ============================================================

def print_threshold_debug(
    results: list[dict],
) -> None:

    print()
    print(
        "=" * 110
    )

    print(
        "CALIBRATED CONTACT GATE / DTW THRESHOLDS"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Reg':<8}"
        f"{'Repeat':<10}"
        f"{'User':<12}"
        f"{'Gate Threshold':>20}"
        f"{'DTW Threshold':>20}"
    )

    print(
        "-" * 110
    )

    for row in results:

        print(
            f"{row['registration_count']:<8}"
            f"{row['repeat']:<10}"
            f"{row['template_user']:<12}"
            f"{row['contact_threshold']:>20.4f}"
            f"{row['dtw_threshold']:>20.4f}"
        )


# ============================================================
# Dataset Information
# ============================================================

def print_dataset_information(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "DATASET INFORMATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Dataset path : {DATA_DIR}"
    )

    print(
        f"X shape      : {X.shape}"
    )

    print()

    labels, counts = np.unique(
        y,
        return_counts=True,
    )

    for label, count in zip(
        labels,
        counts,
    ):

        print(
            f"{str(label):<15}"
            f": {int(count)} samples"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 120
    )

    print(
        "DTW + CONTACT GATE FORMAL DOOR ACCESS BENCHMARK"
    )

    print(
        "=" * 120
    )

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
        "    DTW Identity Authentication"
    )

    print()

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
        f"Test count          : {TEST_COUNT}"
    )

    print(
        f"Repeats             : {REPEATS}"
    )

    print(
        f"Background calib    : "
        f"{BACKGROUND_CALIBRATION_COUNT}"
    )

    print(
        f"DTW window          : {DTW_WINDOW}"
    )

    print(
        f"DTW threshold       : "
        f"mean + {DTW_THRESHOLD_K:.1f} * std"
    )

    # ========================================================
    # Load Dataset
    # ========================================================

    X, y = load_data()

    print_dataset_information(
        X,
        y,
    )

    # ========================================================
    # Run Benchmark
    # ========================================================

    results = run_benchmark(
        X,
        y,
    )

    # ========================================================
    # Print Results
    # ========================================================

    print_contact_gate_summary(
        results
    )

    print_dtw_standalone_summary(
        results
    )

    print_distance_summary(
        results
    )

    print_e2e_summary(
        results
    )

    print_comparison(
        results
    )

    print_registration_effect(
        results
    )

    print_threshold_debug(
        results
    )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "Benchmark Finished"
    )

    print(
        "=" * 120
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()