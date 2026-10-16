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

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"

from data_loader import load_dataset


# ============================================================
# Configuration
# ============================================================

USERS = (
    "amber",
    "jay",
    "666",
    "background",
)

DEFAULT_REGISTRATION_COUNT = 20
DEFAULT_TEST_COUNT = 30
DEFAULT_REPEATS = 20

RANDOM_SEED = 42

# DTW threshold multiplier
THRESHOLD_K = 2.0


# ============================================================
# Frame-level Feature Extraction
# ============================================================

def extract_frame_features(
    sample: np.ndarray,
) -> np.ndarray:
    """
    將一段 50-frame × 16-sensor 的資料
    轉換成：

        (50, 7)

    每一個 frame 保留：

        0. Pressure Sum
        1. Max Pressure
        2. Contact Area
        3. COP X
        4. COP Y
        5. Left / Right Ratio
        6. Top / Bottom Ratio

    注意：
    這裡不再做 sequence-level statistics。

    DTW 必須保留時間軸。
    """

    sample = np.asarray(
        sample,
        dtype=np.float32,
    )

    if sample.ndim != 2:
        raise ValueError(
            f"Expected sample shape (frames, sensors), "
            f"got {sample.shape}"
        )

    if sample.shape[1] != 16:
        raise ValueError(
            f"Expected 16 sensors, "
            f"got {sample.shape[1]}"
        )

    # --------------------------------------------------------
    # Pressure Sum
    # --------------------------------------------------------

    pressure_sum = np.sum(
        sample,
        axis=1,
    )

    # --------------------------------------------------------
    # Maximum Pressure
    # --------------------------------------------------------

    max_pressure = np.max(
        sample,
        axis=1,
    )

    # --------------------------------------------------------
    # Contact Area
    # --------------------------------------------------------

    contact_area = np.sum(
        sample > 1000.0,
        axis=1,
    )

    # --------------------------------------------------------
    # Reshape to 4 × 4 pressure matrix
    # --------------------------------------------------------

    matrix = sample.reshape(
        sample.shape[0],
        4,
        4,
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COP
    # --------------------------------------------------------

    pressure_sum_safe = np.where(
        pressure_sum > 0,
        pressure_sum,
        1.0,
    )

    cop_x = np.sum(
        matrix * x_coords,
        axis=(1, 2),
    ) / pressure_sum_safe

    cop_y = np.sum(
        matrix * y_coords,
        axis=(1, 2),
    ) / pressure_sum_safe

    # --------------------------------------------------------
    # Left / Right
    # --------------------------------------------------------

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
        / np.where(
            right_pressure > 0,
            right_pressure,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Top / Bottom
    # --------------------------------------------------------

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
        / np.where(
            bottom_pressure > 0,
            bottom_pressure,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    frame_features = np.column_stack(
        [
            pressure_sum,
            max_pressure,
            contact_area,
            cop_x,
            cop_y,
            left_right_ratio,
            top_bottom_ratio,
        ]
    )

    return np.asarray(
        frame_features,
        dtype=np.float32,
    )


# ============================================================
# Dataset Feature Extraction
# ============================================================

def extract_dataset_frame_features(
    X: np.ndarray,
) -> np.ndarray:
    """
    Input:

        (samples, 50, 16)

    Output:

        (samples, 50, 7)
    """

    return np.asarray(
        [
            extract_frame_features(sample)
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
    使用 registration data 計算 normalization。

    X:
        (samples, frames, features)

    每一個 feature 使用 registration data
    計算 global min / max。
    """

    minimum = np.min(
        X,
        axis=(0, 1),
    )

    maximum = np.max(
        X,
        axis=(0, 1),
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
    Min-Max normalization。

    每一個 frame 的 feature 都使用
    registration data 得到的 min / max。
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
# DTW
# ============================================================

def dtw_distance(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
) -> float:
    """
    Multivariate Dynamic Time Warping。

    Parameters
    ----------
    sequence_a:
        Shape = (frames_a, features)

    sequence_b:
        Shape = (frames_b, features)

    Returns
    -------
    float
        DTW distance
    """

    sequence_a = np.asarray(
        sequence_a,
        dtype=np.float32,
    )

    sequence_b = np.asarray(
        sequence_b,
        dtype=np.float32,
    )

    if sequence_a.ndim != 2:
        raise ValueError(
            "sequence_a must be 2-dimensional"
        )

    if sequence_b.ndim != 2:
        raise ValueError(
            "sequence_b must be 2-dimensional"
        )

    if sequence_a.shape[1] != sequence_b.shape[1]:
        raise ValueError(
            "Both sequences must have "
            "the same feature dimension"
        )

    n = sequence_a.shape[0]
    m = sequence_b.shape[0]

    # --------------------------------------------------------
    # DTW cost matrix
    # --------------------------------------------------------

    dtw = np.full(
        (n + 1, m + 1),
        np.inf,
        dtype=np.float64,
    )

    dtw[0, 0] = 0.0

    # --------------------------------------------------------
    # Dynamic Programming
    # --------------------------------------------------------

    for i in range(1, n + 1):

        current = sequence_a[i - 1]

        for j in range(1, m + 1):

            other = sequence_b[j - 1]

            # Euclidean distance between
            # two frame feature vectors
            cost = np.linalg.norm(
                current - other
            )

            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )

    return float(
        dtw[n, m]
    )


# ============================================================
# DTW Normalized Distance
# ============================================================

def dtw_distance_normalized(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
) -> float:
    """
    DTW distance normalized by path length.

    這樣可以避免不同序列長度造成
    DTW distance 單純因為 path 長而變大。

    目前你的資料都是 50 frames，
    但保留 normalized DTW 仍然比較合理。
    """

    sequence_a = np.asarray(
        sequence_a,
        dtype=np.float32,
    )

    sequence_b = np.asarray(
        sequence_b,
        dtype=np.float32,
    )

    n = sequence_a.shape[0]
    m = sequence_b.shape[0]

    dtw = np.full(
        (n + 1, m + 1),
        np.inf,
        dtype=np.float64,
    )

    dtw[0, 0] = 0.0

    path_length = np.zeros(
        (n + 1, m + 1),
        dtype=np.int32,
    )

    for i in range(1, n + 1):

        current = sequence_a[i - 1]

        for j in range(1, m + 1):

            other = sequence_b[j - 1]

            cost = np.linalg.norm(
                current - other
            )

            # 找到上一個最小路徑
            previous_costs = (
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )

            previous_index = int(
                np.argmin(
                    previous_costs
                )
            )

            if previous_index == 0:

                previous_i = i - 1
                previous_j = j

            elif previous_index == 1:

                previous_i = i
                previous_j = j - 1

            else:

                previous_i = i - 1
                previous_j = j - 1

            dtw[i, j] = (
                cost
                + dtw[
                    previous_i,
                    previous_j,
                ]
            )

            path_length[i, j] = (
                path_length[
                    previous_i,
                    previous_j,
                ]
                + 1
            )

    final_path_length = path_length[
        n,
        m,
    ]

    if final_path_length == 0:
        return float("inf")

    return float(
        dtw[n, m]
        / final_path_length
    )


# ============================================================
# Template
# ============================================================

def build_template(
    registration_sequences: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    建立 DTW template。

    因為 DTW sequence 不能直接做普通 mean，
    所以使用 medoid：

        找出 registration 中
        「與其他 registration sequences
        平均 DTW distance 最小」的 sample。

    Returns
    -------
    template
    threshold
    registration_distances
    minimum
    maximum
    """

    # --------------------------------------------------------
    # Fit normalization using registration only
    # --------------------------------------------------------

    minimum, maximum = fit_normalization(
        registration_sequences
    )

    normalized = normalize(
        registration_sequences,
        minimum,
        maximum,
    )

    registration_count = (
        len(normalized)
    )

    # --------------------------------------------------------
    # 如果只有一筆 registration
    # --------------------------------------------------------

    if registration_count == 1:

        template = normalized[0]

        distances = np.asarray(
            [0.0],
            dtype=np.float32,
        )

    else:

        # ----------------------------------------------------
        # 計算 registration 彼此的 DTW distance
        # ----------------------------------------------------

        distance_matrix = np.zeros(
            (
                registration_count,
                registration_count,
            ),
            dtype=np.float32,
        )

        for i in range(
            registration_count
        ):

            for j in range(
                i + 1,
                registration_count,
            ):

                distance = (
                    dtw_distance_normalized(
                        normalized[i],
                        normalized[j],
                    )
                )

                distance_matrix[i, j] = (
                    distance
                )

                distance_matrix[j, i] = (
                    distance
                )

        # ----------------------------------------------------
        # Medoid
        # ----------------------------------------------------

        average_distances = (
            np.mean(
                distance_matrix,
                axis=1,
            )
        )

        medoid_index = int(
            np.argmin(
                average_distances
            )
        )

        template = normalized[
            medoid_index
        ]

        # ----------------------------------------------------
        # Registration → template
        # ----------------------------------------------------

        distances = np.asarray(
            [
                dtw_distance_normalized(
                    sequence,
                    template,
                )
                for sequence in normalized
            ],
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = (
        float(np.mean(distances))
        + THRESHOLD_K
        * float(np.std(distances))
    )

    return (
        template,
        threshold,
        distances,
        minimum,
        maximum,
        distance_matrix
        if registration_count > 1
        else np.zeros((1, 1)),
    )


# ============================================================
# Calculate Test Distances
# ============================================================

def calculate_distances(
    template: np.ndarray,
    test_sequences: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    """
    將 test sequences 使用 registration
    的 normalization 後，再計算 DTW distance。
    """

    normalized = normalize(
        test_sequences,
        minimum,
        maximum,
    )

    distances = np.asarray(
        [
            dtw_distance_normalized(
                template,
                sequence,
            )
            for sequence in normalized
        ],
        dtype=np.float32,
    )

    return distances


# ============================================================
# Formal Split
# ============================================================

def create_split(
    user_indices: np.ndarray,
    registration_count: int,
    test_count: int,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    if len(user_indices) < (
        registration_count
        + test_count
    ):

        raise ValueError(
            f"Not enough samples: "
            f"{len(user_indices)} available, "
            f"{registration_count + test_count} required."
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
            + test_count
        ]
    )

    return (
        registration_indices,
        test_indices,
    )


# ============================================================
# One Template Authentication
# ============================================================

def evaluate_template(
    template_user: str,
    template: np.ndarray,
    threshold: float,
    minimum: np.ndarray,
    maximum: np.ndarray,
    X_features: np.ndarray,
    test_indices_by_user: dict[
        str,
        np.ndarray,
    ],
) -> list[dict]:

    results = []

    for test_user in USERS:

        indices = (
            test_indices_by_user[
                test_user
            ]
        )

        test_sequences = (
            X_features[
                indices
            ]
        )

        distances = calculate_distances(
            template,
            test_sequences,
            minimum,
            maximum,
        )

        accepted = (
            distances <= threshold
        )

        accept_count = int(
            np.sum(accepted)
        )

        reject_count = int(
            len(accepted)
            - accept_count
        )

        acceptance_rate = (
            accept_count
            / len(accepted)
            if len(accepted) > 0
            else 0.0
        )

        results.append(
            {
                "template_user": template_user,
                "test_user": test_user,
                "accept": accept_count,
                "reject": reject_count,
                "acceptance_rate": acceptance_rate,
                "average_distance": float(
                    np.mean(distances)
                ),
            }
        )

    return results


# ============================================================
# Print Template Results
# ============================================================

def print_template_results(
    repeat: int,
    template_user: str,
    threshold: float,
    results: list[dict],
) -> None:

    print()
    print("=" * 90)

    print(
        f"Repeat {repeat:02d} | "
        f"Template = {template_user} | "
        f"Threshold = {threshold:.4f}"
    )

    print("=" * 90)

    print(
        f"{'Template':<12}"
        f"{'Test User':<15}"
        f"{'Accept':<10}"
        f"{'Reject':<10}"
        f"{'GAR':<12}"
        f"{'Avg DTW':<15}"
    )

    print("-" * 90)

    for row in results:

        print(
            f"{row['template_user']:<12}"
            f"{row['test_user']:<15}"
            f"{row['accept']:<10}"
            f"{row['reject']:<10}"
            f"{row['acceptance_rate'] * 100:>6.2f}%"
            f"{row['average_distance']:>14.4f}"
        )


# ============================================================
# Formal Benchmark
# ============================================================

def run_formal_benchmark(
    X: np.ndarray,
    labels: np.ndarray,
    registration_count: int = (
        DEFAULT_REGISTRATION_COUNT
    ),
    test_count: int = (
        DEFAULT_TEST_COUNT
    ),
    repeats: int = DEFAULT_REPEATS,
    seed: int = RANDOM_SEED,
) -> list[dict]:

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Frame-level features
    # --------------------------------------------------------

    X_features = (
        extract_dataset_frame_features(
            X
        )
    )

    print(
        f"\nDTW feature shape: "
        f"{X_features.shape}"
    )

    user_indices = {
        user: np.where(
            labels == user
        )[0]
        for user in USERS
    }

    all_rows = []

    # ========================================================
    # Repeat
    # ========================================================

    for repeat in range(
        1,
        repeats + 1,
    ):

        registration_indices_by_user = {}
        test_indices_by_user = {}

        # ----------------------------------------------------
        # Split each user
        # ----------------------------------------------------

        for user in USERS:

            (
                registration_indices,
                test_indices,
            ) = create_split(
                user_indices[user],
                registration_count,
                test_count,
                rng,
            )

            registration_indices_by_user[
                user
            ] = registration_indices

            test_indices_by_user[
                user
            ] = test_indices

        # ====================================================
        # Build template for each user
        # ====================================================

        for template_user in USERS:

            registration_indices = (
                registration_indices_by_user[
                    template_user
                ]
            )

            registration_sequences = (
                X_features[
                    registration_indices
                ]
            )

            (
                template,
                threshold,
                registration_distances,
                minimum,
                maximum,
                _,
            ) = build_template(
                registration_sequences
            )

            # ------------------------------------------------
            # Authentication
            # ------------------------------------------------

            results = evaluate_template(
                template_user=template_user,
                template=template,
                threshold=threshold,
                minimum=minimum,
                maximum=maximum,
                X_features=X_features,
                test_indices_by_user=(
                    test_indices_by_user
                ),
            )

            print_template_results(
                repeat=repeat,
                template_user=template_user,
                threshold=threshold,
                results=results,
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            for row in results:

                all_rows.append(
                    {
                        "repeat": repeat,
                        "template_user": (
                            template_user
                        ),
                        "test_user": (
                            row["test_user"]
                        ),
                        "accept": (
                            row["accept"]
                        ),
                        "reject": (
                            row["reject"]
                        ),
                        "gar": (
                            row[
                                "acceptance_rate"
                            ]
                        ),
                        "average_distance": (
                            row[
                                "average_distance"
                            ]
                        ),
                        "threshold": threshold,
                    }
                )

    return all_rows


# ============================================================
# Summary
# ============================================================

def print_formal_summary(
    rows: list[dict],
) -> None:

    print()
    print()
    print("=" * 110)
    print(
        " DTW FORMAL DOOR ACCESS BENCHMARK SUMMARY"
    )
    print("=" * 110)

    print()

    print(
        f"{'Template':<15}"
        f"{'GAR Mean':<12}"
        f"{'GAR Std':<12}"
        f"{'FRR Mean':<12}"
        f"{'FRR Std':<12}"
        f"{'FAR Mean':<12}"
        f"{'FAR Std':<12}"
        f"{'Threshold':<12}"
    )

    print("-" * 110)

    for template_user in USERS:

        template_rows = [
            row
            for row in rows
            if row["template_user"]
            == template_user
        ]

        # ----------------------------------------------------
        # Genuine
        # ----------------------------------------------------

        genuine_rows = [
            row
            for row in template_rows
            if row["test_user"]
            == template_user
        ]

        genuine_rates = np.asarray(
            [
                row["gar"]
                for row in genuine_rows
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Impostor
        # ----------------------------------------------------

        impostor_rows = [
            row
            for row in template_rows
            if row["test_user"]
            != template_user
        ]

        far_values = []

        for row in impostor_rows:

            total = (
                row["accept"]
                + row["reject"]
            )

            far_values.append(
                row["accept"]
                / total
                if total > 0
                else 0.0
            )

        far_values = np.asarray(
            far_values,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        thresholds = np.asarray(
            [
                row["threshold"]
                for row in template_rows
            ],
            dtype=np.float32,
        )

        gar_mean = float(
            np.mean(
                genuine_rates
            )
        )

        gar_std = float(
            np.std(
                genuine_rates
            )
        )

        frr_values = (
            1.0 - genuine_rates
        )

        frr_mean = float(
            np.mean(
                frr_values
            )
        )

        frr_std = float(
            np.std(
                frr_values
            )
        )

        far_mean = float(
            np.mean(
                far_values
            )
        )

        far_std = float(
            np.std(
                far_values
            )
        )

        threshold_mean = float(
            np.mean(
                thresholds
            )
        )

        print(
            f"{template_user:<15}"
            f"{gar_mean * 100:>7.2f}%"
            f"{gar_std * 100:>9.2f}%"
            f"{frr_mean * 100:>9.2f}%"
            f"{frr_std * 100:>9.2f}%"
            f"{far_mean * 100:>9.2f}%"
            f"{far_std * 100:>9.2f}%"
            f"{threshold_mean:>10.4f}"
        )


# ============================================================
# Template Identity Matrix
# ============================================================

def print_template_identity(
    rows: list[dict],
) -> None:

    print()
    print()
    print("=" * 110)
    print(
        " DTW TEMPLATE IDENTITY MATRIX"
    )
    print("=" * 110)

    print(
        "\n這張表回答："
        "\n「如果 template 是某個人的握力，"
        "其他人的握力會不會被 DTW 誤認？」"
    )

    for template_user in USERS:

        print()
        print(
            f"Template = {template_user}"
        )

        print(
            f"{'Test User':<15}"
            f"{'Accept Rate':<15}"
            f"{'Avg DTW':<18}"
            f"{'Decision'}"
        )

        print("-" * 70)

        for test_user in USERS:

            selected = [
                row
                for row in rows
                if row["template_user"]
                == template_user
                and row["test_user"]
                == test_user
            ]

            rates = np.asarray(
                [
                    row["gar"]
                    for row in selected
                ],
                dtype=np.float32,
            )

            distances = np.asarray(
                [
                    row["average_distance"]
                    for row in selected
                ],
                dtype=np.float32,
            )

            mean_rate = float(
                np.mean(rates)
            )

            mean_distance = float(
                np.mean(distances)
            )

            if test_user == template_user:

                decision = "GENUINE"

            elif mean_rate >= 0.5:

                decision = (
                    "⚠ POSSIBLE FALSE ACCEPT"
                )

            else:

                decision = "REJECT"

            print(
                f"{test_user:<15}"
                f"{mean_rate * 100:>7.2f}%"
                f"{mean_distance:>18.4f}"
                f"   {decision}"
            )


# ============================================================
# Few-Shot Experiment
# ============================================================

def run_few_shot_experiment(
    X: np.ndarray,
    labels: np.ndarray,
    registration_counts: Iterable[int] = (
        5,
        10,
        20,
    ),
    test_count: int = 30,
    repeats: int = 20,
    seed: int = RANDOM_SEED,
) -> list[dict]:

    print()
    print()
    print("=" * 110)
    print(
        " DTW FEW-SHOT REGISTRATION BENCHMARK"
    )
    print("=" * 110)

    print(
        "\n測試："
        "\n5 / 10 / 20 筆註冊資料"
        "\n30 筆 genuine test data"
        "\n其他使用者作為 impostor"
        f"\n每種設定重複 {repeats} 次"
    )

    rng = np.random.default_rng(
        seed
    )

    X_features = (
        extract_dataset_frame_features(
            X
        )
    )

    user_indices = {
        user: np.where(
            labels == user
        )[0]
        for user in USERS
    }

    all_results = []

    for registration_count in (
        registration_counts
    ):

        print()

        print(
            f"\n========== "
            f"Registration = "
            f"{registration_count} "
            f"=========="
        )

        print(
            f"{'User':<15}"
            f"{'GAR Mean':<12}"
            f"{'GAR Std':<12}"
            f"{'FRR Mean':<12}"
            f"{'FAR Mean':<12}"
            f"{'Threshold':<12}"
        )

        print("-" * 80)

        for user in USERS:

            gar_values = []
            far_values = []
            thresholds = []

            for repeat in range(
                repeats
            ):

                # ==================================================
                # Genuine split
                # ==================================================

                indices = rng.permutation(
                    user_indices[user]
                )

                if len(indices) < (
                    registration_count
                    + test_count
                ):

                    raise ValueError(
                        f"{user} does not have enough "
                        f"samples for registration={registration_count}, "
                        f"test={test_count}"
                    )

                registration_indices = (
                    indices[
                        :registration_count
                    ]
                )

                test_indices = (
                    indices[
                        registration_count:
                        registration_count
                        + test_count
                    ]
                )

                registration_sequences = (
                    X_features[
                        registration_indices
                    ]
                )

                test_sequences = (
                    X_features[
                        test_indices
                    ]
                )

                # ==================================================
                # Build DTW template
                # ==================================================

                (
                    template,
                    threshold,
                    _,
                    minimum,
                    maximum,
                    _,
                ) = build_template(
                    registration_sequences
                )

                # ==================================================
                # Genuine authentication
                # ==================================================

                distances = (
                    calculate_distances(
                        template,
                        test_sequences,
                        minimum,
                        maximum,
                    )
                )

                accepted = (
                    distances
                    <= threshold
                )

                gar = float(
                    np.mean(
                        accepted
                    )
                )

                gar_values.append(
                    gar
                )

                thresholds.append(
                    threshold
                )

                # ==================================================
                # Impostor authentication
                # ==================================================

                impostor_accepts = []

                for other_user in USERS:

                    if other_user == user:
                        continue

                    other_user_indices = (
                        user_indices[
                            other_user
                        ]
                    )

                    sample_size = min(
                        test_count,
                        len(
                            other_user_indices
                        ),
                    )

                    other_indices = (
                        rng.choice(
                            other_user_indices,
                            size=sample_size,
                            replace=False,
                        )
                    )

                    other_sequences = (
                        X_features[
                            other_indices
                        ]
                    )

                    other_distances = (
                        calculate_distances(
                            template,
                            other_sequences,
                            minimum,
                            maximum,
                        )
                    )

                    impostor_accepts.extend(
                        (
                            other_distances
                            <= threshold
                        ).tolist()
                    )

                far = float(
                    np.mean(
                        impostor_accepts
                    )
                )

                far_values.append(
                    far
                )

                all_results.append(
                    {
                        "registration_count":
                            registration_count,

                        "user":
                            user,

                        "repeat":
                            repeat + 1,

                        "gar":
                            gar,

                        "frr":
                            1.0 - gar,

                        "far":
                            far,

                        "threshold":
                            threshold,
                    }
                )

            # ==================================================
            # Summary
            # ==================================================

            print(
                f"{user:<15}"
                f"{np.mean(gar_values) * 100:>7.2f}%"
                f"{np.std(gar_values) * 100:>9.2f}%"
                f"{(1 - np.mean(gar_values)) * 100:>9.2f}%"
                f"{np.mean(far_values) * 100:>9.2f}%"
                f"{np.mean(thresholds):>10.4f}"
            )

    return all_results


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 110)

    print(
        " DTW FORMAL DOOR ACCESS BENCHMARK"
    )

    print("=" * 110)

    print(
        "\nDataset:"
        f" {DATA_DIR}"
    )

    print(
        "\nFormal protocol:"
        "\n  Registration : 20 samples"
        "\n  Authentication : 30 unseen samples"
        "\n  Repeated splits : 20"
        "\n  Users : amber / jay / 666 / background"
    )

    print(
        "\nDTW feature representation:"
        "\n  Pressure Sum"
        "\n  Max Pressure"
        "\n  Contact Area"
        "\n  COP X"
        "\n  COP Y"
        "\n  Left / Right Ratio"
        "\n  Top / Bottom Ratio"
    )

    # ========================================================
    # Load dataset
    # ========================================================

    X, raw_y = load_dataset(
        DATA_DIR
    )

    raw_y = np.asarray(
        raw_y
    )

    print(
        f"\nOriginal dataset shape: "
        f"{X.shape}"
    )

    # ========================================================
    # Dataset Distribution
    # ========================================================

    print(
        "\n========== Dataset Distribution =========="
    )

    for user in USERS:

        count = int(
            np.sum(
                raw_y == user
            )
        )

        print(
            f"{user:<15} "
            f"{count} samples"
        )

    # ========================================================
    # Formal 20 / 30 Benchmark
    # ========================================================

    rows = run_formal_benchmark(
        X=X,
        labels=raw_y,
        registration_count=20,
        test_count=30,
        repeats=20,
        seed=42,
    )

    print_formal_summary(
        rows
    )

    print_template_identity(
        rows
    )

    # ========================================================
    # Few Shot
    # ========================================================

    run_few_shot_experiment(
        X=X,
        labels=raw_y,
        registration_counts=(
            5,
            10,
            20,
        ),
        test_count=30,
        repeats=20,
        seed=42,
    )

    # ========================================================

    print()
    print("=" * 110)

    print(
        " DTW Benchmark Finished"
    )

    print("=" * 110)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()