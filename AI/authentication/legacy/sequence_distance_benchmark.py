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

from AI.authentication.legacy.feature_extractor import (
    ENGINEERED_FEATURE_ORDER,
    extract_feature_combination,
)


# ============================================================
# Configuration
# ============================================================

# 真正可以註冊的使用者
USERS = (
    "amber",
    "jay",
    "666",
)

# background 不屬於使用者
NO_CONTACT_USER = "background"

# 每次正式實驗：
# registration_count 筆建立 template
# 另外 30 筆 genuine test
REGISTRATION_COUNTS = (
    5,
    10,
    20,
)

TEST_COUNT = 30

# 每種設定重複 20 次
REPEATS = 20

RANDOM_SEED = 42


# ============================================================
# Data
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:
    return load_dataset(DATA_DIR)


def get_user_samples(
    X: np.ndarray,
    y: np.ndarray,
    user_id: str,
) -> np.ndarray:

    mask = np.asarray(y) == user_id

    return X[mask]


# ============================================================
# Feature Extraction
# ============================================================

def extract_sequences(
    samples: np.ndarray,
) -> np.ndarray:
    """
    將 raw pressure sequence：

        (N, 50, 16)

    轉成：

        (N, 50, 7)

    每一個 frame 都保留工程化特徵：

        pressure_sum
        max_pressure
        contact_area
        cop_x
        cop_y
        left_right_ratio
        top_bottom_ratio

    注意：
    這裡「不做 sequence statistics」。

    我們要保留完整的時間序列。
    """

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    if samples.ndim != 3:
        raise ValueError(
            f"Expected 3D samples, got {samples.shape}"
        )

    if samples.shape[2] != 16:
        raise ValueError(
            f"Expected 16 sensors, got {samples.shape}"
        )

    return extract_feature_combination(
        samples,
        feature_names=ENGINEERED_FEATURE_ORDER,
    ).astype(np.float32)


# ============================================================
# Sequence Template
# ============================================================

def build_sequence_template(
    registration_sequences: np.ndarray,
) -> np.ndarray:
    """
    建立 sequence-level template。

    Input:

        (N, 50, 7)

    Output:

        (50, 7)

    和目前 Sequence Statistics 最大的不同：

    Sequence Statistics：
        整段 sequence
        ↓
        統計數值
        ↓
        一個 vector

    本方法：
        整段 sequence
        ↓
        保留 50 個時間位置
        ↓
        50 × 7 template
    """

    registration_sequences = np.asarray(
        registration_sequences,
        dtype=np.float32,
    )

    if registration_sequences.ndim != 3:
        raise ValueError(
            "Expected registration sequences with shape "
            "(N, frames, features)"
        )

    return np.mean(
        registration_sequences,
        axis=0,
    ).astype(np.float32)


# ============================================================
# Sequence-level Distance
# ============================================================

def sequence_euclidean_distance(
    template_sequence: np.ndarray,
    test_sequence: np.ndarray,
) -> float:
    """
    計算兩段 sequence 的 Euclidean distance。

    template:
        (50, 7)

    test:
        (50, 7)

    做法：

        每個 frame 對應比較
        ↓
        7 個 feature 的差異
        ↓
        全部 50 frames 一起累積
        ↓
        sqrt(sum(diff²))

    注意：

    這仍然要求：

        template frame 1 ↔ test frame 1
        template frame 2 ↔ test frame 2
        ...
        template frame 50 ↔ test frame 50

    所以這是：

        Sequence-level Euclidean

    而不是 DTW。

    DTW 會在下一個實驗處理「時間對齊」問題。
    """

    template_sequence = np.asarray(
        template_sequence,
        dtype=np.float32,
    )

    test_sequence = np.asarray(
        test_sequence,
        dtype=np.float32,
    )

    if template_sequence.shape != test_sequence.shape:
        raise ValueError(
            "Sequence shape mismatch: "
            f"template={template_sequence.shape}, "
            f"test={test_sequence.shape}"
        )

    difference = (
        template_sequence
        - test_sequence
    )

    return float(
        np.linalg.norm(difference)
    )


# ============================================================
# Registration Distance
# ============================================================

def compute_registration_distances(
    template_sequence: np.ndarray,
    registration_sequences: np.ndarray,
) -> np.ndarray:
    """
    計算註冊資料與 template 的 sequence distance。

    用途：

        registration distances
            ↓
        mean + 2 * std
            ↓
        per-user threshold

    threshold 完全只使用 registration data。

    不偷看：
        genuine test
        impostor
        background
    """

    distances = []

    for sequence in registration_sequences:

        distance = sequence_euclidean_distance(
            template_sequence,
            sequence,
        )

        distances.append(distance)

    return np.asarray(
        distances,
        dtype=np.float32,
    )


# ============================================================
# Threshold
# ============================================================

def compute_threshold(
    distances: np.ndarray,
    k: float = 2.0,
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
            "Registration distances cannot be empty"
        )

    return float(
        np.mean(distances)
        + k * np.std(distances)
    )


# ============================================================
# Evaluate Sequences
# ============================================================

def evaluate_sequences(
    template_sequence: np.ndarray,
    threshold: float,
    sequences: np.ndarray,
) -> dict[str, float]:
    """
    用同一套算法評估一批 sequence。

    回傳：

        accept
        reject
        rate
        average_distance
    """

    distances = []

    accept_count = 0

    for sequence in sequences:

        distance = sequence_euclidean_distance(
            template_sequence,
            sequence,
        )

        distances.append(distance)

        if distance <= threshold:
            accept_count += 1

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    total = len(sequences)

    rate = (
        accept_count / total
        if total > 0
        else 0.0
    )

    return {
        "accept": float(accept_count),

        "reject": float(
            total - accept_count
        ),

        "rate": float(rate),

        "average_distance": (
            float(np.mean(distances))
            if len(distances) > 0
            else 0.0
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
    """
    執行一次完整正式門禁實驗。

    流程：

        使用者資料
            ↓
        random split
            ↓
        registration
            ↓
        sequence template
            ↓
        threshold
            ↓
        genuine test
        impostor test
        no-contact test
    """

    # ========================================================
    # 1. Genuine user pool
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
    # 2. Random split
    # ========================================================

    shuffled_indices = rng.permutation(
        len(genuine_pool)
    )

    registration_indices = (
        shuffled_indices[
            :registration_count
        ]
    )

    genuine_test_indices = (
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

    genuine_test_samples = (
        genuine_pool[
            genuine_test_indices
        ]
    )

    # ========================================================
    # 3. Extract sequence features
    # ========================================================

    registration_sequences = extract_sequences(
        registration_samples
    )

    genuine_test_sequences = extract_sequences(
        genuine_test_samples
    )

    # ========================================================
    # 4. Build sequence template
    # ========================================================

    template_sequence = build_sequence_template(
        registration_sequences
    )

    # ========================================================
    # 5. Registration distances
    # ========================================================

    registration_distances = (
        compute_registration_distances(
            template_sequence,
            registration_sequences,
        )
    )

    # ========================================================
    # 6. Threshold
    # ========================================================

    threshold = compute_threshold(
        registration_distances,
        k=2.0,
    )

    # ========================================================
    # 7. Genuine
    # ========================================================

    genuine_result = evaluate_sequences(
        template_sequence,
        threshold,
        genuine_test_sequences,
    )

    # ========================================================
    # 8. Impostors
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
                f"{impostor_user} has "
                f"{len(impostor_pool)} samples, "
                f"but needs {TEST_COUNT}."
            )

        # 每個 repeat 都重新抽 30 筆
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

        impostor_sequences = extract_sequences(
            impostor_samples
        )

        result = evaluate_sequences(
            template_sequence,
            threshold,
            impostor_sequences,
        )

        impostor_results[
            impostor_user
        ] = result

        total_impostor_accept += (
            result["accept"]
        )

        total_impostor_samples += (
            TEST_COUNT
        )

        all_impostor_distances.extend(
            [
                sequence_euclidean_distance(
                    template_sequence,
                    sequence,
                )
                for sequence
                in impostor_sequences
            ]
        )

    # ========================================================
    # 9. FAR
    # ========================================================

    far = (
        total_impostor_accept
        / total_impostor_samples
        if total_impostor_samples > 0
        else 0.0
    )

    # ========================================================
    # 10. No-contact
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

    no_contact_sequences = extract_sequences(
        no_contact_samples
    )

    no_contact_result = evaluate_sequences(
        template_sequence,
        threshold,
        no_contact_sequences,
    )

    # ========================================================
    # 11. Final metrics
    # ========================================================

    gar = genuine_result["rate"]

    frr = 1.0 - gar

    no_contact_far = (
        no_contact_result["rate"]
    )

    return {

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        "template_user": template_user,

        "registration_count": (
            registration_count
        ),

        "repeat": repeat_index,

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        "threshold": threshold,

        # ----------------------------------------------------
        # Genuine
        # ----------------------------------------------------

        "gar": gar,

        "frr": frr,

        "genuine_accept": (
            genuine_result["accept"]
        ),

        "genuine_reject": (
            genuine_result["reject"]
        ),

        "genuine_distance": (
            genuine_result[
                "average_distance"
            ]
        ),

        # ----------------------------------------------------
        # Impostor
        # ----------------------------------------------------

        "far": far,

        "impostor_distance": (
            float(
                np.mean(
                    all_impostor_distances
                )
            )
            if all_impostor_distances
            else 0.0
        ),

        # ----------------------------------------------------
        # No-contact
        # ----------------------------------------------------

        "no_contact_far": (
            no_contact_far
        ),

        "no_contact_distance": (
            no_contact_result[
                "average_distance"
            ]
        ),

        # ----------------------------------------------------
        # Detailed
        # ----------------------------------------------------

        "impostor_results": (
            impostor_results
        ),
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

    all_results = []

    for registration_count in (
        REGISTRATION_COUNTS
    ):

        for repeat in range(
            1,
            REPEATS + 1,
        ):

            for template_user in USERS:

                result = run_one_experiment(
                    X=X,
                    y=y,
                    template_user=template_user,
                    registration_count=registration_count,
                    repeat_index=repeat,
                    rng=rng,
                )

                all_results.append(
                    result
                )

    return all_results


# ============================================================
# Detailed Template -> Test User
# ============================================================

def print_template_results(
    results: list[dict],
) -> None:

    print()
    print("=" * 115)
    print(
        "SEQUENCE-LEVEL EUCLIDEAN "
        "TEMPLATE -> TEST USER"
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

        template_user = row[
            "template_user"
        ]

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
            f"{'NO-CONTACT':<15}"
            f"{'':>10}"
            f"{'':>10}"
            f"{row['no_contact_far'] * 100:>11.2f}%"
            f"{row['no_contact_distance']:>18.4f}"
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
        "SEQUENCE-LEVEL EUCLIDEAN "
        "FORMAL DOOR ACCESS SUMMARY"
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

            if not rows:
                continue

            gar = np.asarray(
                [
                    row["gar"]
                    for row in rows
                ],
                dtype=np.float32,
            )

            frr = np.asarray(
                [
                    row["frr"]
                    for row in rows
                ],
                dtype=np.float32,
            )

            far = np.asarray(
                [
                    row["far"]
                    for row in rows
                ],
                dtype=np.float32,
            )

            no_contact_far = np.asarray(
                [
                    row[
                        "no_contact_far"
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            )

            thresholds = np.asarray(
                [
                    row["threshold"]
                    for row in rows
                ],
                dtype=np.float32,
            )

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
        "SEQUENCE-LEVEL EUCLIDEAN "
        "DISTANCE SUMMARY"
    )
    print("=" * 105)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Genuine Distance':>22}"
        f"{'Impostor Distance':>22}"
        f"{'No-contact Distance':>24}"
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

            if not rows:
                continue

            genuine_distance = np.mean(
                [
                    row[
                        "genuine_distance"
                    ]
                    for row in rows
                ]
            )

            impostor_distance = np.mean(
                [
                    row[
                        "impostor_distance"
                    ]
                    for row in rows
                ]
            )

            no_contact_distance = np.mean(
                [
                    row[
                        "no_contact_distance"
                    ]
                    for row in rows
                ]
            )

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{genuine_distance:>22.4f}"
                f"{impostor_distance:>22.4f}"
                f"{no_contact_distance:>24.4f}"
            )


# ============================================================
# Separation Summary
# ============================================================

def print_separation_summary(
    results: list[dict],
) -> None:
    """
    額外輸出：

        Genuine distance
        Impostor distance
        No-contact distance
        Threshold

    的相對位置。

    這可以幫助判斷：

        template 是不是比較像本人？
        no-contact 是否容易被接受？
    """

    print()
    print("=" * 125)
    print(
        "SEQUENCE-LEVEL DISTANCE SEPARATION"
    )
    print("=" * 125)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Genuine':>15}"
        f"{'Threshold':>15}"
        f"{'No-contact':>15}"
        f"{'Impostor':>15}"
        f"{'Imp-Genuine':>18}"
        f"{'NC-Genuine':>18}"
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

            if not rows:
                continue

            genuine = np.mean(
                [
                    row[
                        "genuine_distance"
                    ]
                    for row in rows
                ]
            )

            threshold = np.mean(
                [
                    row[
                        "threshold"
                    ]
                    for row in rows
                ]
            )

            no_contact = np.mean(
                [
                    row[
                        "no_contact_distance"
                    ]
                    for row in rows
                ]
            )

            impostor = np.mean(
                [
                    row[
                        "impostor_distance"
                    ]
                    for row in rows
                ]
            )

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{genuine:>15.4f}"
                f"{threshold:>15.4f}"
                f"{no_contact:>15.4f}"
                f"{impostor:>15.4f}"
                f"{impostor - genuine:>18.4f}"
                f"{no_contact - genuine:>18.4f}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    X, y = load_data()

    print()
    print("=" * 100)
    print(
        "Sequence-Level Euclidean "
        "Formal Door Access Benchmark"
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

    print()
    print(
        "Method:"
    )

    print(
        "  Raw pressure (50 × 16)"
    )

    print(
        "        ↓"
    )

    print(
        "  Engineered features (50 × 7)"
    )

    print(
        "        ↓"
    )

    print(
        "  Mean sequence template (50 × 7)"
    )

    print(
        "        ↓"
    )

    print(
        "  Sequence-level Euclidean distance"
    )

    print(
        "        ↓"
    )

    print(
        "  Threshold = registration mean "
        "+ 2 × std"
    )

    # ========================================================
    # Benchmark
    # ========================================================

    results = run_benchmark(
        X,
        y,
    )

    # ========================================================
    # Detailed
    # ========================================================

    print_template_results(
        results
    )

    # ========================================================
    # Main metrics
    # ========================================================

    print_summary(
        results
    )

    # ========================================================
    # Distances
    # ========================================================

    print_distance_summary(
        results
    )

    # ========================================================
    # Separation
    # ========================================================

    print_separation_summary(
        results
    )

    print()
    print("=" * 100)
    print(
        "Sequence-Level Euclidean "
        "Benchmark Finished"
    )
    print("=" * 100)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()