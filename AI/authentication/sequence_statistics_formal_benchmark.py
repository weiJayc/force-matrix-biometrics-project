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
# IMPORTANT:
# 把這裡改成你目前 Sequence Statistics 使用的函式 / class
#
# 如果你的 sequence statistics 已經有自己的 benchmark class，
# 可以直接在這裡 import。
#
# 下面這份程式自己實作 sequence statistics，
# 不依賴原本的 registration.py / authentication.py。
# ============================================================


USERS = ("amber", "jay", "666", "background")

DEFAULT_REGISTRATION_COUNT = 20
DEFAULT_TEST_COUNT = 30
DEFAULT_REPEATS = 20

RANDOM_SEED = 42


# ============================================================
# Sequence Statistics
# ============================================================

def extract_sequence_statistics(
    sample: np.ndarray,
) -> np.ndarray:
    """
    將一整段 50-frame 握力資料轉成 sequence-level statistics。

    Input:
        sample: (50, 16)

    Output:
        一維 sequence feature vector
    """

    sample = np.asarray(sample, dtype=np.float32)

    if sample.ndim != 2:
        raise ValueError(
            f"Expected sample shape (frames, sensors), got {sample.shape}"
        )

    if sample.shape[1] != 16:
        raise ValueError(
            f"Expected 16 sensors, got {sample.shape[1]}"
        )

    # --------------------------------------------------------
    # 每一 frame 的基本壓力
    # --------------------------------------------------------

    pressure_sum = np.sum(sample, axis=1)
    max_pressure = np.max(sample, axis=1)

    contact_area = np.sum(sample > 1000.0, axis=1)

    # --------------------------------------------------------
    # COP
    # --------------------------------------------------------

    matrix = sample.reshape(sample.shape[0], 4, 4)

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
        left_pressure /
        np.where(right_pressure > 0, right_pressure, 1.0)
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
        top_pressure /
        np.where(bottom_pressure > 0, bottom_pressure, 1.0)
    )

    # ========================================================
    # Sequence-level statistics
    # ========================================================

    features = []

    frame_features = {
        "pressure_sum": pressure_sum,
        "max_pressure": max_pressure,
        "contact_area": contact_area,
        "cop_x": cop_x,
        "cop_y": cop_y,
        "left_right_ratio": left_right_ratio,
        "top_bottom_ratio": top_bottom_ratio,
    }

    for name, values in frame_features.items():

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Level
        # ----------------------------------------------------

        features.append(np.mean(values))
        features.append(np.std(values))
        features.append(np.min(values))
        features.append(np.max(values))

        # ----------------------------------------------------
        # Temporal change
        # ----------------------------------------------------

        if len(values) > 1:
            diff = np.diff(values)

            features.append(np.mean(np.abs(diff)))
            features.append(np.std(diff))
            features.append(np.max(np.abs(diff)))

        else:
            features.extend(
                [0.0, 0.0, 0.0]
            )

        # ----------------------------------------------------
        # Start / End
        # ----------------------------------------------------

        features.append(values[0])
        features.append(values[-1])

    # --------------------------------------------------------
    # Whole-sequence global features
    # --------------------------------------------------------

    features.append(
        float(np.argmax(pressure_sum))
    )

    features.append(
        float(np.argmax(max_pressure))
    )

    # Total pressure accumulated through the whole grip
    features.append(
        float(np.sum(pressure_sum))
    )

    # Pressure variation through the sequence
    features.append(
        float(np.std(pressure_sum))
    )

    return np.asarray(
        features,
        dtype=np.float32,
    )


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
            extract_sequence_statistics(sample)
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
    Calculate normalization statistics using registration data only.
    """

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    return minimum, maximum


def normalize(
    X: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:

    denominator = maximum - minimum

    denominator = np.where(
        denominator == 0,
        1.0,
        denominator,
    )

    return (
        (X - minimum) / denominator
    ).astype(np.float32)


# ============================================================
# Template
# ============================================================

def build_template(
    registration_features: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    建立 template。

    Returns:
        template
        threshold
        registration distances
    """

    minimum, maximum = fit_normalization(
        registration_features
    )

    normalized = normalize(
        registration_features,
        minimum,
        maximum,
    )

    template = np.mean(
        normalized,
        axis=0,
    )

    distances = np.linalg.norm(
        normalized - template,
        axis=1,
    )

    threshold = (
        float(np.mean(distances))
        + 2.0 * float(np.std(distances))
    )

    return (
        template,
        threshold,
        distances,
    )


# ============================================================
# Distance
# ============================================================

def calculate_distances(
    template: np.ndarray,
    test_features: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:

    normalized = normalize(
        test_features,
        minimum,
        maximum,
    )

    return np.linalg.norm(
        normalized - template,
        axis=1,
    )


# ============================================================
# Formal Split
# ============================================================

def create_split(
    user_indices: np.ndarray,
    registration_count: int,
    test_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    if len(user_indices) < registration_count + test_count:
        raise ValueError(
            f"Not enough samples: "
            f"{len(user_indices)} available, "
            f"{registration_count + test_count} required."
        )

    shuffled = rng.permutation(
        user_indices
    )

    registration_indices = shuffled[
        :registration_count
    ]

    test_indices = shuffled[
        registration_count:
        registration_count + test_count
    ]

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
    labels: np.ndarray,
    test_indices_by_user: dict[str, np.ndarray],
) -> dict:

    results = []

    for test_user in USERS:

        indices = test_indices_by_user[
            test_user
        ]

        distances = calculate_distances(
            template,
            X_features[indices],
            minimum,
            maximum,
        )

        accepted = distances <= threshold

        accept_count = int(
            np.sum(accepted)
        )

        reject_count = int(
            len(accepted) - accept_count
        )

        acceptance_rate = (
            accept_count / len(accepted)
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
    results: list[dict],
) -> None:

    print()
    print("=" * 75)
    print(
        f"Repeat {repeat:02d} | "
        f"Template = {template_user}"
    )
    print("=" * 75)

    print(
        f"{'Template':<12}"
        f"{'Test User':<15}"
        f"{'Accept':<10}"
        f"{'Reject':<10}"
        f"{'GAR':<12}"
        f"{'Avg Distance':<15}"
    )

    print("-" * 75)

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
# Formal 20 / 30 Benchmark
# ============================================================

def run_formal_benchmark(
    X: np.ndarray,
    labels: np.ndarray,
    registration_count: int = DEFAULT_REGISTRATION_COUNT,
    test_count: int = DEFAULT_TEST_COUNT,
    repeats: int = DEFAULT_REPEATS,
    seed: int = RANDOM_SEED,
) -> list[dict]:

    rng = np.random.default_rng(seed)

    X_features = extract_dataset_statistics(
        X
    )

    user_indices = {
        user: np.where(labels == user)[0]
        for user in USERS
    }

    all_rows = []

    # ========================================================
    # Repeat
    # ========================================================

    for repeat in range(1, repeats + 1):

        registration_indices_by_user = {}
        test_indices_by_user = {}

        # ----------------------------------------------------
        # 每個使用者各自切分
        # ----------------------------------------------------

        for user in USERS:

            registration_indices, test_indices = create_split(
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
        # 建立每個使用者 template
        # ====================================================

        for template_user in USERS:

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

            (
                template,
                threshold,
                registration_distances,
            ) = build_template(
                registration_features
            )

            # ------------------------------------------------
            # 注意：
            # normalization statistics 必須來自 registration
            # ------------------------------------------------

            minimum, maximum = fit_normalization(
                registration_features
            )

            normalized_registration = normalize(
                registration_features,
                minimum,
                maximum,
            )

            template = np.mean(
                normalized_registration,
                axis=0,
            )

            # ------------------------------------------------
            # 正式測試
            # ------------------------------------------------

            results = evaluate_template(
                template_user=template_user,
                template=template,
                threshold=threshold,
                minimum=minimum,
                maximum=maximum,
                X_features=X_features,
                labels=labels,
                test_indices_by_user=test_indices_by_user,
            )

            print_template_results(
                repeat,
                template_user,
                results,
            )

            # ------------------------------------------------
            # 儲存結果
            # ------------------------------------------------

            for row in results:

                all_rows.append(
                    {
                        "repeat": repeat,
                        "template_user": template_user,
                        "test_user": row["test_user"],
                        "accept": row["accept"],
                        "reject": row["reject"],
                        "gar": row["acceptance_rate"],
                        "average_distance": row[
                            "average_distance"
                        ],
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
    print("=" * 100)
    print(" FORMAL DOOR ACCESS BENCHMARK SUMMARY")
    print("=" * 100)

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

    print("-" * 100)

    for template_user in USERS:

        template_rows = [
            row
            for row in rows
            if row["template_user"]
            == template_user
        ]

        # ----------------------------------------------
        # Genuine:
        # template user == test user
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Impostor:
        # template user != test user
        # ----------------------------------------------

        impostor_rows = [
            row
            for row in template_rows
            if row["test_user"]
            != template_user
        ]

        # ----------------------------------------------
        # FAR:
        # 所有 impostor 中被錯誤接受
        # ----------------------------------------------

        far_values = []

        for row in impostor_rows:

            total = (
                row["accept"]
                + row["reject"]
            )

            far_values.append(
                row["accept"] / total
                if total > 0
                else 0.0
            )

        far_values = np.asarray(
            far_values,
            dtype=np.float32,
        )

        # ----------------------------------------------
        # Threshold
        # ----------------------------------------------

        thresholds = np.asarray(
            [
                row["threshold"]
                for row in template_rows
            ],
            dtype=np.float32,
        )

        gar_mean = float(
            np.mean(genuine_rates)
        )

        gar_std = float(
            np.std(genuine_rates)
        )

        frr_values = 1.0 - genuine_rates

        frr_mean = float(
            np.mean(frr_values)
        )

        frr_std = float(
            np.std(frr_values)
        )

        far_mean = float(
            np.mean(far_values)
        )

        far_std = float(
            np.std(far_values)
        )

        threshold_mean = float(
            np.mean(thresholds)
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
    print("=" * 100)
    print(" TEMPLATE IDENTITY MATRIX")
    print("=" * 100)

    print(
        "\n這張表回答："
        "\n「如果 template 是某個人的握力，"
        "其他人的資料會不會被它誤認？」"
    )

    for template_user in USERS:

        print()
        print(
            f"Template = {template_user}"
        )

        print(
            f"{'Test User':<15}"
            f"{'Accept Rate':<15}"
            f"{'Avg Distance':<18}"
            f"{'Decision'}"
        )

        print("-" * 65)

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
                decision = "⚠ POSSIBLE FALSE ACCEPT"
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
    print("=" * 100)
    print(" FEW-SHOT REGISTRATION BENCHMARK")
    print("=" * 100)

    print(
        "\n測試："
        "\n5 / 10 / 20 筆註冊資料"
        "\n剩餘 30 筆作為未知測試資料"
        f"\n每種設定重複 {repeats} 次"
    )

    rng = np.random.default_rng(seed)

    X_features = extract_dataset_statistics(
        X
    )

    user_indices = {
        user: np.where(labels == user)[0]
        for user in USERS
    }

    all_results = []

    for registration_count in registration_counts:

        print()
        print(
            f"\n========== "
            f"Registration = {registration_count} "
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

        print("-" * 75)

        for user in USERS:

            gar_values = []
            far_values = []
            thresholds = []

            for repeat in range(repeats):

                indices = rng.permutation(
                    user_indices[user]
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

                registration_features = (
                    X_features[
                        registration_indices
                    ]
                )

                test_features = (
                    X_features[
                        test_indices
                    ]
                )

                template, threshold, _ = (
                    build_template(
                        registration_features
                    )
                )

                minimum, maximum = (
                    fit_normalization(
                        registration_features
                    )
                )

                distances = calculate_distances(
                    template,
                    test_features,
                    minimum,
                    maximum,
                )

                accepted = (
                    distances <= threshold
                )

                gar = float(
                    np.mean(accepted)
                )

                gar_values.append(
                    gar
                )

                thresholds.append(
                    threshold
                )

                # ----------------------------------------
                # 這裡是 genuine-only，
                # 所以 FAR 必須另外用其他使用者測試
                # ----------------------------------------

                impostor_accepts = []

                for other_user in USERS:

                    if other_user == user:
                        continue

                    other_indices = rng.choice(
                        user_indices[
                            other_user
                        ],
                        size=min(
                            test_count,
                            len(
                                user_indices[
                                    other_user
                                ]
                            ),
                        ),
                        replace=False,
                    )

                    other_features = (
                        X_features[
                            other_indices
                        ]
                    )

                    other_distances = (
                        calculate_distances(
                            template,
                            other_features,
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
                        "user": user,
                        "repeat": repeat + 1,
                        "gar": gar,
                        "frr": 1.0 - gar,
                        "far": far,
                        "threshold": threshold,
                    }
                )

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
    print("=" * 100)
    print(" SEQUENCE STATISTICS FORMAL DOOR ACCESS BENCHMARK")
    print("=" * 100)

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

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, raw_y = load_dataset(
        DATA_DIR
    )

    raw_y = np.asarray(
        raw_y
    )

    print(
        f"\nOriginal dataset shape: {X.shape}"
    )

    # --------------------------------------------------------
    # Check user sample count
    # --------------------------------------------------------

    print(
        "\n========== Dataset Distribution =========="
    )

    for user in USERS:

        count = int(
            np.sum(raw_y == user)
        )

        print(
            f"{user:<15} {count} samples"
        )

    # ========================================================
    # 20 registration / 30 test
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
    # Few-shot
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
    print("=" * 100)
    print(" Benchmark Finished")
    print("=" * 100)


if __name__ == "__main__":
    main()