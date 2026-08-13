from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Iterable

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data_loader import load_dataset

from authentication.authentication import AuthenticationSystem
from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER
from authentication.registration import RegistrationSystem
from authentication.template import TemplateManager
from authentication.threshold import ThresholdManager


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"

# 真正的註冊使用者
USERS = ("amber", "jay", "666")

# 只用來做 No-contact rejection test
NO_CONTACT_USER = "background"

# 正式驗證設定
REGISTRATION_COUNTS = (5, 10, 20)
TEST_COUNT = 30
REPEATS = 20

RANDOM_SEED = 42


# ============================================================
# Data loading
# ============================================================

def load_data() -> tuple[np.ndarray, np.ndarray]:
    return load_dataset(DATA_DIR)


def get_user_samples(
    X: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
) -> np.ndarray:
    mask = np.asarray(raw_y) == user_id
    return X[mask]


# ============================================================
# Registration
# ============================================================

def build_template(
    registration_samples: np.ndarray,
    user_id: str,
):
    """
    使用註冊資料建立：

    1. User Template
    2. User Threshold

    注意：
    threshold 只根據 registration samples 計算，
    不會使用其他使用者資料。
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        template_manager = TemplateManager(
            storage_dir=Path(temp_dir) / "templates"
        )

        threshold_manager = ThresholdManager(
            storage_dir=Path(temp_dir) / "thresholds"
        )

        registration_system = RegistrationSystem(
            template_manager=template_manager,
            threshold_manager=threshold_manager,
            feature_names=ENGINEERED_FEATURE_ORDER,
        )

        template, threshold, distances = (
            registration_system.register_user(
                registration_samples,
                user_id=user_id,
                return_details=True,
            )
        )

    return template, threshold, distances


# ============================================================
# Authentication
# ============================================================

def evaluate_samples(
    template,
    threshold,
    samples: np.ndarray,
) -> tuple[int, int, float, float]:
    """
    測試一批 samples。

    Returns
    -------
    accept_count
    reject_count
    acceptance_rate
    average_distance
    """

    auth_system = AuthenticationSystem(
        feature_names=ENGINEERED_FEATURE_ORDER
    )

    accept_count = 0
    reject_count = 0
    distances: list[float] = []

    for sample in samples:

        result = auth_system.authenticate(
            template,
            threshold,
            sample[np.newaxis, :, :],
        )

        distances.append(result.distance)

        if result.accept:
            accept_count += 1
        else:
            reject_count += 1

    total = len(samples)

    acceptance_rate = (
        accept_count / total
        if total > 0
        else 0.0
    )

    average_distance = (
        float(np.mean(distances))
        if distances
        else 0.0
    )

    return (
        accept_count,
        reject_count,
        acceptance_rate,
        average_distance,
    )


# ============================================================
# 1. Registration -> Authentication
# ============================================================

def run_template_authentication_test(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    print()
    print("=" * 90)
    print(" Sequence Statistics - User Authentication")
    print("=" * 90)

    for template_user in USERS:

        user_samples = get_user_samples(
            X,
            y,
            template_user,
        )

        # 使用全部資料建立 template
        template, threshold, _ = build_template(
            user_samples,
            template_user,
        )

        print()
        print(
            f"Template = {template_user}"
        )

        print(
            f"{'Test User':<15}"
            f"{'Accept':>10}"
            f"{'Reject':>10}"
            f"{'GAR':>12}"
            f"{'Avg Distance':>18}"
        )

        print("-" * 70)

        # 測試真正使用者
        for test_user in USERS:

            test_samples = get_user_samples(
                X,
                y,
                test_user,
            )

            (
                accept_count,
                reject_count,
                acceptance_rate,
                avg_distance,
            ) = evaluate_samples(
                template,
                threshold,
                test_samples,
            )

            print(
                f"{test_user:<15}"
                f"{accept_count:>10}"
                f"{reject_count:>10}"
                f"{acceptance_rate * 100:>11.2f}%"
                f"{avg_distance:>18.4f}"
            )


# ============================================================
# 2. No-contact rejection test
# ============================================================

def run_no_contact_test(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    print()
    print("=" * 90)
    print(" No-contact Rejection Test")
    print("=" * 90)

    print(
        "\nBackground 不建立 Template。"
    )

    print(
        "它只用來測試："
        "沒有握門把時，系統是否會誤接受。\n"
    )

    print(
        f"{'Template User':<18}"
        f"{'Accept':>10}"
        f"{'Reject':>10}"
        f"{'FAR':>12}"
        f"{'Avg Distance':>18}"
        f"{'Threshold':>14}"
    )

    print("-" * 85)

    no_contact_samples = get_user_samples(
        X,
        y,
        NO_CONTACT_USER,
    )

    for template_user in USERS:

        user_samples = get_user_samples(
            X,
            y,
            template_user,
        )

        template, threshold, _ = build_template(
            user_samples,
            template_user,
        )

        (
            accept_count,
            reject_count,
            acceptance_rate,
            avg_distance,
        ) = evaluate_samples(
            template,
            threshold,
            no_contact_samples,
        )

        print(
            f"{template_user:<18}"
            f"{accept_count:>10}"
            f"{reject_count:>10}"
            f"{acceptance_rate * 100:>11.2f}%"
            f"{avg_distance:>18.4f}"
            f"{threshold.threshold:>14.4f}"
        )


# ============================================================
# 3. Few-shot formal authentication test
# ============================================================

def run_few_shot_test(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    print()
    print("=" * 90)
    print(" Few-Shot Formal Door Access Test")
    print("=" * 90)

    print(
        f"\nRegistration samples : {REGISTRATION_COUNTS}"
    )

    print(
        f"Unknown test samples : {TEST_COUNT}"
    )

    print(
        f"Repeated splits      : {REPEATS}"
    )

    print(
        "\nBackground 不參與註冊，也不參與一般使用者 FAR。"
    )

    rng = np.random.default_rng(RANDOM_SEED)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results: dict[
        tuple[str, int],
        list[dict[str, float]]
    ] = {}

    for user in USERS:

        user_samples = get_user_samples(
            X,
            y,
            user,
        )

        if len(user_samples) < TEST_COUNT + max(REGISTRATION_COUNTS):
            raise ValueError(
                f"{user} only has {len(user_samples)} samples, "
                f"but at least "
                f"{TEST_COUNT + max(REGISTRATION_COUNTS)} "
                f"are required."
            )

        for registration_count in REGISTRATION_COUNTS:

            results[(user, registration_count)] = []

            for repeat in range(REPEATS):

                # ------------------------------------------------
                # Randomly split user's data
                # ------------------------------------------------

                indices = rng.permutation(
                    len(user_samples)
                )

                registration_indices = (
                    indices[:registration_count]
                )

                test_indices = (
                    indices[
                        registration_count:
                        registration_count + TEST_COUNT
                    ]
                )

                registration_samples = (
                    user_samples[registration_indices]
                )

                genuine_test_samples = (
                    user_samples[test_indices]
                )

                # ------------------------------------------------
                # Build template
                # ------------------------------------------------

                template, threshold, _ = build_template(
                    registration_samples,
                    user,
                )

                auth_system = AuthenticationSystem(
                    feature_names=ENGINEERED_FEATURE_ORDER
                )

                # ------------------------------------------------
                # Genuine test
                # ------------------------------------------------

                genuine_distances = []

                for sample in genuine_test_samples:

                    result = auth_system.authenticate(
                        template,
                        threshold,
                        sample[np.newaxis, :, :],
                    )

                    genuine_distances.append(
                        result.distance
                    )

                genuine_distances = np.asarray(
                    genuine_distances
                )

                genuine_accept = (
                    genuine_distances
                    <= threshold.threshold
                )

                gar = float(
                    np.mean(genuine_accept)
                )

                # ------------------------------------------------
                # Impostor test
                #
                # 其他「真正使用者」
                # ------------------------------------------------

                impostor_distances = []

                for other_user in USERS:

                    if other_user == user:
                        continue

                    other_samples = get_user_samples(
                        X,
                        y,
                        other_user,
                    )

                    # 每個 impostor 使用 TEST_COUNT 筆
                    # 避免某個使用者資料量影響比例

                    other_indices = rng.choice(
                        len(other_samples),
                        size=min(
                            TEST_COUNT,
                            len(other_samples)
                        ),
                        replace=False,
                    )

                    selected_samples = (
                        other_samples[other_indices]
                    )

                    for sample in selected_samples:

                        result = auth_system.authenticate(
                            template,
                            threshold,
                            sample[np.newaxis, :, :],
                        )

                        impostor_distances.append(
                            result.distance
                        )

                impostor_distances = np.asarray(
                    impostor_distances
                )

                impostor_accept = (
                    impostor_distances
                    <= threshold.threshold
                )

                far = float(
                    np.mean(impostor_accept)
                )

                # ------------------------------------------------
                # No-contact test
                # ------------------------------------------------

                no_contact_samples = get_user_samples(
                    X,
                    y,
                    NO_CONTACT_USER,
                )

                no_contact_indices = rng.choice(
                    len(no_contact_samples),
                    size=min(
                        TEST_COUNT,
                        len(no_contact_samples)
                    ),
                    replace=False,
                )

                selected_no_contact = (
                    no_contact_samples[
                        no_contact_indices
                    ]
                )

                no_contact_distances = []

                for sample in selected_no_contact:

                    result = auth_system.authenticate(
                        template,
                        threshold,
                        sample[np.newaxis, :, :],
                    )

                    no_contact_distances.append(
                        result.distance
                    )

                no_contact_distances = np.asarray(
                    no_contact_distances
                )

                no_contact_accept = (
                    no_contact_distances
                    <= threshold.threshold
                )

                no_contact_far = float(
                    np.mean(no_contact_accept)
                )

                # ------------------------------------------------
                # Save repeat result
                # ------------------------------------------------

                results[
                    (user, registration_count)
                ].append(
                    {
                        "gar": gar,
                        "frr": 1.0 - gar,
                        "far": far,
                        "no_contact_far": no_contact_far,
                        "threshold": float(
                            threshold.threshold
                        ),
                        "genuine_distance": float(
                            np.mean(genuine_distances)
                        ),
                        "impostor_distance": float(
                            np.mean(impostor_distances)
                        ),
                        "no_contact_distance": float(
                            np.mean(
                                no_contact_distances
                            )
                        ),
                    }
                )

    # ========================================================
    # Print detailed results
    # ========================================================

    for registration_count in REGISTRATION_COUNTS:

        print()
        print(
            f"\n========== "
            f"{registration_count} Registration Samples"
            f" =========="
        )

        print(
            f"{'User':<15}"
            f"{'GAR Mean':>12}"
            f"{'GAR Std':>12}"
            f"{'FRR Mean':>12}"
            f"{'FAR Mean':>12}"
            f"{'No-contact FAR':>18}"
            f"{'Threshold':>14}"
        )

        print("-" * 100)

        for user in USERS:

            rows = results[
                (user, registration_count)
            ]

            gar_values = np.array(
                [r["gar"] for r in rows]
            )

            frr_values = np.array(
                [r["frr"] for r in rows]
            )

            far_values = np.array(
                [r["far"] for r in rows]
            )

            no_contact_far_values = np.array(
                [r["no_contact_far"] for r in rows]
            )

            threshold_values = np.array(
                [r["threshold"] for r in rows]
            )

            print(
                f"{user:<15}"
                f"{np.mean(gar_values) * 100:>11.2f}%"
                f"{np.std(gar_values) * 100:>11.2f}%"
                f"{np.mean(frr_values) * 100:>11.2f}%"
                f"{np.mean(far_values) * 100:>11.2f}%"
                f"{np.mean(no_contact_far_values) * 100:>17.2f}%"
                f"{np.mean(threshold_values):>14.4f}"
            )


# ============================================================
# 4. Main
# ============================================================

def main() -> None:

    X, y = load_data()

    print()
    print("=" * 90)
    print(" Sequence Statistics Formal Door Access Benchmark")
    print("=" * 90)

    print(
        f"\nDataset shape : {X.shape}"
    )

    print(
        f"Registered users : {', '.join(USERS)}"
    )

    print(
        f"No-contact data : {NO_CONTACT_USER}"
    )

    # --------------------------------------------------------
    # Basic template comparison
    # --------------------------------------------------------

    run_template_authentication_test(
        X,
        y,
    )

    # --------------------------------------------------------
    # No-contact rejection
    # --------------------------------------------------------

    run_no_contact_test(
        X,
        y,
    )

    # --------------------------------------------------------
    # Formal few-shot test
    # --------------------------------------------------------

    run_few_shot_test(
        X,
        y,
    )

    print()
    print("=" * 90)
    print(" Benchmark Finished")
    print("=" * 90)


if __name__ == "__main__":
    main()