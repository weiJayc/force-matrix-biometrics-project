from __future__ import annotations

from pathlib import Path
import sys
import tempfile

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

from AI.authentication.legacy.authentication import AuthenticationSystem
from AI.authentication.legacy.feature_extractor import ENGINEERED_FEATURE_ORDER
from AI.authentication.legacy.registration import RegistrationSystem
from AI.authentication.legacy.template import TemplateManager
from AI.authentication.legacy.threshold import ThresholdManager


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

# 正式測試
REGISTRATION_COUNTS = (
    5,
    10,
    20,
)

TEST_COUNT = 30
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
# Build Template
# ============================================================

def build_template(
    registration_samples: np.ndarray,
    user_id: str,
):
    """
    只使用 registration_samples 建立 template + threshold。

    注意：
    不會偷看 genuine test、
    impostor test、
    background。
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

        template, threshold, registration_distances = (
            registration_system.register_user(
                registration_samples,
                user_id=user_id,
                return_details=True,
            )
        )

    return (
        template,
        threshold,
        registration_distances,
    )


# ============================================================
# Authenticate ONE sample
# ============================================================

def authenticate_sample(
    auth_system: AuthenticationSystem,
    template,
    threshold,
    sample: np.ndarray,
):
    return auth_system.authenticate(
        template,
        threshold,
        sample[np.newaxis, :, :],
    )


# ============================================================
# Evaluate samples
# ============================================================

def evaluate_samples(
    auth_system: AuthenticationSystem,
    template,
    threshold,
    samples: np.ndarray,
) -> dict[str, float]:

    distances = []

    accept_count = 0

    for sample in samples:

        result = authenticate_sample(
            auth_system,
            template,
            threshold,
            sample,
        )

        distances.append(result.distance)

        if result.accept:
            accept_count += 1

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    total = len(samples)

    acceptance_rate = (
        accept_count / total
        if total > 0
        else 0.0
    )

    return {
        "accept": float(accept_count),

        "reject": float(total - accept_count),

        "rate": float(acceptance_rate),

        "average_distance": (
            float(np.mean(distances))
            if len(distances) > 0
            else 0.0
        ),
    }


# ============================================================
# One complete experiment
# ============================================================

def run_one_experiment(
    X: np.ndarray,
    y: np.ndarray,
    template_user: str,
    registration_count: int,
    repeat_index: int,
    rng: np.random.Generator,
) -> dict:

    # --------------------------------------------------------
    # Get template user's data
    # --------------------------------------------------------

    genuine_pool = get_user_samples(
        X,
        y,
        template_user,
    )

    required = registration_count + TEST_COUNT

    if len(genuine_pool) < required:

        raise ValueError(
            f"{template_user} has {len(genuine_pool)} samples, "
            f"but needs {required}."
        )

    # --------------------------------------------------------
    # Random split
    #
    # IMPORTANT:
    # registration and genuine test NEVER overlap
    # --------------------------------------------------------

    shuffled_indices = rng.permutation(
        len(genuine_pool)
    )

    registration_indices = (
        shuffled_indices[:registration_count]
    )

    genuine_indices = (
        shuffled_indices[
            registration_count:
            registration_count + TEST_COUNT
        ]
    )

    registration_samples = (
        genuine_pool[registration_indices]
    )

    genuine_samples = (
        genuine_pool[genuine_indices]
    )

    # --------------------------------------------------------
    # Build template
    # --------------------------------------------------------

    template, threshold, registration_distances = (
        build_template(
            registration_samples,
            template_user,
        )
    )

    auth_system = AuthenticationSystem(
        feature_names=ENGINEERED_FEATURE_ORDER
    )

    # ========================================================
    # Genuine
    # ========================================================

    genuine_result = evaluate_samples(
        auth_system,
        template,
        threshold,
        genuine_samples,
    )

    # ========================================================
    # Impostor
    #
    # 每一個其他使用者取 30 筆
    # ========================================================

    impostor_results = {}

    all_impostor_distances = []
    all_impostor_accepts = []

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
                f"{impostor_user} does not have "
                f"{TEST_COUNT} samples."
            )

        # 每次 repeat 都重新抽 30 筆
        impostor_indices = rng.choice(
            len(impostor_pool),
            size=TEST_COUNT,
            replace=False,
        )

        impostor_samples = (
            impostor_pool[impostor_indices]
        )

        result = evaluate_samples(
            auth_system,
            template,
            threshold,
            impostor_samples,
        )

        impostor_results[impostor_user] = result

        # 收集所有 impostor
        all_impostor_distances.append(
            result["average_distance"]
        )

        all_impostor_accepts.append(
            result["accept"] / TEST_COUNT
        )

    # --------------------------------------------------------
    # Overall FAR
    #
    # 所有其他使用者的測試樣本一起計算
    # --------------------------------------------------------

    total_impostor_accept = sum(
        result["accept"]
        for result in impostor_results.values()
    )

    total_impostor_samples = (
        len(impostor_results) * TEST_COUNT
    )

    far = (
        total_impostor_accept
        / total_impostor_samples
    )

    # ========================================================
    # No-contact
    # ========================================================

    no_contact_pool = get_user_samples(
        X,
        y,
        NO_CONTACT_USER,
    )

    if len(no_contact_pool) < TEST_COUNT:

        raise ValueError(
            f"{NO_CONTACT_USER} does not have "
            f"{TEST_COUNT} samples."
        )

    no_contact_indices = rng.choice(
        len(no_contact_pool),
        size=TEST_COUNT,
        replace=False,
    )

    no_contact_samples = (
        no_contact_pool[no_contact_indices]
    )

    no_contact_result = evaluate_samples(
        auth_system,
        template,
        threshold,
        no_contact_samples,
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
        # ------------------------------------
        # Identity
        # ------------------------------------

        "template_user": template_user,

        "registration_count": registration_count,

        "repeat": repeat_index,

        # ------------------------------------
        # Threshold
        # ------------------------------------

        "threshold": float(
            threshold.threshold
        ),

        # ------------------------------------
        # Genuine
        # ------------------------------------

        "gar": gar,

        "frr": frr,

        "genuine_accept": genuine_result[
            "accept"
        ],

        "genuine_reject": genuine_result[
            "reject"
        ],

        "genuine_distance": genuine_result[
            "average_distance"
        ],

        # ------------------------------------
        # Impostor
        # ------------------------------------

        "far": far,

        "impostor_distance": float(
            np.mean(all_impostor_distances)
        ),

        # ------------------------------------
        # No-contact
        # ------------------------------------

        "no_contact_far": no_contact_far,

        "no_contact_distance": (
            no_contact_result[
                "average_distance"
            ]
        ),

        # ------------------------------------
        # Template -> Test User
        # ------------------------------------

        "impostor_results": impostor_results,
    }


# ============================================================
# Run ALL experiments
# ============================================================

def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict]:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    all_results = []

    for registration_count in REGISTRATION_COUNTS:

        print()
        print("=" * 100)
        print(
            f"Registration = {registration_count} "
            f"| Test = {TEST_COUNT} "
            f"| Repeats = {REPEATS}"
        )
        print("=" * 100)

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

                all_results.append(result)

    return all_results


# ============================================================
# Print Template -> Test User
# ============================================================

def print_template_results(
    results: list[dict],
) -> None:

    print()
    print("=" * 100)
    print(
        "Template -> Test User"
    )
    print("=" * 100)

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

    print("-" * 100)

    for row in results:

        template_user = row[
            "template_user"
        ]

        impostor_results = row[
            "impostor_results"
        ]

        # Genuine
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

        # Impostors
        for test_user, test_result in (
            impostor_results.items()
        ):

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


# ============================================================
# Print summary
# ============================================================

def print_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 110)
    print(
        "FINAL FORMAL DOOR ACCESS SUMMARY"
    )
    print("=" * 110)

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

    print("-" * 110)

    for registration_count in REGISTRATION_COUNTS:

        for user in USERS:

            rows = [
                r
                for r in results
                if (
                    r["registration_count"]
                    == registration_count
                    and
                    r["template_user"]
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
# Print Distance Summary
# ============================================================

def print_distance_summary(
    results: list[dict],
) -> None:

    print()
    print("=" * 100)
    print(
        "Distance Summary"
    )
    print("=" * 100)

    print(
        f"{'Reg':<6}"
        f"{'User':<15}"
        f"{'Genuine Distance':>20}"
        f"{'Impostor Distance':>20}"
        f"{'No-contact Distance':>22}"
    )

    print("-" * 100)

    for registration_count in REGISTRATION_COUNTS:

        for user in USERS:

            rows = [
                r
                for r in results
                if (
                    r["registration_count"]
                    == registration_count
                    and
                    r["template_user"]
                    == user
                )
            ]

            genuine_distance = np.mean([
                r["genuine_distance"]
                for r in rows
            ])

            impostor_distance = np.mean([
                r["impostor_distance"]
                for r in rows
            ])

            no_contact_distance = np.mean([
                r["no_contact_distance"]
                for r in rows
            ])

            print(
                f"{registration_count:<6}"
                f"{user:<15}"
                f"{genuine_distance:>20.4f}"
                f"{impostor_distance:>20.4f}"
                f"{no_contact_distance:>22.4f}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    X, y = load_data()

    print()
    print("=" * 100)
    print(
        "Sequence Statistics "
        "Formal Door Access Benchmark"
    )
    print("=" * 100)

    print(
        f"Dataset shape : {X.shape}"
    )

    print(
        f"Users : {USERS}"
    )

    print(
        f"No-contact : {NO_CONTACT_USER}"
    )

    print(
        f"Registration : {REGISTRATION_COUNTS}"
    )

    print(
        f"Unknown genuine test : {TEST_COUNT}"
    )

    print(
        f"Repeats : {REPEATS}"
    )

    # ========================================================
    # EVERYTHING uses SAME experiment algorithm
    # ========================================================

    results = run_benchmark(
        X,
        y,
    )

    # ========================================================
    # Detailed template -> test user
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
    # Distance
    # ========================================================

    print_distance_summary(
        results
    )

    print()
    print("=" * 100)
    print(
        "Benchmark Finished"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()