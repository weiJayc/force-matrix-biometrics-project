from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Iterable

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.authentication import AuthenticationSystem
from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER
from authentication.registration import RegistrationSystem
from authentication.template import TemplateManager
from authentication.threshold import ThresholdManager
from data_loader import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
DEFAULT_USERS = ("amber", "jay", "666", "background")


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    return load_dataset(DATA_DIR)


def _get_user_samples(X: np.ndarray, raw_y: np.ndarray, user_id: str) -> np.ndarray:
    mask = np.asarray(raw_y) == user_id
    return X[mask]


def _get_other_samples(X: np.ndarray, raw_y: np.ndarray, user_id: str) -> np.ndarray:
    mask = np.asarray(raw_y) != user_id
    return X[mask]


def build_template_for_user(
    X: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
    registration_count: int | None = None,
) -> tuple[object, object, np.ndarray]:
    user_samples = _get_user_samples(X, raw_y, user_id)
    if registration_count is not None:
        registration_samples = user_samples[:registration_count]
    else:
        registration_samples = user_samples

    with tempfile.TemporaryDirectory() as temp_dir:
        template_manager = TemplateManager(storage_dir=Path(temp_dir) / "templates")
        threshold_manager = ThresholdManager(storage_dir=Path(temp_dir) / "thresholds")
        registration_system = RegistrationSystem(
            template_manager=template_manager,
            threshold_manager=threshold_manager,
            feature_names=ENGINEERED_FEATURE_ORDER,
        )
        template, threshold, distances = registration_system.register_user(
            registration_samples,
            user_id=user_id,
            return_details=True,
        )

    return template, threshold, distances


def evaluate_template_authentication(
    X: np.ndarray,
    raw_y: np.ndarray,
    template_user: str,
    test_users: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    template, threshold, _ = build_template_for_user(X, raw_y, template_user)
    auth_system = AuthenticationSystem(feature_names=ENGINEERED_FEATURE_ORDER)

    if test_users is None:
        test_users = DEFAULT_USERS

    results: list[dict[str, object]] = []
    for test_user in test_users:
        test_samples = _get_user_samples(X, raw_y, test_user)
        distances: list[float] = []
        accept_count = 0
        reject_count = 0

        for sample in test_samples:
            result = auth_system.authenticate(template, threshold, sample[np.newaxis, :, :])
            distances.append(result.distance)
            if result.accept:
                accept_count += 1
            else:
                reject_count += 1

        acceptance_rate = accept_count / len(test_samples) if len(test_samples) else 0.0
        results.append(
            {
                "template_user": template_user,
                "test_user": test_user,
                "accept": accept_count,
                "reject": reject_count,
                "acceptance_rate": acceptance_rate,
                "average_distance": float(np.mean(distances)) if distances else 0.0,
            }
        )

    return results


def evaluate_user_thresholds(X: np.ndarray, raw_y: np.ndarray, users: Iterable[str] | None = None) -> list[dict[str, object]]:
    if users is None:
        users = DEFAULT_USERS

    rows: list[dict[str, object]] = []
    for user_id in users:
        template, threshold, distances = build_template_for_user(X, raw_y, user_id)
        rows.append(
            {
                "user": user_id,
                "threshold": float(threshold.threshold),
                "mean_distance": float(np.mean(distances)),
                "std": float(np.std(distances)),
            }
        )
    return rows


def run_few_shot_experiment(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Iterable[str] | None = None,
    registration_counts: Iterable[int] = (5, 10, 20, 50),
) -> list[dict[str, object]]:
    if users is None:
        users = DEFAULT_USERS

    rows: list[dict[str, object]] = []
    auth_system = AuthenticationSystem(feature_names=ENGINEERED_FEATURE_ORDER)

    for user_id in users:
        genuine_samples = _get_user_samples(X, raw_y, user_id)
        impostor_samples = _get_other_samples(X, raw_y, user_id)

        for count in registration_counts:
            template, threshold, _ = build_template_for_user(X, raw_y, user_id, registration_count=count)
            genuine_distances: list[float] = []
            impostor_distances: list[float] = []

            for sample in genuine_samples:
                result = auth_system.authenticate(template, threshold, sample[np.newaxis, :, :])
                genuine_distances.append(result.distance)

            for sample in impostor_samples:
                result = auth_system.authenticate(template, threshold, sample[np.newaxis, :, :])
                impostor_distances.append(result.distance)

            genuine_array = np.asarray(genuine_distances, dtype=np.float32)
            impostor_array = np.asarray(impostor_distances, dtype=np.float32)
            far = float(np.mean(impostor_array <= threshold.threshold))
            frr = float(np.mean(genuine_array > threshold.threshold))
            acceptance_rate = float(np.mean(genuine_array <= threshold.threshold))

            rows.append(
                {
                    "user": user_id,
                    "registration_samples": count,
                    "threshold": float(threshold.threshold),
                    "far": far,
                    "frr": frr,
                    "acceptance_rate": acceptance_rate,
                }
            )

    return rows


def print_template_results(results: list[dict[str, object]]) -> None:
    print("\n========== Registration -> Authentication Summary ==========")
    print(f"{'Template User':<14} {'Test User':<14} {'Accept':<8} {'Reject':<8} {'Acceptance Rate':<16} {'Average Distance':<18}")
    print("-" * 100)
    for row in results:
        print(
            f"{str(row['template_user']):<14} {str(row['test_user']):<14} {int(row['accept']):<8} {int(row['reject']):<8} {row['acceptance_rate'] * 100:>6.2f}% {float(row['average_distance']):>12.4f}"
        )


def print_user_thresholds(rows: list[dict[str, object]]) -> None:
    print("\n========== Per-User Threshold Summary ==========")
    print(f"{'User':<14} {'Threshold':<12} {'Mean Distance':<14} {'Std':<10}")
    print("-" * 60)
    for row in rows:
        print(f"{str(row['user']):<14} {float(row['threshold']):<12.4f} {float(row['mean_distance']):<14.4f} {float(row['std']):<10.4f}")


def print_few_shot_results(rows: list[dict[str, object]]) -> None:
    print("\n========== Few-Shot Registration Experiment ==========")
    print(f"{'User':<14} {'Registration Samples':<22} {'Threshold':<12} {'FAR':<10} {'FRR':<10} {'Acceptance Rate':<18}")
    print("-" * 100)
    for row in rows:
        print(
            f"{str(row['user']):<14} {int(row['registration_samples']):<22} {float(row['threshold']):<12.4f} {float(row['far']):<10.4f} {float(row['frr']):<10.4f} {float(row['acceptance_rate']):<18.4f}"
        )


def main() -> None:
    X, raw_y = _load_data()

    for template_user in DEFAULT_USERS:
        results = evaluate_template_authentication(X, raw_y, template_user, test_users=DEFAULT_USERS)
        print_template_results(results)

    threshold_rows = evaluate_user_thresholds(X, raw_y, users=DEFAULT_USERS)
    print_user_thresholds(threshold_rows)

    few_shot_rows = run_few_shot_experiment(X, raw_y, users=DEFAULT_USERS)
    print_few_shot_results(few_shot_rows)


if __name__ == "__main__":
    main()
