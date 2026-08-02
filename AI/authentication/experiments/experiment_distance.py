from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER

from .common import (
    BenchmarkRecord,
    calculate_far_frr_acceptance,
    compute_cosine_distances,
    compute_euclidean_distances,
    load_benchmark_dataset,
    prepare_companion_vectors,
    prepare_registration_vectors,
    select_user_samples,
    summarize_records,
    DEFAULT_USERS,
)


def _evaluate_distance_strategy(
    user_id: str,
    distance_name: str,
    distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    registration_vectors: np.ndarray,
    genuine_vectors: np.ndarray,
    impostor_vectors: np.ndarray,
    k_value: float,
) -> BenchmarkRecord:
    template_vector = np.mean(registration_vectors, axis=0).astype(np.float32)
    registration_distances = distance_function(registration_vectors, template_vector)
    genuine_distances = distance_function(genuine_vectors, template_vector)
    impostor_distances = distance_function(impostor_vectors, template_vector)

    threshold = float(np.mean(registration_distances) + k_value * np.std(registration_distances))
    far, frr, acceptance_rate = calculate_far_frr_acceptance(genuine_distances, impostor_distances, threshold)

    return BenchmarkRecord(
        user_id=user_id,
        method="Distance Metric Comparison",
        template="Centroid",
        distance=distance_name,
        threshold_strategy="Mean + k × Std",
        threshold=threshold,
        far=far,
        frr=frr,
        acceptance_rate=acceptance_rate,
        average_genuine_distance=float(np.mean(genuine_distances)),
        average_impostor_distance=float(np.mean(impostor_distances)),
    )


def run_distance_metric_experiment(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
    k_value: float = 2.0,
    feature_names: Sequence[str] = ENGINEERED_FEATURE_ORDER,
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []

    for user_id in users:
        registration_samples, validation_genuine_samples, impostor_samples = select_user_samples(
            X,
            raw_y,
            user_id,
            registration_count=registration_count,
        )

        registration_vectors, sensor_min, sensor_max = prepare_registration_vectors(
            registration_samples,
            feature_names=feature_names,
        )
        validation_genuine_vectors = prepare_companion_vectors(
            validation_genuine_samples,
            sensor_min,
            sensor_max,
            feature_names=feature_names,
        )
        impostor_vectors = prepare_companion_vectors(
            impostor_samples,
            sensor_min,
            sensor_max,
            feature_names=feature_names,
        )

        records.append(
            _evaluate_distance_strategy(
                user_id,
                "Euclidean",
                compute_euclidean_distances,
                registration_vectors,
                validation_genuine_vectors,
                impostor_vectors,
                k_value,
            )
        )
        records.append(
            _evaluate_distance_strategy(
                user_id,
                "Cosine",
                compute_cosine_distances,
                registration_vectors,
                validation_genuine_vectors,
                impostor_vectors,
                k_value,
            )
        )

    return records


def run_distance_metric_experiment_summary(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
    k_value: float = 2.0,
    feature_names: Sequence[str] = ENGINEERED_FEATURE_ORDER,
) -> list[BenchmarkRecord]:
    return summarize_records(
        run_distance_metric_experiment(
            X,
            raw_y,
            users=users,
            registration_count=registration_count,
            k_value=k_value,
            feature_names=feature_names,
        )
    )


def main() -> None:
    X, raw_y = load_benchmark_dataset()
    summary = run_distance_metric_experiment_summary(X, raw_y)
    from .common import format_records_table

    print("\n========== Distance Metric Benchmark ==========")
    print(format_records_table(summary))


if __name__ == "__main__":
    main()
