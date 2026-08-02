from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER

from .common import (
    BenchmarkRecord,
    calculate_far_frr_acceptance,
    compute_euclidean_distances,
    find_best_threshold,
    load_benchmark_dataset,
    prepare_companion_vectors,
    prepare_registration_vectors,
    select_user_samples,
    summarize_records,
    DEFAULT_USERS,
)


@dataclass(frozen=True)
class ThresholdExperimentConfig:
    users: Sequence[str] = DEFAULT_USERS
    registration_count: int = 10
    k_value: float = 2.0
    feature_names: Sequence[str] = ENGINEERED_FEATURE_ORDER


def run_threshold_strategy_experiment(
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

        template_vector = np.mean(registration_vectors, axis=0).astype(np.float32)
        registration_distances = compute_euclidean_distances(registration_vectors, template_vector)
        genuine_distances = compute_euclidean_distances(validation_genuine_vectors, template_vector)
        impostor_distances = compute_euclidean_distances(impostor_vectors, template_vector)

        baseline_threshold = float(np.mean(registration_distances) + k_value * np.std(registration_distances))
        baseline_far, baseline_frr, baseline_acceptance = calculate_far_frr_acceptance(
            genuine_distances,
            impostor_distances,
            baseline_threshold,
        )

        records.append(
            BenchmarkRecord(
                user_id=user_id,
                method="Threshold Strategy Comparison",
                template="Centroid",
                distance="Euclidean",
                threshold_strategy="Mean + k × Std",
                threshold=baseline_threshold,
                far=baseline_far,
                frr=baseline_frr,
                acceptance_rate=baseline_acceptance,
                average_genuine_distance=float(np.mean(genuine_distances)),
                average_impostor_distance=float(np.mean(impostor_distances)),
            )
        )

        sweep_threshold, sweep_far, sweep_frr = find_best_threshold(genuine_distances, impostor_distances)
        sweep_far, sweep_frr, sweep_acceptance = calculate_far_frr_acceptance(
            genuine_distances,
            impostor_distances,
            sweep_threshold,
        )

        records.append(
            BenchmarkRecord(
                user_id=user_id,
                method="Threshold Strategy Comparison",
                template="Centroid",
                distance="Euclidean",
                threshold_strategy="Threshold Sweep",
                threshold=sweep_threshold,
                far=sweep_far,
                frr=sweep_frr,
                acceptance_rate=sweep_acceptance,
                average_genuine_distance=float(np.mean(genuine_distances)),
                average_impostor_distance=float(np.mean(impostor_distances)),
            )
        )

    return records


def run_threshold_strategy_experiment_summary(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
    k_value: float = 2.0,
    feature_names: Sequence[str] = ENGINEERED_FEATURE_ORDER,
) -> list[BenchmarkRecord]:
    return summarize_records(
        run_threshold_strategy_experiment(
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
    summary = run_threshold_strategy_experiment_summary(X, raw_y)
    from .common import format_records_table

    print("\n========== Threshold Strategy Benchmark ==========")
    print(format_records_table(summary))


if __name__ == "__main__":
    main()
