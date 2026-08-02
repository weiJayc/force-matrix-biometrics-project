from __future__ import annotations

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
    load_benchmark_dataset,
    prepare_companion_vectors,
    prepare_registration_vectors,
    select_user_samples,
    summarize_records,
    DEFAULT_USERS,
)


def build_centroid_template(registration_vectors: np.ndarray) -> np.ndarray:
    return np.mean(registration_vectors, axis=0).astype(np.float32)


def build_robust_template(registration_vectors: np.ndarray, remove_fraction: float = 0.1) -> np.ndarray:
    initial_template = build_centroid_template(registration_vectors)
    distances = compute_euclidean_distances(registration_vectors, initial_template)

    remove_count = int(np.ceil(len(registration_vectors) * remove_fraction))
    remove_count = min(max(remove_count, 1), len(registration_vectors) - 1)
    keep_indices = np.argsort(distances)[: len(registration_vectors) - remove_count]
    return np.mean(registration_vectors[keep_indices], axis=0).astype(np.float32)


def build_median_template(registration_vectors: np.ndarray) -> np.ndarray:
    return np.median(registration_vectors, axis=0).astype(np.float32)


def _evaluate_template_strategy(
    user_id: str,
    template_name: str,
    template_vector: np.ndarray,
    registration_vectors: np.ndarray,
    genuine_vectors: np.ndarray,
    impostor_vectors: np.ndarray,
    k_value: float,
) -> BenchmarkRecord:
    registration_distances = compute_euclidean_distances(registration_vectors, template_vector)
    genuine_distances = compute_euclidean_distances(genuine_vectors, template_vector)
    impostor_distances = compute_euclidean_distances(impostor_vectors, template_vector)

    threshold = float(np.mean(registration_distances) + k_value * np.std(registration_distances))
    far, frr, acceptance_rate = calculate_far_frr_acceptance(genuine_distances, impostor_distances, threshold)

    return BenchmarkRecord(
        user_id=user_id,
        method="Template Strategy Comparison",
        template=template_name,
        distance="Euclidean",
        threshold_strategy="Mean + k × Std",
        threshold=threshold,
        far=far,
        frr=frr,
        acceptance_rate=acceptance_rate,
        average_genuine_distance=float(np.mean(genuine_distances)),
        average_impostor_distance=float(np.mean(impostor_distances)),
    )


def run_template_strategy_experiment(
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

        centroid_template = build_centroid_template(registration_vectors)
        robust_template = build_robust_template(registration_vectors)
        median_template = build_median_template(registration_vectors)

        records.append(
            _evaluate_template_strategy(
                user_id,
                "Centroid",
                centroid_template,
                registration_vectors,
                validation_genuine_vectors,
                impostor_vectors,
                k_value,
            )
        )
        records.append(
            _evaluate_template_strategy(
                user_id,
                "Robust Template",
                robust_template,
                registration_vectors,
                validation_genuine_vectors,
                impostor_vectors,
                k_value,
            )
        )
        records.append(
            _evaluate_template_strategy(
                user_id,
                "Median Template",
                median_template,
                registration_vectors,
                validation_genuine_vectors,
                impostor_vectors,
                k_value,
            )
        )

    return records


def run_template_strategy_experiment_summary(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
    k_value: float = 2.0,
    feature_names: Sequence[str] = ENGINEERED_FEATURE_ORDER,
) -> list[BenchmarkRecord]:
    return summarize_records(
        run_template_strategy_experiment(
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
    summary = run_template_strategy_experiment_summary(X, raw_y)
    from .common import format_records_table

    print("\n========== Template Strategy Benchmark ==========")
    print(format_records_table(summary))


if __name__ == "__main__":
    main()
