from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from .common import (
    DEFAULT_USERS,
    FEATURE_NAMES,
    NormalizationBenchmarkRecord,
    UserNormalizationStats,
    calculate_far_frr_acceptance,
    compute_cosine_distances,
    compute_euclidean_distances,
    compute_roc_metrics,
    compute_zero_range_count,
    extract_flattened_vectors,
    format_benchmark_table,
    format_zero_range_table,
    load_benchmark_dataset,
    normalize_minmax,
    normalize_robust,
    normalize_zscore,
    select_user_samples,
)


@dataclass(frozen=True)
class NormalizationBenchmarkConfig:
    users: Sequence[str] = DEFAULT_USERS
    registration_count: int = 10


def _build_global_registration_pool(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Sequence[str],
    registration_count: int,
) -> np.ndarray:
    registration_vectors = []
    for user_id in users:
        registration_samples, _, _ = select_user_samples(X, raw_y, user_id, registration_count)
        registration_vectors.append(extract_flattened_vectors(registration_samples, feature_names=FEATURE_NAMES))
    return np.concatenate(registration_vectors, axis=0).astype(np.float32)


def _apply_normalization(
    strategy: str,
    registration_vectors: np.ndarray,
    validation_genuine_vectors: np.ndarray,
    impostor_vectors: np.ndarray,
    global_registration_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "Per-user MinMax":
        stats_min = registration_vectors.min(axis=0)
        stats_max = registration_vectors.max(axis=0)
        return (
            normalize_minmax(registration_vectors, stats_min, stats_max),
            normalize_minmax(validation_genuine_vectors, stats_min, stats_max),
            normalize_minmax(impostor_vectors, stats_min, stats_max),
        )

    if strategy == "Global MinMax":
        stats_min = global_registration_pool.min(axis=0)
        stats_max = global_registration_pool.max(axis=0)
        return (
            normalize_minmax(registration_vectors, stats_min, stats_max),
            normalize_minmax(validation_genuine_vectors, stats_min, stats_max),
            normalize_minmax(impostor_vectors, stats_min, stats_max),
        )

    if strategy == "Z-score":
        mean = global_registration_pool.mean(axis=0)
        std = global_registration_pool.std(axis=0)
        return (
            normalize_zscore(registration_vectors, mean, std),
            normalize_zscore(validation_genuine_vectors, mean, std),
            normalize_zscore(impostor_vectors, mean, std),
        )

    if strategy == "RobustScaler":
        median = np.median(global_registration_pool, axis=0)
        q25 = np.percentile(global_registration_pool, 25, axis=0)
        q75 = np.percentile(global_registration_pool, 75, axis=0)
        iqr = q75 - q25
        return (
            normalize_robust(registration_vectors, median, iqr),
            normalize_robust(validation_genuine_vectors, median, iqr),
            normalize_robust(impostor_vectors, median, iqr),
        )

    raise ValueError(f"Unknown normalization strategy: {strategy}")


def _evaluate_distance_metric(
    user_id: str,
    normalization: str,
    distance_name: str,
    registration_vectors: np.ndarray,
    validation_genuine_vectors: np.ndarray,
    impostor_vectors: np.ndarray,
) -> tuple[NormalizationBenchmarkRecord, UserNormalizationStats]:
    template_vector = np.mean(registration_vectors, axis=0).astype(np.float32)

    if distance_name == "Euclidean":
        genuine_distances = compute_euclidean_distances(validation_genuine_vectors, template_vector)
        impostor_distances = compute_euclidean_distances(impostor_vectors, template_vector)
    elif distance_name == "Cosine":
        genuine_distances = compute_cosine_distances(validation_genuine_vectors, template_vector)
        impostor_distances = compute_cosine_distances(impostor_vectors, template_vector)
    else:
        raise ValueError(f"Unknown distance metric: {distance_name}")

    auc, eer, best_threshold, best_far, best_frr, best_acceptance_rate, _, _, _ = compute_roc_metrics(
        genuine_distances,
        impostor_distances,
    )

    record = NormalizationBenchmarkRecord(
        user_id=user_id,
        normalization=normalization,
        distance=distance_name,
        threshold=best_threshold,
        far=best_far,
        frr=best_frr,
        acceptance_rate=best_acceptance_rate,
        auc=auc,
        eer=eer,
        average_genuine_distance=float(np.mean(genuine_distances)),
        average_impostor_distance=float(np.mean(impostor_distances)),
    )

    stats = UserNormalizationStats(
        user_id=user_id,
        zero_range_features=0,
        zero_range_ratio=0.0,
        euclidean_mean_genuine=float(np.mean(compute_euclidean_distances(validation_genuine_vectors, template_vector))),
        euclidean_mean_impostor=float(np.mean(compute_euclidean_distances(impostor_vectors, template_vector))),
        euclidean_max_impostor=float(np.max(compute_euclidean_distances(impostor_vectors, template_vector))),
    )

    return record, stats


def _aggregate_benchmark_records(
    distance_pool: dict[tuple[str, str], dict[str, list[np.ndarray]]],
) -> list[NormalizationBenchmarkRecord]:
    records: list[NormalizationBenchmarkRecord] = []

    for (normalization, distance_name), grouped in sorted(distance_pool.items()):
        genuine_distances = np.concatenate(grouped["genuine"]) if grouped["genuine"] else np.asarray([], dtype=np.float32)
        impostor_distances = np.concatenate(grouped["impostor"]) if grouped["impostor"] else np.asarray([], dtype=np.float32)

        auc, eer, best_threshold, best_far, best_frr, best_acceptance_rate, _, _, _ = compute_roc_metrics(
            genuine_distances,
            impostor_distances,
        )

        records.append(
            NormalizationBenchmarkRecord(
                user_id="all",
                normalization=normalization,
                distance=distance_name,
                threshold=best_threshold,
                far=best_far,
                frr=best_frr,
                acceptance_rate=best_acceptance_rate,
                auc=auc,
                eer=eer,
                average_genuine_distance=float(np.mean(genuine_distances)),
                average_impostor_distance=float(np.mean(impostor_distances)),
            )
        )

    return records


def run_normalization_benchmark(
    X: np.ndarray | None = None,
    raw_y: np.ndarray | None = None,
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
) -> tuple[list[NormalizationBenchmarkRecord], list[UserNormalizationStats]]:
    if X is None or raw_y is None:
        X, raw_y = load_benchmark_dataset()

    global_registration_pool = _build_global_registration_pool(X, raw_y, users, registration_count)

    user_stats: list[UserNormalizationStats] = []
    distance_pool: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}

    for user_id in users:
        registration_samples, validation_genuine_samples, impostor_samples = select_user_samples(
            X,
            raw_y,
            user_id,
            registration_count,
        )

        raw_registration_vectors = extract_flattened_vectors(registration_samples, feature_names=FEATURE_NAMES)
        raw_validation_genuine_vectors = extract_flattened_vectors(validation_genuine_samples, feature_names=FEATURE_NAMES)
        raw_impostor_vectors = extract_flattened_vectors(impostor_samples, feature_names=FEATURE_NAMES)

        zero_range_count = compute_zero_range_count(raw_registration_vectors)
        zero_range_ratio = zero_range_count / raw_registration_vectors.shape[1]

        strategy_stats = {
            "Per-user MinMax": _apply_normalization(
                "Per-user MinMax",
                raw_registration_vectors,
                raw_validation_genuine_vectors,
                raw_impostor_vectors,
                global_registration_pool,
            ),
            "Global MinMax": _apply_normalization(
                "Global MinMax",
                raw_registration_vectors,
                raw_validation_genuine_vectors,
                raw_impostor_vectors,
                global_registration_pool,
            ),
            "Z-score": _apply_normalization(
                "Z-score",
                raw_registration_vectors,
                raw_validation_genuine_vectors,
                raw_impostor_vectors,
                global_registration_pool,
            ),
            "RobustScaler": _apply_normalization(
                "RobustScaler",
                raw_registration_vectors,
                raw_validation_genuine_vectors,
                raw_impostor_vectors,
                global_registration_pool,
            ),
        }

        for normalization, (registration_vectors, genuine_vectors, impostor_vectors) in strategy_stats.items():
            for distance_name in ("Euclidean", "Cosine"):
                template_vector = np.mean(registration_vectors, axis=0).astype(np.float32)

                if distance_name == "Euclidean":
                    genuine_distances = compute_euclidean_distances(genuine_vectors, template_vector)
                    impostor_distances = compute_euclidean_distances(impostor_vectors, template_vector)
                elif distance_name == "Cosine":
                    genuine_distances = compute_cosine_distances(genuine_vectors, template_vector)
                    impostor_distances = compute_cosine_distances(impostor_vectors, template_vector)
                else:
                    raise ValueError(f"Unknown distance metric: {distance_name}")

                distance_pool.setdefault((normalization, distance_name), {"genuine": [], "impostor": []})
                distance_pool[(normalization, distance_name)]["genuine"].append(np.asarray(genuine_distances, dtype=np.float32))
                distance_pool[(normalization, distance_name)]["impostor"].append(np.asarray(impostor_distances, dtype=np.float32))

        per_user_registration_min = raw_registration_vectors.min(axis=0)
        per_user_registration_max = raw_registration_vectors.max(axis=0)
        per_user_registration_vectors = normalize_minmax(raw_registration_vectors, per_user_registration_min, per_user_registration_max)
        per_user_genuine_vectors = normalize_minmax(raw_validation_genuine_vectors, per_user_registration_min, per_user_registration_max)
        per_user_impostor_vectors = normalize_minmax(raw_impostor_vectors, per_user_registration_min, per_user_registration_max)
        per_user_template = np.mean(per_user_registration_vectors, axis=0).astype(np.float32)
        euclidean_genuine = compute_euclidean_distances(per_user_genuine_vectors, per_user_template)
        euclidean_impostor = compute_euclidean_distances(per_user_impostor_vectors, per_user_template)

        user_stats.append(
            UserNormalizationStats(
                user_id=user_id,
                zero_range_features=zero_range_count,
                zero_range_ratio=zero_range_ratio,
                euclidean_mean_genuine=float(np.mean(euclidean_genuine)),
                euclidean_mean_impostor=float(np.mean(euclidean_impostor)),
                euclidean_max_impostor=float(np.max(euclidean_impostor)),
            )
        )

    return _aggregate_benchmark_records(distance_pool), user_stats


def main() -> None:
    records, user_stats = run_normalization_benchmark()
    print("\n========== Normalization Benchmark ==========")
    print(format_benchmark_table(records))
    print("\n========== Zero-Range Features ==========")
    print(format_zero_range_table(user_stats))


if __name__ == "__main__":
    main()
