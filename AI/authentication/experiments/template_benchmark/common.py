from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.experiments.normalization_benchmark.common import (
    DEFAULT_USERS,
    FEATURE_NAMES,
    apply_feature_mask,
    build_usable_feature_mask,
    compute_euclidean_distances,
    compute_roc_metrics,
    extract_flattened_vectors,
    load_benchmark_dataset,
    select_user_samples,
)

DATA_DIR = Path(__file__).resolve().parents[4] / "dataset"


@dataclass(frozen=True)
class TemplateBenchmarkRecord:
    user_id: str
    template_method: str
    threshold: float
    far: float
    frr: float
    acceptance_rate: float
    auc: float
    eer: float
    average_genuine_distance: float
    average_impostor_distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "template_method": self.template_method,
            "threshold": float(self.threshold),
            "far": float(self.far),
            "frr": float(self.frr),
            "acceptance_rate": float(self.acceptance_rate),
            "auc": float(self.auc),
            "eer": float(self.eer),
            "average_genuine_distance": float(self.average_genuine_distance),
            "average_impostor_distance": float(self.average_impostor_distance),
        }


@dataclass(frozen=True)
class TemplateDistributionSet:
    template_method: str
    genuine_distances: np.ndarray
    impostor_distances: np.ndarray


def load_template_benchmark_dataset(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    return load_benchmark_dataset(data_dir or DATA_DIR)


def build_user_pipeline_vectors(
    registration_samples: np.ndarray,
    validation_genuine_samples: np.ndarray,
    impostor_samples: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from authentication.feature_extractor import prepare_feature_vectors

    registration_flat, _, sensor_min, sensor_max = prepare_feature_vectors(
        registration_samples,
        feature_names=feature_names or FEATURE_NAMES,
    )
    usable_feature_mask = build_usable_feature_mask(registration_flat)

    registration_vectors = apply_feature_mask(registration_flat, usable_feature_mask)

    genuine_flat, _, _, _ = prepare_feature_vectors(
        validation_genuine_samples,
        feature_names=feature_names or FEATURE_NAMES,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
    )
    impostor_flat, _, _, _ = prepare_feature_vectors(
        impostor_samples,
        feature_names=feature_names or FEATURE_NAMES,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
    )

    genuine_vectors = apply_feature_mask(genuine_flat, usable_feature_mask)
    impostor_vectors = apply_feature_mask(impostor_flat, usable_feature_mask)
    return registration_vectors, genuine_vectors, impostor_vectors, usable_feature_mask


def build_mean_template(vectors: np.ndarray) -> np.ndarray:
    return np.mean(vectors, axis=0).astype(np.float32)


def build_median_template(vectors: np.ndarray) -> np.ndarray:
    return np.median(vectors, axis=0).astype(np.float32)


def build_trimmed_mean_template(vectors: np.ndarray, trim_fraction: float = 0.1) -> np.ndarray:
    if vectors.shape[0] < 3:
        return build_mean_template(vectors)

    trim_count = int(np.floor(vectors.shape[0] * trim_fraction))
    trim_count = min(trim_count, max((vectors.shape[0] - 1) // 2, 0))
    sorted_vectors = np.sort(vectors, axis=0)
    trimmed_vectors = sorted_vectors[trim_count : vectors.shape[0] - trim_count]
    if trimmed_vectors.shape[0] == 0:
        trimmed_vectors = sorted_vectors
    return np.mean(trimmed_vectors, axis=0).astype(np.float32)


def build_robust_mean_template(
    vectors: np.ndarray,
    max_iter: int = 25,
    huber_c: float = 1.345,
    tol: float = 1e-5,
) -> np.ndarray:
    center = np.median(vectors, axis=0).astype(np.float32)

    for _ in range(max_iter):
        distances = np.linalg.norm(vectors - center, axis=1)
        mad = float(np.median(np.abs(distances - np.median(distances)))) * 1.4826
        if mad == 0.0:
            mad = float(np.std(distances))
        if mad == 0.0:
            break

        threshold = huber_c * mad
        weights = np.ones_like(distances, dtype=np.float32)
        outlier_mask = distances > threshold
        weights[outlier_mask] = threshold / np.maximum(distances[outlier_mask], 1e-6)
        weighted_center = np.average(vectors, axis=0, weights=weights).astype(np.float32)

        if float(np.linalg.norm(weighted_center - center)) < tol:
            center = weighted_center
            break
        center = weighted_center

    return center.astype(np.float32)


def compute_far_frr_acceptance(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    genuine_distances = np.asarray(genuine_distances, dtype=np.float32)
    impostor_distances = np.asarray(impostor_distances, dtype=np.float32)
    far = float(np.mean(impostor_distances <= threshold)) if impostor_distances.size else 0.0
    frr = float(np.mean(genuine_distances > threshold)) if genuine_distances.size else 0.0
    acceptance_rate = float(np.mean(genuine_distances <= threshold)) if genuine_distances.size else 0.0
    return far, frr, acceptance_rate


def summarize_template_records(records: Sequence[TemplateBenchmarkRecord]) -> list[TemplateBenchmarkRecord]:
    grouped: dict[str, list[TemplateBenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(record.template_method, []).append(record)

    summary: list[TemplateBenchmarkRecord] = []
    for template_method, items in grouped.items():
        summary.append(
            TemplateBenchmarkRecord(
                user_id="all",
                template_method=template_method,
                threshold=float(np.mean([item.threshold for item in items])),
                far=float(np.mean([item.far for item in items])),
                frr=float(np.mean([item.frr for item in items])),
                acceptance_rate=float(np.mean([item.acceptance_rate for item in items])),
                auc=float(np.mean([item.auc for item in items])),
                eer=float(np.mean([item.eer for item in items])),
                average_genuine_distance=float(np.mean([item.average_genuine_distance for item in items])),
                average_impostor_distance=float(np.mean([item.average_impostor_distance for item in items])),
            )
        )

    return sorted(summary, key=lambda item: item.template_method)


def format_template_table(records: Sequence[TemplateBenchmarkRecord]) -> str:
    headers = [
        "Template Method",
        "Threshold",
        "FAR",
        "FRR",
        "Acceptance Rate",
        "AUC",
        "EER",
        "Average Genuine Distance",
        "Average Impostor Distance",
    ]

    rows = [
        [
            record.template_method,
            f"{record.threshold:.4f}",
            f"{record.far:.4f}",
            f"{record.frr:.4f}",
            f"{record.acceptance_rate:.4f}",
            f"{record.auc:.4f}",
            f"{record.eer:.4f}",
            f"{record.average_genuine_distance:.4f}",
            f"{record.average_impostor_distance:.4f}",
        ]
        for record in records
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)
