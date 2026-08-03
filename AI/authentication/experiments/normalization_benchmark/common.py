from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER, extract_feature_combination
from preprocess import flatten_samples
from data_loader import load_dataset

DATA_DIR = Path(__file__).resolve().parents[4] / "dataset"
DEFAULT_USERS = ("amber", "jay", "666", "background")
FEATURE_NAMES = ENGINEERED_FEATURE_ORDER


@dataclass(frozen=True)
class NormalizationBenchmarkRecord:
    user_id: str
    normalization: str
    distance: str
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
            "normalization": self.normalization,
            "distance": self.distance,
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
class UserNormalizationStats:
    user_id: str
    original_features: int
    zero_range_features: int
    usable_features: int
    removed_features: int
    zero_range_ratio: float
    euclidean_mean_genuine: float
    euclidean_mean_impostor: float
    euclidean_max_impostor: float


def load_benchmark_dataset(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    return load_dataset(data_dir or DATA_DIR)


def select_user_samples(
    X: np.ndarray,
    raw_y: np.ndarray,
    user_id: str,
    registration_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_samples = X[np.asarray(raw_y) == user_id]
    impostor_samples = X[np.asarray(raw_y) != user_id]

    if len(user_samples) <= registration_count:
        raise ValueError(f"User {user_id} requires more than {registration_count} samples")

    registration_samples = user_samples[:registration_count]
    validation_genuine_samples = user_samples[registration_count:]

    if len(validation_genuine_samples) == 0:
        raise ValueError(f"User {user_id} has no validation samples after registration split")
    if len(impostor_samples) == 0:
        raise ValueError(f"User {user_id} has no impostor samples")

    return registration_samples, validation_genuine_samples, impostor_samples


def extract_flattened_vectors(
    samples: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    features = extract_feature_combination(samples, feature_names=feature_names or FEATURE_NAMES)
    return flatten_samples(features).astype(np.float32)


def compute_zero_range_count(vectors: np.ndarray) -> int:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")
    ranges = vectors.max(axis=0) - vectors.min(axis=0)
    return int(np.sum(ranges == 0))


def build_usable_feature_mask(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")
    ranges = vectors.max(axis=0) - vectors.min(axis=0)
    return (ranges != 0).astype(bool)


def apply_feature_mask(vectors: np.ndarray, usable_feature_mask: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    usable_feature_mask = np.asarray(usable_feature_mask, dtype=bool)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")
    if vectors.shape[1] != usable_feature_mask.shape[0]:
        raise ValueError(
            f"Mask length mismatch: vectors={vectors.shape[1]}, mask={usable_feature_mask.shape[0]}"
        )
    return vectors[:, usable_feature_mask].astype(np.float32)


def normalize_minmax(vectors: np.ndarray, stats_min: np.ndarray, stats_max: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    stats_min = np.asarray(stats_min, dtype=np.float32)
    stats_max = np.asarray(stats_max, dtype=np.float32)
    scale = stats_max - stats_min
    scale[scale == 0] = 1.0
    return ((vectors - stats_min) / scale).astype(np.float32)


def normalize_zscore(vectors: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    safe_std = np.where(std == 0, 1.0, std)
    return ((vectors - mean) / safe_std).astype(np.float32)


def normalize_robust(vectors: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    median = np.asarray(median, dtype=np.float32)
    iqr = np.asarray(iqr, dtype=np.float32)
    safe_iqr = np.where(iqr == 0, 1.0, iqr)
    return ((vectors - median) / safe_iqr).astype(np.float32)


def compute_euclidean_distances(vectors: np.ndarray, template_vector: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    template_vector = np.asarray(template_vector, dtype=np.float32)
    return np.linalg.norm(vectors - template_vector, axis=1).astype(np.float32)


def compute_cosine_distances(vectors: np.ndarray, template_vector: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    template_vector = np.asarray(template_vector, dtype=np.float32)

    template_norm = float(np.linalg.norm(template_vector)) or 1.0
    vector_norms = np.linalg.norm(vectors, axis=1)
    safe_vector_norms = np.where(vector_norms == 0.0, 1.0, vector_norms)
    similarities = (vectors @ template_vector) / (safe_vector_norms * template_norm)
    similarities = np.clip(similarities, -1.0, 1.0)
    return (1.0 - similarities).astype(np.float32)


def calculate_far_frr_acceptance(
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


def compute_roc_metrics(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray, list[tuple[float, float, float, float]]]:
    genuine_distances = np.asarray(genuine_distances, dtype=np.float64)
    impostor_distances = np.asarray(impostor_distances, dtype=np.float64)

    thresholds = np.unique(np.concatenate([genuine_distances, impostor_distances]))
    thresholds = np.concatenate((
        [thresholds.min() - 1e-9],
        thresholds,
        [thresholds.max() + 1e-9],
    ))

    points: list[tuple[float, float, float, float]] = []
    for threshold in thresholds:
        far, frr, acceptance_rate = calculate_far_frr_acceptance(genuine_distances, impostor_distances, float(threshold))
        points.append((float(threshold), float(far), float(1.0 - frr), float(acceptance_rate)))

    points.sort(key=lambda item: item[1])
    far_values = np.array([item[1] for item in points], dtype=np.float64)
    tpr_values = np.array([item[2] for item in points], dtype=np.float64)
    auc = float(np.trapezoid(tpr_values, far_values))

    eer_index = int(np.argmin(np.abs(far_values - (1.0 - tpr_values))))
    eer = float((far_values[eer_index] + (1.0 - tpr_values[eer_index])) / 2.0)
    best_index = int(np.argmin(far_values + (1.0 - tpr_values)))

    best_threshold = float(points[best_index][0])
    best_far = float(far_values[best_index])
    best_frr = float(1.0 - tpr_values[best_index])
    best_acceptance_rate = float(points[best_index][3])

    return auc, eer, best_threshold, best_far, best_frr, best_acceptance_rate, far_values, tpr_values, points


def summarize_user_distance_stats(
    user_id: str,
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> UserNormalizationStats:
    genuine_distances = np.asarray(genuine_distances, dtype=np.float32)
    impostor_distances = np.asarray(impostor_distances, dtype=np.float32)
    return UserNormalizationStats(
        user_id=user_id,
        original_features=0,
        zero_range_features=0,
        usable_features=0,
        removed_features=0,
        zero_range_ratio=0.0,
        euclidean_mean_genuine=float(np.mean(genuine_distances)),
        euclidean_mean_impostor=float(np.mean(impostor_distances)),
        euclidean_max_impostor=float(np.max(impostor_distances)),
    )


def format_benchmark_table(records: Sequence[NormalizationBenchmarkRecord]) -> str:
    headers = [
        "Normalization",
        "Distance",
        "Threshold",
        "FAR",
        "FRR",
        "Acceptance Rate",
        "AUC",
        "EER",
        "Avg Genuine Dist",
        "Avg Impostor Dist",
    ]

    rows = [
        [
            record.normalization,
            record.distance,
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


def format_zero_range_table(rows: Sequence[UserNormalizationStats]) -> str:
    headers = [
        "User",
        "Original Features",
        "Zero-Range Features",
        "Usable Features",
        "Removed Features",
        "Zero-Range Ratio",
        "Mean Genuine Dist",
        "Mean Impostor Dist",
        "Max Impostor Dist",
    ]
    table_rows = [
        [
            row.user_id,
            str(row.original_features),
            str(row.zero_range_features),
            str(row.usable_features),
            str(row.removed_features),
            f"{row.zero_range_ratio:.4f}",
            f"{row.euclidean_mean_genuine:.4f}",
            f"{row.euclidean_mean_impostor:.4f}",
            f"{row.euclidean_max_impostor:.4f}",
        ]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in table_rows)
    return "\n".join(lines)
