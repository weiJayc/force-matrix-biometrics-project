from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER, prepare_feature_vectors
from data_loader import load_dataset

DATA_DIR = Path(__file__).resolve().parents[3] / "dataset"
DEFAULT_USERS = ("amber", "jay", "666", "background")


@dataclass(frozen=True)
class BenchmarkRecord:
    user_id: str
    method: str
    template: str
    distance: str
    threshold_strategy: str
    threshold: float
    far: float
    frr: float
    acceptance_rate: float
    average_genuine_distance: float
    average_impostor_distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "method": self.method,
            "template": self.template,
            "distance": self.distance,
            "threshold_strategy": self.threshold_strategy,
            "threshold": float(self.threshold),
            "far": float(self.far),
            "frr": float(self.frr),
            "acceptance_rate": float(self.acceptance_rate),
            "average_genuine_distance": float(self.average_genuine_distance),
            "average_impostor_distance": float(self.average_impostor_distance),
        }


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


def prepare_registration_vectors(
    registration_samples: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flattened_vectors, _, sensor_min, sensor_max = prepare_feature_vectors(
        registration_samples,
        feature_names=feature_names or ENGINEERED_FEATURE_ORDER,
    )
    return flattened_vectors.astype(np.float32), sensor_min, sensor_max


def prepare_companion_vectors(
    samples: np.ndarray,
    sensor_min: np.ndarray,
    sensor_max: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    flattened_vectors, _, _, _ = prepare_feature_vectors(
        samples,
        feature_names=feature_names or ENGINEERED_FEATURE_ORDER,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
    )
    return flattened_vectors.astype(np.float32)


def compute_euclidean_distances(vectors: np.ndarray, template_vector: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    template_vector = np.asarray(template_vector, dtype=np.float32)
    return np.linalg.norm(vectors - template_vector, axis=1).astype(np.float32)


def compute_cosine_distances(vectors: np.ndarray, template_vector: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    template_vector = np.asarray(template_vector, dtype=np.float32)

    template_norm = float(np.linalg.norm(template_vector))
    if template_norm == 0.0:
        template_norm = 1.0

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


def find_best_threshold(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> tuple[float, float, float]:
    genuine_distances = np.asarray(genuine_distances, dtype=np.float32)
    impostor_distances = np.asarray(impostor_distances, dtype=np.float32)
    combined = np.unique(np.concatenate([genuine_distances, impostor_distances]))

    if combined.size == 0:
        raise ValueError("Threshold sweep requires genuine and impostor distances")

    best_threshold = float(combined[0])
    best_far = 1.0
    best_frr = 1.0
    best_score = float("inf")

    for threshold in combined:
        far, frr, _ = calculate_far_frr_acceptance(genuine_distances, impostor_distances, float(threshold))
        score = far + frr
        if score < best_score:
            best_score = score
            best_threshold = float(threshold)
            best_far = far
            best_frr = frr

    return best_threshold, best_far, best_frr


def summarize_records(records: Iterable[BenchmarkRecord]) -> list[BenchmarkRecord]:
    grouped: dict[tuple[str, str, str, str], list[BenchmarkRecord]] = {}
    for record in records:
        key = (record.method, record.template, record.distance, record.threshold_strategy)
        grouped.setdefault(key, []).append(record)

    summary: list[BenchmarkRecord] = []
    for (method, template, distance, threshold_strategy), group in grouped.items():
        summary.append(
            BenchmarkRecord(
                user_id="all",
                method=method,
                template=template,
                distance=distance,
                threshold_strategy=threshold_strategy,
                threshold=float(np.mean([item.threshold for item in group])),
                far=float(np.mean([item.far for item in group])),
                frr=float(np.mean([item.frr for item in group])),
                acceptance_rate=float(np.mean([item.acceptance_rate for item in group])),
                average_genuine_distance=float(np.mean([item.average_genuine_distance for item in group])),
                average_impostor_distance=float(np.mean([item.average_impostor_distance for item in group])),
            )
        )

    return sorted(summary, key=lambda item: (item.method, item.template, item.distance, item.threshold_strategy))


def format_records_table(records: Sequence[BenchmarkRecord]) -> str:
    headers = [
        "Method",
        "Template",
        "Distance",
        "Threshold Strategy",
        "Threshold",
        "FAR",
        "FRR",
        "Acceptance Rate",
        "Average Genuine Distance",
        "Average Impostor Distance",
    ]

    rows = [
        [
            record.method,
            record.template,
            record.distance,
            record.threshold_strategy,
            f"{record.threshold:.4f}",
            f"{record.far:.4f}",
            f"{record.frr:.4f}",
            f"{record.acceptance_rate:.4f}",
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
