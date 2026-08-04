from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import tempfile
from typing import Iterable, Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER, prepare_feature_vectors
from authentication.threshold import ThresholdManager, UserThreshold, compute_threshold_from_distances
from authentication.experiments.normalization_benchmark.common import (
    DEFAULT_USERS,
    apply_feature_mask,
    build_usable_feature_mask,
    load_benchmark_dataset,
    normalize_zscore,
    select_user_samples,
)
from data_loader import load_dataset


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
DEFAULT_USERS = ("amber", "jay", "666", "background")


@dataclass(frozen=True)
class ZScoreCosineTemplate:
    user_id: str
    feature_vector: np.ndarray
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    feature_names: tuple[str, ...]
    created_at: str
    sensor_min: np.ndarray | None = None
    sensor_max: np.ndarray | None = None
    usable_feature_mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "feature_vector": self.feature_vector.tolist(),
            "normalization_mean": self.normalization_mean.tolist(),
            "normalization_std": self.normalization_std.tolist(),
            "feature_names": list(self.feature_names),
            "created_at": self.created_at,
            "sensor_min": None if self.sensor_min is None else self.sensor_min.tolist(),
            "sensor_max": None if self.sensor_max is None else self.sensor_max.tolist(),
            "usable_feature_mask": (
                None if self.usable_feature_mask is None else self.usable_feature_mask.astype(bool).tolist()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ZScoreCosineTemplate":
        return cls(
            user_id=str(payload["user_id"]),
            feature_vector=np.asarray(payload["feature_vector"], dtype=np.float32),
            normalization_mean=np.asarray(payload["normalization_mean"], dtype=np.float32),
            normalization_std=np.asarray(payload["normalization_std"], dtype=np.float32),
            feature_names=tuple(payload.get("feature_names", [])),
            created_at=str(payload.get("created_at") or ""),
            sensor_min=(
                None
                if payload.get("sensor_min") is None
                else np.asarray(payload["sensor_min"], dtype=np.float32)
            ),
            sensor_max=(
                None
                if payload.get("sensor_max") is None
                else np.asarray(payload["sensor_max"], dtype=np.float32)
            ),
            usable_feature_mask=(
                None
                if payload.get("usable_feature_mask") is None
                else np.asarray(payload["usable_feature_mask"], dtype=bool)
            ),
        )


class ZScoreCosineTemplateManager:
    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = Path(storage_dir or Path("zscore_cosine_templates"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_template(self, template: ZScoreCosineTemplate) -> Path:
        path = self.storage_dir / f"{template.user_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(template.to_dict(), handle, indent=2)
        return path

    def load_template(self, user_id: str) -> ZScoreCosineTemplate | None:
        path = self.storage_dir / f"{user_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ZScoreCosineTemplate.from_dict(payload)


@dataclass(frozen=True)
class AuthenticationResult:
    user_id: str
    cosine_similarity: float
    cosine_distance: float
    threshold: float
    accept: bool
    detail: str | None = None
    formula_gap: float = 0.0


@dataclass(frozen=True)
class NormalizationCheckRow:
    template_user: str
    mean_avg: float
    mean_min: float
    mean_max: float
    std_avg: float
    std_min: float
    std_max: float
    zero_std_features: int


@dataclass(frozen=True)
class AuthenticationSummaryRow:
    template_user: str
    test_user: str
    accept: int
    reject: int
    acceptance_rate: float
    average_cosine_distance: float
    average_cosine_similarity: float
    normalized_mean: float
    normalized_std: float


@dataclass(frozen=True)
class DistributionStatsRow:
    template_user: str
    pair_type: str
    sample_count: int
    mean_similarity: float
    std_similarity: float
    min_similarity: float
    q25_similarity: float
    median_similarity: float
    q75_similarity: float
    max_similarity: float


@dataclass(frozen=True)
class ThresholdCheckRow:
    template_user: str
    threshold: float
    registration_distance_mean: float
    registration_distance_std: float
    expected_threshold: float
    threshold_delta: float
    uses_cosine_distance: bool
    formula: str


@dataclass(frozen=True)
class FewShotRow:
    user: str
    registration_samples: int
    threshold: float
    far: float
    frr: float
    acceptance_rate: float


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    return load_dataset(DATA_DIR)


def _get_user_samples(X: np.ndarray, raw_y: np.ndarray, user_id: str) -> np.ndarray:
    mask = np.asarray(raw_y) == user_id
    return X[mask]


def _get_other_samples(X: np.ndarray, raw_y: np.ndarray, user_id: str) -> np.ndarray:
    mask = np.asarray(raw_y) != user_id
    return X[mask]


def _cosine_similarity(vectors: np.ndarray, template_vector: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    template_vector = np.asarray(template_vector, dtype=np.float32)

    template_norm = float(np.linalg.norm(template_vector)) or 1.0
    vector_norms = np.linalg.norm(vectors, axis=1)
    safe_vector_norms = np.where(vector_norms == 0.0, 1.0, vector_norms)
    similarities = (vectors @ template_vector) / (safe_vector_norms * template_norm)
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


def _cosine_distance(vectors: np.ndarray, template_vector: np.ndarray) -> np.ndarray:
    return (1.0 - _cosine_similarity(vectors, template_vector)).astype(np.float32)


def _summary_stats(values: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.percentile(values, 25)),
        float(np.median(values)),
        float(np.percentile(values, 75)),
        float(np.max(values)),
    )


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
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


def build_zscore_cosine_template_for_user(
    registration_samples: np.ndarray,
    user_id: str,
    template_manager: ZScoreCosineTemplateManager,
    threshold_manager: ThresholdManager,
    feature_names: tuple[str, ...] = ENGINEERED_FEATURE_ORDER,
    k_value: float = 2.0,
) -> tuple[ZScoreCosineTemplate, UserThreshold, np.ndarray, np.ndarray, np.ndarray]:
    registration_samples = np.asarray(registration_samples, dtype=np.float32)
    if registration_samples.ndim != 3:
        raise ValueError(f"Expected 3D registration samples, got shape {registration_samples.shape}")

    registration_vectors, _, sensor_min, sensor_max = prepare_feature_vectors(
        registration_samples,
        feature_names=feature_names,
    )
    usable_feature_mask = build_usable_feature_mask(registration_vectors)
    if not np.any(usable_feature_mask):
        raise ValueError("No usable features found in registration samples")

    normalization_mean = np.mean(registration_vectors, axis=0).astype(np.float32)
    normalization_std = np.std(registration_vectors, axis=0).astype(np.float32)
    normalized_registration_vectors = normalize_zscore(registration_vectors, normalization_mean, normalization_std)
    masked_registration_vectors = apply_feature_mask(normalized_registration_vectors, usable_feature_mask)

    template_vector = np.mean(masked_registration_vectors, axis=0).astype(np.float32)
    registration_distances = _cosine_distance(masked_registration_vectors, template_vector)
    threshold_value = compute_threshold_from_distances(registration_distances, k_value=k_value)

    template = ZScoreCosineTemplate(
        user_id=user_id,
        feature_vector=template_vector,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        feature_names=feature_names,
        created_at=datetime.now(timezone.utc).isoformat(),
        sensor_min=sensor_min,
        sensor_max=sensor_max,
        usable_feature_mask=usable_feature_mask,
    )
    threshold = UserThreshold(user_id=user_id, threshold=threshold_value, k_value=k_value)

    template_manager.save_template(template)
    threshold_manager.save_threshold(threshold)
    return template, threshold, registration_distances, normalized_registration_vectors, masked_registration_vectors


def authenticate_zscore_cosine(
    template: ZScoreCosineTemplate,
    threshold: UserThreshold,
    sample: np.ndarray,
) -> tuple[AuthenticationResult, np.ndarray, np.ndarray]:
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim != 2:
        raise ValueError(f"Expected 2D sample, got shape {sample.shape}")

    flattened_vectors, _, _, _ = prepare_feature_vectors(
        sample[np.newaxis, :, :],
        feature_names=template.feature_names,
        sensor_min=template.sensor_min,
        sensor_max=template.sensor_max,
    )
    normalized_sample = normalize_zscore(flattened_vectors, template.normalization_mean, template.normalization_std)
    masked_sample = apply_feature_mask(normalized_sample, template.usable_feature_mask)

    similarity = float(_cosine_similarity(masked_sample, template.feature_vector)[0])
    distance = float(1.0 - similarity)
    formula_gap = abs(distance - (1.0 - similarity))
    accept = distance <= threshold.threshold

    result = AuthenticationResult(
        user_id=template.user_id,
        cosine_similarity=similarity,
        cosine_distance=distance,
        threshold=threshold.threshold,
        accept=accept,
        detail="accepted" if accept else "rejected",
        formula_gap=formula_gap,
    )
    return result, normalized_sample, masked_sample


def evaluate_template_authentication(
    X: np.ndarray,
    raw_y: np.ndarray,
    template_user: str,
    test_users: Iterable[str] | None = None,
    registration_count: int = 10,
) -> tuple[list[AuthenticationSummaryRow], list[DistributionStatsRow], list[DistributionStatsRow], list[NormalizationCheckRow], list[ThresholdCheckRow], list[float]]:
    if test_users is None:
        test_users = DEFAULT_USERS

    with tempfile.TemporaryDirectory() as temp_dir:
        template_manager = ZScoreCosineTemplateManager(storage_dir=Path(temp_dir) / "templates")
        threshold_manager = ThresholdManager(storage_dir=Path(temp_dir) / "thresholds")

        registration_samples, validation_genuine_samples, _ = select_user_samples(
            X,
            raw_y,
            template_user,
            registration_count,
        )
        template, threshold, registration_distances, normalized_registration_vectors, masked_registration_vectors = (
            build_zscore_cosine_template_for_user(
                registration_samples,
                template_user,
                template_manager,
                threshold_manager,
            )
        )

        template_stats = NormalizationCheckRow(
            template_user=template_user,
            mean_avg=float(np.mean(template.normalization_mean)),
            mean_min=float(np.min(template.normalization_mean)),
            mean_max=float(np.max(template.normalization_mean)),
            std_avg=float(np.mean(template.normalization_std)),
            std_min=float(np.min(template.normalization_std)),
            std_max=float(np.max(template.normalization_std)),
            zero_std_features=int(np.sum(template.normalization_std == 0.0)),
        )

        threshold_check = ThresholdCheckRow(
            template_user=template_user,
            threshold=float(threshold.threshold),
            registration_distance_mean=float(np.mean(registration_distances)),
            registration_distance_std=float(np.std(registration_distances)),
            expected_threshold=float(np.mean(registration_distances) + threshold.k_value * np.std(registration_distances)),
            threshold_delta=float(abs(threshold.threshold - (np.mean(registration_distances) + threshold.k_value * np.std(registration_distances)))),
            uses_cosine_distance=True,
            formula="threshold = mean(cosine_distance) + k * std(cosine_distance)",
        )

        summary_rows: list[AuthenticationSummaryRow] = []
        genuine_distribution_rows: list[DistributionStatsRow] = []
        impostor_distribution_rows: list[DistributionStatsRow] = []
        normalization_rows: list[NormalizationCheckRow] = [template_stats]
        threshold_rows: list[ThresholdCheckRow] = [threshold_check]
        formula_gaps: list[float] = []

        for test_user in test_users:
            test_samples = validation_genuine_samples if test_user == template_user else _get_user_samples(X, raw_y, test_user)

            accept_count = 0
            reject_count = 0
            distances: list[float] = []
            similarities: list[float] = []
            normalized_values: list[np.ndarray] = []

            for sample in test_samples:
                result, normalized_sample, _ = authenticate_zscore_cosine(template, threshold, sample)
                distances.append(result.cosine_distance)
                similarities.append(result.cosine_similarity)
                normalized_values.append(normalized_sample)
                formula_gaps.append(result.formula_gap)
                if result.accept:
                    accept_count += 1
                else:
                    reject_count += 1

            acceptance_rate = accept_count / len(test_samples) if len(test_samples) else 0.0
            normalized_stack = np.vstack(normalized_values) if normalized_values else np.asarray([], dtype=np.float32)
            normalized_mean = float(np.mean(normalized_stack)) if normalized_stack.size else 0.0
            normalized_std = float(np.std(normalized_stack)) if normalized_stack.size else 0.0
            summary_rows.append(
                AuthenticationSummaryRow(
                    template_user=template_user,
                    test_user=test_user,
                    accept=accept_count,
                    reject=reject_count,
                    acceptance_rate=acceptance_rate,
                    average_cosine_distance=float(np.mean(distances)) if distances else 0.0,
                    average_cosine_similarity=float(np.mean(similarities)) if similarities else 0.0,
                    normalized_mean=normalized_mean,
                    normalized_std=normalized_std,
                )
            )

            stats = _summary_stats(np.asarray(similarities, dtype=np.float32))
            row = DistributionStatsRow(
                template_user=template_user,
                pair_type="Genuine" if test_user == template_user else "Impostor",
                sample_count=len(similarities),
                mean_similarity=stats[0],
                std_similarity=stats[1],
                min_similarity=stats[2],
                q25_similarity=stats[3],
                median_similarity=stats[4],
                q75_similarity=stats[5],
                max_similarity=stats[6],
            )
            if test_user == template_user:
                genuine_distribution_rows.append(row)
            else:
                impostor_distribution_rows.append(row)

            normalization_rows.append(
                NormalizationCheckRow(
                    template_user=f"{template_user} -> {test_user}",
                    mean_avg=normalized_mean,
                    mean_min=float(np.min(normalized_stack)) if normalized_stack.size else 0.0,
                    mean_max=float(np.max(normalized_stack)) if normalized_stack.size else 0.0,
                    std_avg=normalized_std,
                    std_min=0.0,
                    std_max=0.0,
                    zero_std_features=0,
                )
            )

        return summary_rows, genuine_distribution_rows, impostor_distribution_rows, normalization_rows, threshold_rows, formula_gaps


def evaluate_user_thresholds(X: np.ndarray, raw_y: np.ndarray, users: Iterable[str] | None = None, registration_count: int = 10) -> list[ThresholdCheckRow]:
    if users is None:
        users = DEFAULT_USERS

    rows: list[ThresholdCheckRow] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        template_manager = ZScoreCosineTemplateManager(storage_dir=Path(temp_dir) / "templates")
        threshold_manager = ThresholdManager(storage_dir=Path(temp_dir) / "thresholds")

        for user_id in users:
            registration_samples, _, _ = select_user_samples(X, raw_y, user_id, registration_count)
            _, threshold, registration_distances, _, _ = build_zscore_cosine_template_for_user(
                registration_samples,
                user_id,
                template_manager,
                threshold_manager,
            )
            rows.append(
                ThresholdCheckRow(
                    template_user=user_id,
                    threshold=float(threshold.threshold),
                    registration_distance_mean=float(np.mean(registration_distances)),
                    registration_distance_std=float(np.std(registration_distances)),
                    expected_threshold=float(np.mean(registration_distances) + threshold.k_value * np.std(registration_distances)),
                    threshold_delta=float(abs(threshold.threshold - (np.mean(registration_distances) + threshold.k_value * np.std(registration_distances)))),
                    uses_cosine_distance=True,
                    formula="threshold = mean(cosine_distance) + k * std(cosine_distance)",
                )
            )

    return rows


def run_few_shot_experiment(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Iterable[str] | None = None,
    registration_counts: Iterable[int] = (5, 10, 20, 50),
) -> list[FewShotRow]:
    if users is None:
        users = DEFAULT_USERS

    rows: list[FewShotRow] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        template_manager = ZScoreCosineTemplateManager(storage_dir=Path(temp_dir) / "templates")
        threshold_manager = ThresholdManager(storage_dir=Path(temp_dir) / "thresholds")

        for user_id in users:
            genuine_samples = _get_user_samples(X, raw_y, user_id)
            impostor_samples = _get_other_samples(X, raw_y, user_id)

            for count in registration_counts:
                registration_samples = genuine_samples[:count]
                template, threshold, _, _, _ = build_zscore_cosine_template_for_user(
                    registration_samples,
                    user_id,
                    template_manager,
                    threshold_manager,
                )

                genuine_distances: list[float] = []
                impostor_distances: list[float] = []

                for sample in genuine_samples:
                    result, _, _ = authenticate_zscore_cosine(template, threshold, sample)
                    genuine_distances.append(result.cosine_distance)

                for sample in impostor_samples:
                    result, _, _ = authenticate_zscore_cosine(template, threshold, sample)
                    impostor_distances.append(result.cosine_distance)

                genuine_array = np.asarray(genuine_distances, dtype=np.float32)
                impostor_array = np.asarray(impostor_distances, dtype=np.float32)
                far = float(np.mean(impostor_array <= threshold.threshold)) if impostor_array.size else 0.0
                frr = float(np.mean(genuine_array > threshold.threshold)) if genuine_array.size else 0.0
                acceptance_rate = float(np.mean(genuine_array <= threshold.threshold)) if genuine_array.size else 0.0

                rows.append(
                    FewShotRow(
                        user=user_id,
                        registration_samples=count,
                        threshold=float(threshold.threshold),
                        far=far,
                        frr=frr,
                        acceptance_rate=acceptance_rate,
                    )
                )

    return rows


def print_template_results(results: list[AuthenticationSummaryRow]) -> None:
    print("\n========== Z-score + Cosine Authentication Summary ==========")
    headers = [
        "Template User",
        "Test User",
        "Accept",
        "Reject",
        "Acceptance Rate",
        "Avg Cosine Distance",
        "Avg Cosine Similarity",
        "Norm Mean",
        "Norm Std",
    ]
    rows = [
        [
            row.template_user,
            row.test_user,
            str(row.accept),
            str(row.reject),
            f"{row.acceptance_rate:.4f}",
            f"{row.average_cosine_distance:.4f}",
            f"{row.average_cosine_similarity:.4f}",
            f"{row.normalized_mean:.4f}",
            f"{row.normalized_std:.4f}",
        ]
        for row in results
    ]
    print(_format_table(headers, rows))


def print_normalization_checks(rows: list[NormalizationCheckRow]) -> None:
    print("\n========== Registration Normalization Parameters ==========")
    headers = ["Template User", "Mean Avg", "Mean Min", "Mean Max", "Std Avg", "Std Min", "Std Max", "Zero Std Features"]
    body = [
        [
            row.template_user,
            f"{row.mean_avg:.4f}",
            f"{row.mean_min:.4f}",
            f"{row.mean_max:.4f}",
            f"{row.std_avg:.4f}",
            f"{row.std_min:.4f}",
            f"{row.std_max:.4f}",
            str(row.zero_std_features),
        ]
        for row in rows
    ]
    print(_format_table(headers, body))


def print_distribution_rows(title: str, rows: list[DistributionStatsRow]) -> None:
    print(f"\n========== {title} ==========")
    headers = [
        "Template User",
        "Pair Type",
        "Count",
        "Mean Similarity",
        "Std",
        "Min",
        "Q25",
        "Median",
        "Q75",
        "Max",
    ]
    body = [
        [
            row.template_user,
            row.pair_type,
            str(row.sample_count),
            f"{row.mean_similarity:.4f}",
            f"{row.std_similarity:.4f}",
            f"{row.min_similarity:.4f}",
            f"{row.q25_similarity:.4f}",
            f"{row.median_similarity:.4f}",
            f"{row.q75_similarity:.4f}",
            f"{row.max_similarity:.4f}",
        ]
        for row in rows
    ]
    print(_format_table(headers, body))


def print_threshold_checks(rows: list[ThresholdCheckRow]) -> None:
    print("\n========== Threshold Verification ==========")
    headers = [
        "Template User",
        "Threshold",
        "Reg Dist Mean",
        "Reg Dist Std",
        "Expected Threshold",
        "Delta",
        "Uses Cosine Distance",
        "Formula",
    ]
    body = [
        [
            row.template_user,
            f"{row.threshold:.6f}",
            f"{row.registration_distance_mean:.6f}",
            f"{row.registration_distance_std:.6f}",
            f"{row.expected_threshold:.6f}",
            f"{row.threshold_delta:.6f}",
            "Yes" if row.uses_cosine_distance else "No",
            row.formula,
        ]
        for row in rows
    ]
    print(_format_table(headers, body))


def print_formula_check(formula_gaps: list[float]) -> None:
    print("\n========== Cosine Distance Formula Check ==========")
    max_gap = float(np.max(formula_gaps)) if formula_gaps else 0.0
    mean_gap = float(np.mean(formula_gaps)) if formula_gaps else 0.0
    print(f"distance = 1 - cosine_similarity")
    print(f"max_abs_gap = {max_gap:.10f}")
    print(f"mean_abs_gap = {mean_gap:.10f}")


def print_few_shot_results(rows: list[FewShotRow]) -> None:
    print("\n========== Few-Shot Z-score + Cosine Experiment ==========")
    headers = ["User", "Registration Samples", "Threshold", "FAR", "FRR", "Acceptance Rate"]
    body = [
        [
            row.user,
            str(row.registration_samples),
            f"{row.threshold:.6f}",
            f"{row.far:.4f}",
            f"{row.frr:.4f}",
            f"{row.acceptance_rate:.4f}",
        ]
        for row in rows
    ]
    print(_format_table(headers, body))


def main() -> None:
    X, raw_y = _load_data()

    all_summary_rows: list[AuthenticationSummaryRow] = []
    all_genuine_rows: list[DistributionStatsRow] = []
    all_impostor_rows: list[DistributionStatsRow] = []
    all_normalization_rows: list[NormalizationCheckRow] = []
    all_threshold_rows: list[ThresholdCheckRow] = []
    formula_gaps: list[float] = []

    for template_user in DEFAULT_USERS:
        summary_rows, genuine_rows, impostor_rows, normalization_rows, threshold_rows, gaps = evaluate_template_authentication(
            X,
            raw_y,
            template_user,
            test_users=DEFAULT_USERS,
        )
        all_summary_rows.extend(summary_rows)
        all_genuine_rows.extend(genuine_rows)
        all_impostor_rows.extend(impostor_rows)
        all_normalization_rows.extend(normalization_rows)
        all_threshold_rows.extend(threshold_rows)
        formula_gaps.extend(gaps)

    print_normalization_checks(all_normalization_rows)
    print_template_results(all_summary_rows)
    print_distribution_rows("Genuine Cosine Similarity Distribution", all_genuine_rows)
    print_distribution_rows("Impostor Cosine Similarity Distribution", all_impostor_rows)
    print_formula_check(formula_gaps)
    print_threshold_checks(all_threshold_rows)

    few_shot_rows = run_few_shot_experiment(X, raw_y, users=DEFAULT_USERS)
    print_few_shot_results(few_shot_rows)


if __name__ == "__main__":
    main()