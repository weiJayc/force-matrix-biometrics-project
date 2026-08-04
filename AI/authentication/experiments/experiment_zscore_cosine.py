from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import tempfile
from typing import Callable, Sequence

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.authentication import AuthenticationResult, AuthenticationSystem
from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER, prepare_feature_vectors
from authentication.registration import RegistrationSystem
from authentication.template import TemplateManager, UserTemplate
from authentication.threshold import ThresholdManager, UserThreshold, compute_threshold_from_distances
from authentication.experiments.normalization_benchmark.common import (
    DEFAULT_USERS,
    apply_feature_mask,
    build_usable_feature_mask,
    calculate_far_frr_acceptance,
    compute_cosine_distances,
    compute_roc_metrics,
    load_benchmark_dataset,
    normalize_zscore,
    select_user_samples,
)


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
        import json

        with path.open("w", encoding="utf-8") as handle:
            json.dump(template.to_dict(), handle, indent=2)
        return path

    def load_template(self, user_id: str) -> ZScoreCosineTemplate | None:
        path = self.storage_dir / f"{user_id}.json"
        if not path.exists():
            return None
        import json

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ZScoreCosineTemplate.from_dict(payload)


@dataclass(frozen=True)
class ZScoreCosineAuthenticationResult:
    user_id: str
    distance: float
    threshold: float
    accept: bool
    detail: str | None = None


@dataclass(frozen=True)
class AuthenticationSummaryRecord:
    template_user: str
    test_user: str
    accept: int
    reject: int
    acceptance_rate: float
    average_distance: float


@dataclass(frozen=True)
class PipelineComparisonRecord:
    method: str
    normalization: str
    distance: str
    far: float
    frr: float
    acceptance_rate: float
    auc: float
    eer: float
    mean_genuine_distance: float
    mean_impostor_distance: float


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


def _get_user_samples(X: np.ndarray, raw_y: np.ndarray, user_id: str) -> np.ndarray:
    return X[np.asarray(raw_y) == user_id]


def _build_original_model(
    registration_samples: np.ndarray,
    user_id: str,
    template_manager: TemplateManager,
    threshold_manager: ThresholdManager,
    k_value: float = 2.0,
) -> tuple[UserTemplate, UserThreshold, np.ndarray]:
    registration_system = RegistrationSystem(
        template_manager=template_manager,
        threshold_manager=threshold_manager,
        feature_names=ENGINEERED_FEATURE_ORDER,
        k_value=k_value,
    )
    return registration_system.register_user(registration_samples, user_id=user_id, return_details=True)


def _build_zscore_cosine_model(
    registration_samples: np.ndarray,
    user_id: str,
    template_manager: ZScoreCosineTemplateManager,
    threshold_manager: ThresholdManager,
    feature_names: tuple[str, ...] = ENGINEERED_FEATURE_ORDER,
    k_value: float = 2.0,
) -> tuple[ZScoreCosineTemplate, UserThreshold, np.ndarray]:
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
    registration_distances = compute_cosine_distances(masked_registration_vectors, template_vector)
    threshold_value = compute_threshold_from_distances(registration_distances, k_value=k_value)

    template = ZScoreCosineTemplate(
        user_id=user_id,
        feature_vector=template_vector,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        feature_names=feature_names,
        created_at=datetime.utcnow().isoformat(),
        sensor_min=sensor_min,
        sensor_max=sensor_max,
        usable_feature_mask=usable_feature_mask,
    )
    threshold = UserThreshold(user_id=user_id, threshold=threshold_value, k_value=k_value)

    template_manager.save_template(template)
    threshold_manager.save_threshold(threshold)
    return template, threshold, registration_distances


def _authenticate_original(
    template: UserTemplate,
    threshold: UserThreshold,
    sample: np.ndarray,
    auth_system: AuthenticationSystem,
) -> tuple[float, bool]:
    result = auth_system.authenticate(template, threshold, sample[np.newaxis, :, :])
    return float(result.distance), bool(result.accept)


def _authenticate_zscore_cosine(
    template: ZScoreCosineTemplate,
    threshold: UserThreshold,
    sample: np.ndarray,
) -> tuple[float, bool]:
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
    distance = float(compute_cosine_distances(masked_sample, template.feature_vector)[0])
    accept = distance <= threshold.threshold
    return distance, accept


def _evaluate_pipeline(
    X: np.ndarray,
    raw_y: np.ndarray,
    users: Sequence[str],
    registration_count: int,
    build_model_fn: Callable[[np.ndarray, str, object, ThresholdManager], tuple[object, UserThreshold, np.ndarray]],
    authenticate_fn: Callable[[object, UserThreshold, np.ndarray], tuple[float, bool]],
    template_manager: object,
    threshold_manager: ThresholdManager,
    method_name: str,
    normalization_name: str,
    distance_name: str,
) -> tuple[list[AuthenticationSummaryRecord], PipelineComparisonRecord]:
    summary_rows: list[AuthenticationSummaryRecord] = []
    genuine_distances: list[np.ndarray] = []
    impostor_distances: list[np.ndarray] = []
    user_far_values: list[float] = []
    user_frr_values: list[float] = []
    user_acceptance_values: list[float] = []

    for template_user in users:
        registration_samples, validation_genuine_samples, _ = select_user_samples(
            X,
            raw_y,
            template_user,
            registration_count,
        )
        template, threshold, _ = build_model_fn(
            registration_samples,
            template_user,
            template_manager,
            threshold_manager,
        )

        per_user_genuine_distances: list[float] = []
        per_user_impostor_distances: list[float] = []

        for test_user in users:
            test_samples = validation_genuine_samples if test_user == template_user else _get_user_samples(X, raw_y, test_user)
            accept_count = 0
            reject_count = 0
            distances: list[float] = []

            for sample in test_samples:
                distance, accept = authenticate_fn(template, threshold, sample)
                distances.append(distance)
                if accept:
                    accept_count += 1
                else:
                    reject_count += 1

            average_distance = float(np.mean(distances)) if distances else 0.0
            acceptance_rate = accept_count / len(test_samples) if len(test_samples) else 0.0
            summary_rows.append(
                AuthenticationSummaryRecord(
                    template_user=template_user,
                    test_user=test_user,
                    accept=accept_count,
                    reject=reject_count,
                    acceptance_rate=acceptance_rate,
                    average_distance=average_distance,
                )
            )

            if test_user == template_user:
                per_user_genuine_distances.extend(distances)
            else:
                per_user_impostor_distances.extend(distances)

        genuine_array = np.asarray(per_user_genuine_distances, dtype=np.float32)
        impostor_array = np.asarray(per_user_impostor_distances, dtype=np.float32)
        user_far, user_frr, user_acceptance = calculate_far_frr_acceptance(
            genuine_array,
            impostor_array,
            threshold.threshold,
        )
        user_far_values.append(user_far)
        user_frr_values.append(user_frr)
        user_acceptance_values.append(user_acceptance)
        genuine_distances.append(genuine_array)
        impostor_distances.append(impostor_array)

    all_genuine_distances = np.concatenate(genuine_distances) if genuine_distances else np.asarray([], dtype=np.float32)
    all_impostor_distances = np.concatenate(impostor_distances) if impostor_distances else np.asarray([], dtype=np.float32)
    auc, eer, _, _, _, _, _, _, _ = compute_roc_metrics(all_genuine_distances, all_impostor_distances)

    comparison_record = PipelineComparisonRecord(
        method=method_name,
        normalization=normalization_name,
        distance=distance_name,
        far=float(np.mean(user_far_values)) if user_far_values else 0.0,
        frr=float(np.mean(user_frr_values)) if user_frr_values else 0.0,
        acceptance_rate=float(np.mean(user_acceptance_values)) if user_acceptance_values else 0.0,
        auc=auc,
        eer=eer,
        mean_genuine_distance=float(np.mean(all_genuine_distances)) if all_genuine_distances.size else 0.0,
        mean_impostor_distance=float(np.mean(all_impostor_distances)) if all_impostor_distances.size else 0.0,
    )

    return summary_rows, comparison_record


def _format_authentication_summary(rows: Sequence[AuthenticationSummaryRecord]) -> str:
    headers = ["Template User", "Test User", "Accept", "Reject", "Acceptance Rate", "Average Cosine Distance"]
    body = [
        [
            row.template_user,
            row.test_user,
            str(row.accept),
            str(row.reject),
            f"{row.acceptance_rate:.4f}",
            f"{row.average_distance:.4f}",
        ]
        for row in rows
    ]
    return _format_table(headers, body)


def _format_comparison_table(rows: Sequence[PipelineComparisonRecord]) -> str:
    headers = ["Method", "Normalization", "Distance", "FAR", "FRR", "Acceptance Rate", "AUC", "EER"]
    body = [
        [
            row.method,
            row.normalization,
            row.distance,
            f"{row.far:.4f}",
            f"{row.frr:.4f}",
            f"{row.acceptance_rate:.4f}",
            f"{row.auc:.4f}",
            f"{row.eer:.4f}",
        ]
        for row in rows
    ]
    return _format_table(headers, body)


def _format_metric_details(rows: Sequence[PipelineComparisonRecord]) -> str:
    headers = ["Method", "Mean Genuine Distance", "Mean Impostor Distance"]
    body = [
        [
            row.method,
            f"{row.mean_genuine_distance:.4f}",
            f"{row.mean_impostor_distance:.4f}",
        ]
        for row in rows
    ]
    return _format_table(headers, body)


def run_zscore_cosine_benchmark(
    X: np.ndarray | None = None,
    raw_y: np.ndarray | None = None,
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
    k_value: float = 2.0,
) -> tuple[list[AuthenticationSummaryRecord], list[PipelineComparisonRecord]]:
    if X is None or raw_y is None:
        X, raw_y = load_benchmark_dataset()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        original_template_manager = TemplateManager(storage_dir=temp_path / "original_templates")
        original_threshold_manager = ThresholdManager(storage_dir=temp_path / "original_thresholds")
        original_auth_system = AuthenticationSystem(feature_names=ENGINEERED_FEATURE_ORDER)

        zscore_template_manager = ZScoreCosineTemplateManager(storage_dir=temp_path / "zscore_templates")
        zscore_threshold_manager = ThresholdManager(storage_dir=temp_path / "zscore_thresholds")

        original_summary, original_metrics = _evaluate_pipeline(
            X,
            raw_y,
            users,
            registration_count,
            lambda reg_samples, user_id, _template_manager, threshold_manager: _build_original_model(
                reg_samples,
                user_id,
                original_template_manager,
                threshold_manager,
                k_value=k_value,
            ),
            lambda template, threshold, sample: _authenticate_original(
                template,
                threshold,
                sample,
                original_auth_system,
            ),
            original_template_manager,
            original_threshold_manager,
            method_name="Original Pipeline",
            normalization_name="Current",
            distance_name="Euclidean",
        )

        zscore_summary, zscore_metrics = _evaluate_pipeline(
            X,
            raw_y,
            users,
            registration_count,
            lambda reg_samples, user_id, template_manager, threshold_manager: _build_zscore_cosine_model(
                reg_samples,
                user_id,
                template_manager,
                threshold_manager,
                feature_names=ENGINEERED_FEATURE_ORDER,
                k_value=k_value,
            ),
            _authenticate_zscore_cosine,
            zscore_template_manager,
            zscore_threshold_manager,
            method_name="Z-score + Cosine Pipeline",
            normalization_name="Z-score",
            distance_name="Cosine",
        )

        _ = original_summary
        summary_rows = zscore_summary
        comparison_rows = [original_metrics, zscore_metrics]

    return summary_rows, comparison_rows


def main() -> None:
    summary_rows, comparison_rows = run_zscore_cosine_benchmark()

    print("\n========== Z-score + Cosine Authentication Summary ==========")
    print(_format_authentication_summary(summary_rows))

    print("\n========== Pipeline Comparison ==========")
    print(_format_comparison_table(comparison_rows))

    print("\n========== Mean Distance Comparison ==========")
    print(_format_metric_details(comparison_rows))


if __name__ == "__main__":
    main()