from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER, prepare_feature_vectors
from authentication.template import TemplateManager, UserTemplate
from authentication.threshold import ThresholdManager, UserThreshold, compute_threshold_from_distances


class RegistrationSystem:
    """Create a user template and per-user threshold from registration samples."""

    def __init__(
        self,
        template_manager: Optional[TemplateManager] = None,
        threshold_manager: Optional[ThresholdManager] = None,
        feature_names: Tuple[str, ...] | None = None,
        k_value: float = 2.0,
    ) -> None:
        self.template_manager = template_manager or TemplateManager()
        self.threshold_manager = threshold_manager or ThresholdManager()
        self.feature_names = feature_names if feature_names is not None else ENGINEERED_FEATURE_ORDER
        self.k_value = k_value

    def register_user(
        self,
        registration_samples: np.ndarray,
        user_id: str,
        return_details: bool = False,
    ) -> Tuple[UserTemplate, UserThreshold] | tuple[UserTemplate, UserThreshold, np.ndarray]:
        """Create a centroid template and threshold from registration samples."""
        registration_samples = np.asarray(registration_samples, dtype=np.float32)
        if registration_samples.ndim != 3:
            raise ValueError(f"Expected 3D registration samples, got shape {registration_samples.shape}")

        registration_flat, normalized_registration, sensor_min, sensor_max = prepare_feature_vectors(
            registration_samples,
            feature_names=self.feature_names,
        )

        feature_ranges = np.max(registration_flat, axis=0) - np.min(registration_flat, axis=0)
        usable_feature_mask = feature_ranges != 0
        if not np.any(usable_feature_mask):
            raise ValueError("No usable features found in registration samples")

        usable_registration_flat = registration_flat[:, usable_feature_mask]

        centroid = np.mean(usable_registration_flat, axis=0).astype(np.float32)
        distances = np.linalg.norm(usable_registration_flat - centroid, axis=1).astype(np.float32)
        threshold_value = compute_threshold_from_distances(distances, k_value=self.k_value)

        template = UserTemplate(
            user_id=user_id,
            feature_vector=centroid,
            feature_names=self.feature_names,
            created_at=datetime.utcnow().isoformat(),
            sensor_min=sensor_min,
            sensor_max=sensor_max,
            usable_feature_mask=usable_feature_mask,
        )
        threshold = UserThreshold(user_id=user_id, threshold=threshold_value, k_value=self.k_value)

        self.template_manager.save_template(template)
        self.threshold_manager.save_threshold(threshold)
        if return_details:
            return template, threshold, distances
        return template, threshold

    def get_template(self, user_id: str) -> Optional[UserTemplate]:
        return self.template_manager.load_template(user_id)

    def get_threshold(self, user_id: str) -> Optional[UserThreshold]:
        return self.threshold_manager.load_threshold(user_id)
