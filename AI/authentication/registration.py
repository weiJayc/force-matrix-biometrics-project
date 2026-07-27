from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from authentication.feature_extractor import extract_feature_combination
from authentication.template import TemplateManager, UserTemplate
from authentication.threshold import ThresholdManager, UserThreshold, compute_threshold_from_distances
from preprocess import flatten_samples, normalize_samples, normalize_with_train


class RegistrationSystem:
    """Create a user template and per-user threshold from registration samples."""

    def __init__(
        self,
        template_manager: Optional[TemplateManager] = None,
        threshold_manager: Optional[ThresholdManager] = None,
        feature_names: Tuple[str, ...] = (),
        k_value: float = 2.0,
    ) -> None:
        self.template_manager = template_manager or TemplateManager()
        self.threshold_manager = threshold_manager or ThresholdManager()
        self.feature_names = feature_names
        self.k_value = k_value

    def register_user(
        self,
        registration_samples: np.ndarray,
        user_id: str,
    ) -> Tuple[UserTemplate, UserThreshold]:
        """Create a centroid template and threshold from registration samples."""
        registration_samples = np.asarray(registration_samples, dtype=np.float32)
        if registration_samples.ndim != 3:
            raise ValueError(f"Expected 3D registration samples, got shape {registration_samples.shape}")

        features = extract_feature_combination(registration_samples, feature_names=self.feature_names)
        registration_flat = flatten_samples(features)

        # Use the same normalization strategy as the existing authentication pipeline.
        registration_flat = registration_flat.astype(np.float32, copy=False)
        registration_mean = registration_flat.mean(axis=0)
        registration_std = registration_flat.std(axis=0)
        registration_std[registration_std == 0] = 1.0
        normalized_registration = (registration_flat - registration_mean) / registration_std

        centroid = np.mean(normalized_registration, axis=0).astype(np.float32)
        distances = np.linalg.norm(normalized_registration - centroid, axis=1).astype(np.float32)
        threshold_value = compute_threshold_from_distances(distances, k_value=self.k_value)

        template = UserTemplate(
            user_id=user_id,
            feature_vector=centroid,
            feature_names=self.feature_names,
            created_at=datetime.utcnow().isoformat(),
        )
        threshold = UserThreshold(user_id=user_id, threshold=threshold_value, k_value=self.k_value)

        self.template_manager.save_template(template)
        self.threshold_manager.save_threshold(threshold)
        return template, threshold

    def get_template(self, user_id: str) -> Optional[UserTemplate]:
        return self.template_manager.load_template(user_id)

    def get_threshold(self, user_id: str) -> Optional[UserThreshold]:
        return self.threshold_manager.load_threshold(user_id)
