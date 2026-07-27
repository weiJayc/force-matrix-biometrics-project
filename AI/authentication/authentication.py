from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from authentication.feature_extractor import extract_feature_combination
from authentication.knn_auth import KNNAuthenticator
from authentication.template import UserTemplate
from authentication.threshold import UserThreshold
from preprocess import flatten_samples


@dataclass
class AuthenticationResult:
    user_id: str
    distance: float
    threshold: float
    accept: bool
    detail: Optional[str] = None


class AuthenticationSystem:
    """Perform user authentication against a stored template and threshold."""

    def __init__(self, feature_names: tuple[str, ...] = ()) -> None:
        self.feature_names = feature_names

    def authenticate(
        self,
        template: UserTemplate,
        threshold: UserThreshold,
        sample: np.ndarray,
    ) -> AuthenticationResult:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim != 3:
            raise ValueError(f"Expected 3D sample, got shape {sample.shape}")

        features = extract_feature_combination(sample, feature_names=self.feature_names)
        flattened = flatten_samples(features).astype(np.float32)

        template_vector = template.feature_vector.astype(np.float32, copy=False)
        if flattened.shape[1] != template_vector.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: sample={flattened.shape[1]}, template={template_vector.shape[0]}"
            )

        normalized_sample = flattened.astype(np.float32, copy=False)
        normalized_template = template_vector.astype(np.float32, copy=False)

        distance = float(np.linalg.norm(normalized_sample[0] - normalized_template))
        accept = distance <= threshold.threshold

        return AuthenticationResult(
            user_id=template.user_id,
            distance=distance,
            threshold=threshold.threshold,
            accept=accept,
            detail="accepted" if accept else "rejected",
        )
