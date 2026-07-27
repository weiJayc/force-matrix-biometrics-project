from pathlib import Path
import sys

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data_loader import load_dataset
from authentication.feature_extractor import (
    extract_dataset_features,
    extract_feature_combination,
    extract_hybrid_features,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


def test_hybrid_feature_shape_and_content() -> None:
    frame = np.arange(16, dtype=np.float32)
    hybrid = extract_hybrid_features(frame)

    assert hybrid.shape == (23,)
    assert np.allclose(hybrid[:16], frame)
    assert hybrid[16:].shape == (7,)
    assert not np.isnan(hybrid).any()


def test_dataset_hybrid_feature_shape() -> None:
    X, _ = load_dataset(DATA_DIR)
    X_hybrid = extract_hybrid_features(X)

    print("Original:", X.shape)
    print("Hybrid:", X_hybrid.shape)
    print("NaN present:", bool(np.isnan(X_hybrid).any()))

    assert X_hybrid.shape == (X.shape[0], X.shape[1], 23)
    assert not np.isnan(X_hybrid).any()


def test_ablation_feature_combination_shape() -> None:
    frame = np.arange(16, dtype=np.float32)
    combined = extract_feature_combination(frame, ["pressure_sum", "cop_x", "cop_y"])

    assert combined.shape == (19,)
    assert np.allclose(combined[:16], frame)
    assert not np.isnan(combined).any()


if __name__ == "__main__":
    test_hybrid_feature_shape_and_content()
    test_dataset_hybrid_feature_shape()
    test_ablation_feature_combination_shape()
