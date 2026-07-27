from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import extract_feature_combination
from authentication.evaluate_threshold import find_best_threshold
from authentication.knn_auth import KNNAuthenticator
from data_loader import load_dataset
from preprocess import flatten_samples, normalize_samples, normalize_with_train

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
AUTHORIZED_USER = "amber"

EXPERIMENTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Raw", ()),
    ("Raw + Pressure Sum", ("pressure_sum",)),
    ("Raw + Maximum Pressure", ("max_pressure",)),
    ("Raw + Contact Area", ("contact_area",)),
    ("Raw + COP", ("cop_x", "cop_y")),
    ("Raw + Left/Right Ratio", ("left_right_ratio",)),
    ("Raw + Top/Bottom Ratio", ("top_bottom_ratio",)),
]

HYBRID_FEATURES: Tuple[str, ...] = (
    "pressure_sum",
    "max_pressure",
    "contact_area",
    "cop_x",
    "cop_y",
    "left_right_ratio",
    "top_bottom_ratio",
)


def prepare_ablation_data(
    feature_names: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, raw_y = load_dataset(DATA_DIR)

    user_mask = raw_y == AUTHORIZED_USER
    impostor_mask = ~user_mask

    user_samples = X[user_mask]
    impostor_samples = X[impostor_mask]

    from sklearn.model_selection import train_test_split

    user_train, user_test = train_test_split(
        user_samples,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    user_train_features = extract_feature_combination(user_train, feature_names=feature_names)
    user_test_features = extract_feature_combination(user_test, feature_names=feature_names)
    impostor_features = extract_feature_combination(impostor_samples, feature_names=feature_names)

    user_train_norm, user_test_norm, sensor_min, sensor_max = normalize_with_train(
        user_train_features,
        user_test_features,
    )
    impostor_norm = normalize_samples(impostor_features, sensor_min, sensor_max)

    user_train_flat = flatten_samples(user_train_norm)
    user_test_flat = flatten_samples(user_test_norm)
    impostor_flat = flatten_samples(impostor_norm)

    return (
        user_train_flat,
        user_test_flat,
        impostor_flat,
        user_train_features,
        user_test_features,
        impostor_features,
        raw_y[impostor_mask],
        impostor_samples,
    )


def run_experiment(feature_names: Tuple[str, ...]) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_train_flat, user_test_flat, impostor_flat, _, _, _, impostor_labels, _ = prepare_ablation_data(feature_names)

    authenticator = KNNAuthenticator()
    authenticator.fit(user_train_flat)

    genuine_distances = authenticator.compute_distance(user_test_flat)
    impostor_distances = authenticator.compute_distance(impostor_flat)

    best_threshold, best_far, best_frr, _ = find_best_threshold(
        genuine_distances,
        impostor_distances,
    )

    return best_threshold, best_far, best_frr, genuine_distances, impostor_distances, impostor_labels, impostor_flat


def summarize_impostor_distances(impostor_distances: np.ndarray, impostor_labels: np.ndarray) -> None:
    for label in ["666", "jay", "background"]:
        mask = impostor_labels == label
        subset = impostor_distances[mask]
        if subset.size == 0:
            print(f"{label}: count=0 mean=0.0000 std=0.0000 min=0.0000 max=0.0000")
            continue
        print(
            f"{label}: count={subset.size} mean={subset.mean():.4f} std={subset.std():.4f} "
            f"min={subset.min():.4f} max={subset.max():.4f}"
        )


def print_results() -> None:
    print("Experiment | Feature | Flatten Dimension | Best Threshold | FAR | FRR")
    print("-" * 120)

    for experiment_name, feature_names in EXPERIMENTS:
        best_threshold, best_far, best_frr, _, _, _, _ = run_experiment(feature_names)
        flatten_dim = 50 * (16 + len(feature_names))
        print(
            f"{experiment_name} | {experiment_name} | {flatten_dim} | {best_threshold:.1f} | {best_far:.4f} | {best_frr:.4f}"
        )

    print("\n========== Impostor Distance Summary (Hybrid) ==========")
    _, _, _, _, impostor_distances, impostor_labels, _ = run_experiment(HYBRID_FEATURES)
    summarize_impostor_distances(impostor_distances, impostor_labels)


def main() -> None:
    print("========== Feature Ablation Study ==========")
    print_results()


if __name__ == "__main__":
    main()
