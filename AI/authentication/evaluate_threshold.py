from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.preprocess_auth import prepare_authentication_data
from authentication.knn_auth import KNNAuthenticator
from data_loader import load_dataset


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


def calculate_far_frr(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """
    計算特定 threshold 下的 FAR 與 FRR。

    Parameters
    ----------
    genuine_distances : ndarray
        本人距離陣列。
    impostor_distances : ndarray
        陌生人距離陣列。
    threshold : float
        門檻值。

    Returns
    -------
    tuple[float, float]
        FAR 與 FRR。
    """

    far = np.mean(impostor_distances <= threshold)
    frr = np.mean(genuine_distances > threshold)

    return float(far), float(frr)


def find_best_threshold(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    start: float = 0.5,
    end: float = 8.0,
    step: float = 0.1,
) -> Tuple[float, float, float, List[Tuple[float, float, float]]]:
    """
    搜尋使 FAR + FRR 最小的 threshold。

    Parameters
    ----------
    genuine_distances : ndarray
        本人距離陣列。
    impostor_distances : ndarray
        陌生人距離陣列。
    start : float
        搜尋起始值。
    end : float
        搜尋結束值。
    step : float
        搜尋間隔。

    Returns
    -------
    tuple[float, float, float, list[tuple[float, float, float]]]
        最佳 threshold、最佳 FAR、最佳 FRR，以及所有 threshold 結果。
    """

    results: List[Tuple[float, float, float]] = []

    best_threshold = start
    best_far = 1.0
    best_frr = 1.0
    best_score = float("inf")

    thresholds = np.arange(start, end + step / 2, step)

    for threshold in thresholds:
        far, frr = calculate_far_frr(genuine_distances, impostor_distances, float(threshold))
        score = far + frr

        results.append((float(threshold), far, frr))

        if score < best_score:
            best_score = score
            best_threshold = float(threshold)
            best_far = far
            best_frr = frr

    return best_threshold, best_far, best_frr, results


def print_threshold_results(
    results: List[Tuple[float, float, float]],
    best_threshold: float,
    best_far: float,
    best_frr: float,
) -> None:
    """
    印出 threshold sweep 結果。
    """

    print("\n========== Threshold Sweep ==========")
    for threshold, far, frr in results:
        print(f"threshold={threshold:.1f} FAR={far:.4f} FRR={frr:.4f}")

    print("\n========== Best Threshold ==========")
    print(f"Best Threshold: {best_threshold:.1f}")
    print(f"Best FAR: {best_far:.4f}")
    print(f"Best FRR: {best_frr:.4f}")


def analyze_impostor_by_label(
    impostor_distances: np.ndarray,
    impostor_labels: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """
    依據 label 分析不同陌生人類別的距離統計。

    Parameters
    ----------
    impostor_distances : ndarray
        陌生人距離陣列。
    impostor_labels : ndarray
        對應的 label 陣列。

    Returns
    -------
    dict[str, tuple[float, float]]
        各類別的 mean 與 std。
    """

    summary: Dict[str, Tuple[float, float]] = {}

    for label in np.unique(impostor_labels):
        label_mask = impostor_labels == label
        label_distances = impostor_distances[label_mask]
        summary[str(label)] = (
            float(np.mean(label_distances)),
            float(np.std(label_distances)),
        )

    return summary


def print_label_summary(summary: Dict[str, Tuple[float, float]]) -> None:
    """
    印出每個 label 的距離摘要。
    """

    print("\n========== Impostor by Label ==========")
    for label, (mean_value, std_value) in summary.items():
        print(f"{label}: mean={mean_value:.4f} std={std_value:.4f}")


def main() -> None:
    """
    執行 threshold sweep、FAR/FRR 評估以及 impostor label 分析。
    """

    raw_X, raw_y = load_dataset(DATA_DIR)
    authorized_user = "amber"

    user_mask = raw_y == authorized_user
    impostor_mask = ~user_mask

    user_train, user_test, impostor, _, _ = prepare_authentication_data(
        DATA_DIR,
        authorized_user=authorized_user,
    )

    authenticator = KNNAuthenticator()
    authenticator.fit(user_train)

    genuine_distances = authenticator.compute_distance(user_test)
    impostor_distances = authenticator.compute_distance(impostor)

    best_threshold, best_far, best_frr, results = find_best_threshold(
        genuine_distances,
        impostor_distances,
    )

    print_threshold_results(results, best_threshold, best_far, best_frr)

    impostor_labels = raw_y[impostor_mask]
    label_summary = analyze_impostor_by_label(impostor_distances, impostor_labels)
    print_label_summary(label_summary)


if __name__ == "__main__":
    main()
