from pathlib import Path
import sys

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.preprocess_auth import prepare_authentication_data
from authentication.knn_auth import KNNAuthenticator


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


def print_distance_summary(name: str, distances: np.ndarray) -> None:
    """
    印出距離統計摘要。

    Parameters
    ----------
    name : str
        距離類型名稱。
    distances : ndarray
        一維距離陣列。
    """

    print(f"\n========== {name} Distance ==========")
    print("count:", len(distances))
    print("min  :", float(np.min(distances)))
    print("max  :", float(np.max(distances)))
    print("mean :", float(np.mean(distances)))
    print("std  :", float(np.std(distances)))
    print(f"{name}:")
    print(distances)


def main() -> None:
    """
    執行 KNN 距離型認證評估。
    """

    user_train, user_test, impostor, _, _ = prepare_authentication_data(
        DATA_DIR,
        authorized_user="amber",
    )

    authenticator = KNNAuthenticator()
    authenticator.fit(user_train)

    genuine_distances = authenticator.compute_distance(user_test)
    impostor_distances = authenticator.compute_distance(impostor)

    print_distance_summary("Genuine", genuine_distances)
    print_distance_summary("Impostor", impostor_distances)


if __name__ == "__main__":
    main()
