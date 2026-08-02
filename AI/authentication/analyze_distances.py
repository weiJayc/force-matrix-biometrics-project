from pathlib import Path
import sys
import tempfile

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.authentication import AuthenticationSystem
from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER
from authentication.registration import RegistrationSystem
from authentication.template import TemplateManager
from authentication.threshold import ThresholdManager
from data_loader import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
DEFAULT_USERS = ("amber", "jay", "666", "background")


def summarize(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def main() -> None:
    X, raw_y = load_dataset(DATA_DIR)

    for user_id in DEFAULT_USERS:
        user_samples = X[raw_y == user_id]
        other_samples = X[raw_y != user_id]

        with tempfile.TemporaryDirectory() as temp_dir:
            template_manager = TemplateManager(storage_dir=Path(temp_dir) / "templates")
            threshold_manager = ThresholdManager(storage_dir=Path(temp_dir) / "thresholds")
            registration_system = RegistrationSystem(
                template_manager=template_manager,
                threshold_manager=threshold_manager,
                feature_names=ENGINEERED_FEATURE_ORDER,
            )
            template, threshold, _ = registration_system.register_user(
                user_samples[:10],
                user_id=user_id,
                return_details=True,
            )

        auth_system = AuthenticationSystem(feature_names=ENGINEERED_FEATURE_ORDER)
        genuine = []
        impostor = []

        for sample in user_samples:
            genuine.append(auth_system.authenticate(template, threshold, sample[np.newaxis, :, :]).distance)

        for sample in other_samples:
            impostor.append(auth_system.authenticate(template, threshold, sample[np.newaxis, :, :]).distance)

        genuine_arr = np.asarray(genuine, dtype=np.float32)
        impostor_arr = np.asarray(impostor, dtype=np.float32)

        print(f"USER {user_id}")
        print("genuine", summarize(genuine_arr))
        print("impostor", summarize(impostor_arr))
        print("threshold", float(threshold.threshold))
        print("genuine_accept_rate", float(np.mean(genuine_arr <= threshold.threshold)))
        print("impostor_accept_rate", float(np.mean(impostor_arr <= threshold.threshold)))
        print("---")


if __name__ == "__main__":
    main()
