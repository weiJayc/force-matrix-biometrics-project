from pathlib import Path
import sys

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.authentication import AuthenticationSystem
from authentication.registration import RegistrationSystem
from authentication.template import TemplateManager
from authentication.threshold import ThresholdManager
from data_loader import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


def main() -> None:
    X, raw_y = load_dataset(DATA_DIR)
    amber_samples = X[raw_y == "amber"]
    registration_samples = amber_samples[:10]

    template_manager = TemplateManager(storage_dir=Path(__file__).resolve().parent / "templates")
    threshold_manager = ThresholdManager(storage_dir=Path(__file__).resolve().parent / "thresholds")

    registration_system = RegistrationSystem(
        template_manager=template_manager,
        threshold_manager=threshold_manager,
        feature_names=(),
    )
    template, threshold = registration_system.register_user(registration_samples, user_id="amber")

    auth_system = AuthenticationSystem(feature_names=())
    sample = amber_samples[10]
    result = auth_system.authenticate(template, threshold, sample[np.newaxis, :, :])

    print("Template created:", template.user_id)
    print("Template vector shape:", template.feature_vector.shape)
    print("Threshold:", threshold.threshold)
    print("Authentication result:", result)


if __name__ == "__main__":
    main()
