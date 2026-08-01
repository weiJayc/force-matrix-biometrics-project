from pathlib import Path
import sys
import tempfile

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.authentication import AuthenticationSystem
from authentication.registration import RegistrationSystem
from authentication.template import TemplateManager
from authentication.threshold import ThresholdManager
from data_loader import load_dataset


def test_registration_and_authentication_use_shared_normalization():
    data_dir = Path(__file__).resolve().parents[2] / "dataset"
    X, raw_y = load_dataset(data_dir)

    amber_samples = X[raw_y == "amber"]
    registration_samples = amber_samples[:10]
    verification_sample = amber_samples[10][np.newaxis, :, :]

    with tempfile.TemporaryDirectory() as temp_dir:
        template_manager = TemplateManager(storage_dir=Path(temp_dir) / "templates")
        threshold_manager = ThresholdManager(storage_dir=Path(temp_dir) / "thresholds")

        registration_system = RegistrationSystem(
            template_manager=template_manager,
            threshold_manager=threshold_manager,
            feature_names=(),
        )
        template, threshold = registration_system.register_user(registration_samples, user_id="amber")

        assert template.sensor_min is not None
        assert template.sensor_max is not None

        auth_system = AuthenticationSystem(feature_names=())
        result = auth_system.authenticate(template, threshold, verification_sample)

        assert np.isfinite(result.distance)
        assert result.distance <= threshold.threshold
