from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.preprocess_auth import prepare_authentication_data


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


user_train, user_test, impostor, sensor_min, sensor_max = prepare_authentication_data(
    DATA_DIR,
    authorized_user="amber",
)


print("========== Authentication Pipeline ==========")
print("user_train:", user_train.shape)
print("user_test :", user_test.shape)
print("impostor :", impostor.shape)
print("sensor_min:", sensor_min.shape)
print("sensor_max:", sensor_max.shape)
print("user_train range:", user_train.min(), user_train.max())
print("impostor range:", impostor.min(), impostor.max())