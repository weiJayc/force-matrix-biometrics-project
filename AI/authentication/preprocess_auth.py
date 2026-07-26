from pathlib import Path
from typing import Tuple, Union

import numpy as np

from data_loader import load_dataset
from preprocess import flatten_samples, normalize_samples, normalize_with_train
from authentication.feature_extractor import extract_dataset_features


def prepare_authentication_data(
    data_dir: Union[Path, str],
    authorized_user: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    建立生物特徵驗證所需的資料管線。

    流程如下：
    1. 讀取全部資料
    2. 指定授權使用者
    3. 切出 Genuine Samples
    4. 其餘使用者整理為 Impostor Samples
    5. 使用 train 資料計算 normalization 參數
    6. 將資料展平成一維向量

    Parameters
    ----------
    data_dir : Path | str
        資料集目錄。
    authorized_user : str
        授權使用者的 label。
    test_size : float
        Genuine samples 的測試集比例。
    random_state : int
        隨機種子。

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
        user_train, user_test, impostor, sensor_min, sensor_max
    """

    data_dir = Path(data_dir)

    X, raw_y = load_dataset(data_dir)

    user_mask = raw_y == authorized_user
    impostor_mask = ~user_mask

    user_samples = X[user_mask]
    impostor_samples = X[impostor_mask]

    print(f"========== Authentication User: {authorized_user} ==========")
    print("User samples:", user_samples.shape)
    print("Impostor samples:", impostor_samples.shape)

    from sklearn.model_selection import train_test_split

    user_train, user_test = train_test_split(
        user_samples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    user_train_features = extract_dataset_features(user_train)
    user_test_features = extract_dataset_features(user_test)
    impostor_features = extract_dataset_features(impostor_samples)

    print("\n========== Authentication Split ==========")
    print("user_train:", user_train.shape)
    print("user_test :", user_test.shape)
    print("impostor :", impostor_samples.shape)

    user_train_norm, user_test_norm, sensor_min, sensor_max = normalize_with_train(
        user_train_features,
        user_test_features,
    )

    impostor_norm = normalize_samples(
        impostor_features,
        sensor_min,
        sensor_max,
    )

    user_train_flat = flatten_samples(user_train_norm)
    user_test_flat = flatten_samples(user_test_norm)
    impostor_flat = flatten_samples(impostor_norm)

    print("\n========== Authentication Flatten ==========")
    print("user_train_flat:", user_train_flat.shape)
    print("user_test_flat :", user_test_flat.shape)
    print("impostor_flat :", impostor_flat.shape)
    print("NaN in user_train_flat:", bool(np.isnan(user_train_flat).any()))
    print("NaN in user_test_flat:", bool(np.isnan(user_test_flat).any()))
    print("NaN in impostor_flat:", bool(np.isnan(impostor_flat).any()))

    return user_train_flat, user_test_flat, impostor_flat, sensor_min, sensor_max
