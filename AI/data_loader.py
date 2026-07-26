import pandas as pd
import numpy as np


def load_csv(file_path):
    """
    讀取單一 CSV

    Returns
    -------
    data : ndarray (50,16)
    user_id : str
    """

    df = pd.read_csv(file_path)

    user_id = str(df["label"].iloc[0])

    sensor_columns = [f"value_{i}" for i in range(16)]

    data = df[sensor_columns].values.astype(np.float32)

    return data, user_id


def load_dataset(data_dir):
    """
    讀取整個 Dataset

    Returns
    -------
    X : (N,50,16)
    raw_y : (N,)
    """

    X = []
    raw_y = []

    csv_files = sorted(data_dir.rglob("*.csv"))

    for csv_file in csv_files:

        data, user_id = load_csv(csv_file)

        if data.shape != (50, 16):
            print(f"Skip {csv_file}, shape={data.shape}")
            continue

        X.append(data)
        raw_y.append(user_id)

    return np.array(X), np.array(raw_y)