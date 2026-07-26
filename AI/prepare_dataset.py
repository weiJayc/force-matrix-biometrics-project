import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = Path("../dataset")


def load_csv(file_path):

    df = pd.read_csv(file_path)

    # 取得CSV內的使用者ID
    user_id = df["label"].iloc[0]

    # 只取16個壓力sensor
    sensor_columns = [
        f"value_{i}" for i in range(16)
    ]

    data = df[sensor_columns].values

    return data, user_id



def load_dataset(data_dir):

    X = []
    raw_labels = []


    # 搜尋所有csv
    csv_files = list(data_dir.rglob("*.csv"))


    for csv_file in csv_files:

        data, user_id = load_csv(csv_file)

        # 檢查frame數量
        if data.shape != (50,16):
            print(
                "資料格式錯誤:",
                csv_file,
                data.shape
            )
            continue


        X.append(data)

        raw_labels.append(user_id)



    return np.array(X), np.array(raw_labels)

X, raw_y = load_dataset(DATA_DIR)


encoder = LabelEncoder()

y = encoder.fit_transform(raw_y)


for index, label in enumerate(encoder.classes_):
    print(label, "->", index)

## Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train")

unique, count = np.unique(y_train, return_counts=True)

for u, c in zip(unique, count):
    print(u, c)

print()

print("Test")

unique, count = np.unique(y_test, return_counts=True)

for u, c in zip(unique, count):
    print(u, c)

# 開始做Normalization
# flatten data
train_flat = X_train.reshape(-1, 16)

# 每個 Sensor 自己算 min/max
sensor_min = train_flat.min(axis=0)
sensor_max = train_flat.max(axis=0)

print(sensor_min)
print(sensor_max)

# Normalization
X_train = (X_train - sensor_min) / (sensor_max - sensor_min)
X_test = (X_test - sensor_min) / (sensor_max - sensor_min)

print(X_train.min())
print(X_train.max())