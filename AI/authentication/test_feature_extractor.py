from pathlib import Path
import sys

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data_loader import load_dataset
from authentication.feature_extractor import extract_dataset_features


DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


def main() -> None:
    """
    檢查原始資料與工程化特徵的形狀與 NaN 狀態。
    """

    X, _ = load_dataset(DATA_DIR)

    print("Original:", X.shape)
    X_features = extract_dataset_features(X)
    print("Feature:", X_features.shape)
    print("NaN present:", bool(np.isnan(X_features).any()))


if __name__ == "__main__":
    main()
