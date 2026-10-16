from typing import Optional

import numpy as np
from sklearn.metrics import pairwise_distances


class KNNAuthenticator:
    """
    基於 KNN 距離的生物辨識驗證器。

    這個類別只負責計算查詢樣本與授權使用者模板之間的距離，
    不做分類，也不做 threshold 判斷。
    """

    def __init__(self) -> None:
        self.template: Optional[np.ndarray] = None

    def fit(self, X_user: np.ndarray) -> None:
        """
        建立授權使用者模板。

        Parameters
        ----------
        X_user : ndarray
            授權使用者訓練資料，形狀為 (N, 800)。
        """

        if X_user.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {X_user.shape}")

        self.template = X_user.astype(np.float32, copy=False)

    def compute_distance(self, X_query: np.ndarray) -> np.ndarray:
        """
        計算查詢樣本與使用者模板的最近鄰距離。

        Parameters
        ----------
        X_query : ndarray
            查詢資料，形狀為 (N, 800)。

        Returns
        -------
        ndarray
            每筆查詢樣本的最小 Euclidean Distance。
        """

        if self.template is None:
            raise ValueError("Template is not initialized. Call fit() first.")

        if X_query.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {X_query.shape}")

        if X_query.shape[1] != self.template.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: query={X_query.shape[1]}, "
                f"template={self.template.shape[1]}"
            )

        distances = pairwise_distances(
            X_query,
            self.template,
            metric="euclidean",
        )

        return distances.min(axis=1)