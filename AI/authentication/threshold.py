from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, Optional

import numpy as np


@dataclass
class UserThreshold:
    user_id: str
    threshold: float
    k_value: float = 2.0

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "threshold": float(self.threshold), "k_value": float(self.k_value)}

    @classmethod
    def from_dict(cls, payload: dict) -> "UserThreshold":
        return cls(
            user_id=payload["user_id"],
            threshold=float(payload["threshold"]),
            k_value=float(payload.get("k_value", 2.0)),
        )


class ThresholdManager:
    """Persist per-user thresholds and allow future tuning of k."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = Path(storage_dir or Path("thresholds"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_threshold(self, threshold: UserThreshold) -> Path:
        path = self.storage_dir / f"{threshold.user_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(threshold.to_dict(), handle, indent=2)
        return path

    def load_threshold(self, user_id: str) -> Optional[UserThreshold]:
        path = self.storage_dir / f"{user_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return UserThreshold.from_dict(payload)

    def list_thresholds(self) -> list[str]:
        return sorted([path.stem for path in self.storage_dir.glob("*.json")])

    def delete_threshold(self, user_id: str) -> None:
        path = self.storage_dir / f"{user_id}.json"
        if path.exists():
            path.unlink()


def compute_threshold_from_distances(distances: np.ndarray, k_value: float = 2.0) -> float:
    """Compute threshold as mean + k * std from registration distances."""
    distances = np.asarray(distances, dtype=np.float32)
    if distances.size == 0:
        raise ValueError("Distances cannot be empty")
    return float(np.mean(distances) + k_value * np.std(distances))
