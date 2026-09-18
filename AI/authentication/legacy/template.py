from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Dict, Optional

import numpy as np


@dataclass
class UserTemplate:
    """A lightweight user template based on the current feature representation."""

    user_id: str
    feature_vector: np.ndarray
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    created_at: Optional[str] = None
    sensor_min: Optional[np.ndarray] = None
    sensor_max: Optional[np.ndarray] = None
    usable_feature_mask: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "feature_vector": self.feature_vector.tolist(),
            "feature_names": list(self.feature_names),
            "created_at": self.created_at,
            "sensor_min": None if self.sensor_min is None else self.sensor_min.tolist(),
            "sensor_max": None if self.sensor_max is None else self.sensor_max.tolist(),
            "usable_feature_mask": (
                None if self.usable_feature_mask is None else self.usable_feature_mask.astype(bool).tolist()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "UserTemplate":
        return cls(
            user_id=payload["user_id"],
            feature_vector=np.asarray(payload["feature_vector"], dtype=np.float32),
            feature_names=tuple(payload.get("feature_names", [])),
            created_at=payload.get("created_at"),
            sensor_min=(
                None
                if payload.get("sensor_min") is None
                else np.asarray(payload["sensor_min"], dtype=np.float32)
            ),
            sensor_max=(
                None
                if payload.get("sensor_max") is None
                else np.asarray(payload["sensor_max"], dtype=np.float32)
            ),
            usable_feature_mask=(
                None
                if payload.get("usable_feature_mask") is None
                else np.asarray(payload["usable_feature_mask"], dtype=bool)
            ),
        )


class TemplateManager:
    """Simple persistence layer for user templates."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = Path(storage_dir or Path("templates"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_template(self, template: UserTemplate) -> Path:
        path = self.storage_dir / f"{template.user_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(template.to_dict(), handle, indent=2)
        return path

    def load_template(self, user_id: str) -> Optional[UserTemplate]:
        path = self.storage_dir / f"{user_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return UserTemplate.from_dict(payload)

    def list_templates(self) -> list[str]:
        return sorted([path.stem for path in self.storage_dir.glob("*.json")])

    def delete_template(self, user_id: str) -> None:
        path = self.storage_dir / f"{user_id}.json"
        if path.exists():
            path.unlink()
