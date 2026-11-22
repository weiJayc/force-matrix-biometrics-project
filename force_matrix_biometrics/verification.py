from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .cnn import _require_torch, build_model, resize_frames


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "cnn2000_time_epoch250_lr2e-4"
DEFAULT_THRESHOLD = 0.50
BACKGROUND_LABEL = "background"


@dataclass(frozen=True)
class VerificationResult:
    outcome: Literal["pass", "fail", "retry"]
    detail: str


class CNNVerifier:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, threshold: float = DEFAULT_THRESHOLD) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Verification threshold must be in (0, 1].")

        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.labels = tuple(metadata["labels"])
        self.identities = tuple(label for label in self.labels if label != BACKGROUND_LABEL)
        self.target_frames = int(metadata["target_frames"])
        self.grid_shape = (int(metadata["rows"]), int(metadata["cols"]))
        self.max_value = float(metadata["max_value"])
        self.threshold = threshold

        self._torch, _, _, _ = _require_torch()
        self._model = build_model(len(self.labels))
        weights = self._torch.load(model_dir / "pressure_cnn.pt", map_location="cpu", weights_only=True)
        self._model.load_state_dict(weights)
        self._model.eval()

    def verify(self, frames: Sequence[np.ndarray], identity: str) -> VerificationResult:
        if identity not in self.identities:
            raise ValueError("Select an identity from the model's enrolled classes.")

        if len(frames) < self.target_frames:
            return VerificationResult("retry", f"至少需要 {self.target_frames} 個感測影格。")

        try:
            data = np.asarray(frames, dtype=np.float32)
        except ValueError:
            return VerificationResult("retry", "感測影格的尺寸不一致。")
        if data.shape != (len(frames), *self.grid_shape):
            return VerificationResult("retry", "感測矩陣的尺寸與模型設定不符。")
        if not np.isfinite(data).all() or data.min() < 0 or data.max() > self.max_value:
            return VerificationResult("retry", "感測值超出有效範圍。")
        if not data.any():
            return VerificationResult("retry", "未偵測到壓力。")

        resized = resize_frames(data.tolist(), self.target_frames)
        features = self._torch.tensor(resized, dtype=self._torch.float32)[None, None]
        with self._torch.inference_mode():
            probabilities = self._model(features / self.max_value).softmax(dim=1)[0]

        if not probabilities.isfinite().all().item():
            return VerificationResult("retry", "模型未產生有效分數。")
        prediction = self.labels[probabilities.argmax().item()]
        if prediction == BACKGROUND_LABEL:
            return VerificationResult("retry", "模型判定為背景，請重新按壓感測器。")

        score = probabilities[self.labels.index(identity)].item()
        detail = f"{identity} 分數 {score:.1%} · 門檻 {self.threshold:.1%}"
        if prediction != identity:
            return VerificationResult("fail", f"模型判定與所選身分不符 · {detail}")
        return VerificationResult("pass" if score >= self.threshold else "fail", detail)
