from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_ROOT = PROJECT_ROOT / "AI"
AUTH_ROOT = AI_ROOT / "authentication"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))
if str(AUTH_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTH_ROOT))

from data_loader import load_csv
from authentication.sequence_statistics_contact_gate_benchmark_live import (
    extract_dataset_statistics,
)

MODEL_PATH = (
    PROJECT_ROOT
    / "AI"
    / "authentication"
    / "eer_experiment_fixed_dataset"
    / "random_forest_model.pkl"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "AI"
    / "authentication"
    / "eer_experiment_fixed_dataset"
    / "dataset_manifest.csv"
)
LOG_DIR = PROJECT_ROOT / "offline_rf_recognition_logs"
THRESHOLD = 0.650554
FEATURE_COUNT = 67
EXPECTED_CLASSES = ["666", "Amber", "Andy", "Apple", "Jay", "Reyna", "Tiffany"]


class RecognitionError(RuntimeError):
    pass


def load_model(model_path: Path = MODEL_PATH) -> RandomForestClassifier:
    try:
        payload = joblib.load(model_path)
    except Exception as error:
        raise RecognitionError(f"Failed to load model PKL: {error}") from error

    if not isinstance(payload, dict) or "model" not in payload:
        raise RecognitionError("PKL format is invalid: expected an experiment payload containing 'model'.")

    model = payload["model"]
    if not isinstance(model, RandomForestClassifier):
        raise RecognitionError(
            f"PKL does not contain a RandomForestClassifier (found {type(model).__name__})."
        )
    if getattr(model, "n_features_in_", None) != FEATURE_COUNT:
        raise RecognitionError(
            f"Model expects {getattr(model, 'n_features_in_', 'unknown')} features, "
            f"but the original RF pipeline requires {FEATURE_COUNT}."
        )

    actual_classes = [str(name) for name in model.classes_.tolist()]
    if actual_classes != EXPECTED_CLASSES:
        raise RecognitionError(
            f"Unexpected model classes: {actual_classes}; expected {EXPECTED_CLASSES}."
        )
    if payload.get("feature_dimension") != FEATURE_COUNT:
        raise RecognitionError("PKL feature_dimension does not match 67.")
    return model


def validate_csv(csv_path: Path) -> np.ndarray:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise RecognitionError(f"CSV file does not exist: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise RecognitionError(f"Selected file is not a CSV: {csv_path.name}")

    try:
        frame = pd.read_csv(csv_path)
    except Exception as error:
        raise RecognitionError(f"Unable to read CSV: {error}") from error

    required = [f"value_{index}" for index in range(16)]
    missing = [column for column in ["label", *required] if column not in frame.columns]
    if missing:
        raise RecognitionError(
            f"CSV is missing required columns: {', '.join(missing)}"
        )

    sensor_frame = frame.loc[:, required]
    if len(frame) != 50:
        raise RecognitionError(
            f"CSV must contain exactly 50 Frames; found {len(frame)} rows."
        )
    if sensor_frame.columns.size != 16:
        raise RecognitionError(
            f"CSV must contain exactly 16 sensor values; found {sensor_frame.columns.size}."
        )
    if frame.isna().to_numpy().any():
        raise RecognitionError("CSV contains empty or NaN values.")

    try:
        numeric = sensor_frame.apply(pd.to_numeric, errors="raise").to_numpy(
            dtype=np.float32
        )
    except (TypeError, ValueError) as error:
        raise RecognitionError(f"CSV contains a non-numeric sensor value: {error}") from error
    if not np.isfinite(numeric).all():
        raise RecognitionError("CSV contains infinite sensor values.")

    if "frame_index" in frame.columns:
        try:
            frame_index = pd.to_numeric(frame["frame_index"], errors="raise").to_numpy(
                dtype=np.int64
            )
        except (TypeError, ValueError) as error:
            raise RecognitionError(
                f"CSV contains an invalid frame_index value: {error}"
            ) from error
        if not np.array_equal(frame_index, np.arange(50)):
            raise RecognitionError(
                "frame_index must contain exactly 0 through 49 in frame order."
            )

    labels = frame["label"].dropna().astype(str).unique().tolist()
    if len(labels) != 1:
        raise RecognitionError(
            f"CSV label must identify one value consistently; found {labels}."
        )

    try:
        data, _ = load_csv(csv_path)
    except Exception as error:
        raise RecognitionError(
            f"Original data_loader.load_csv() failed: {error}"
        ) from error
    if data.shape != (50, 16):
        raise RecognitionError(
            f"Original CSV shape is {data.shape}; expected (50, 16)."
        )
    if not np.isfinite(data).all():
        raise RecognitionError("Original data loader produced non-finite sensor data.")
    return np.asarray(data, dtype=np.float32)


def recognize(csv_path: Path, model: RandomForestClassifier) -> dict[str, Any]:
    data = validate_csv(Path(csv_path))
    features = extract_dataset_statistics(data[np.newaxis, :, :])
    if features.shape != (1, FEATURE_COUNT):
        raise RecognitionError(
            f"Feature extraction returned shape {features.shape}; expected (1, 67)."
        )
    if not np.isfinite(features).all():
        raise RecognitionError("Feature extraction produced NaN or infinite values.")

    predicted = model.predict(features)
    probabilities = model.predict_proba(features)[0]
    probability_by_class = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_.tolist(), probabilities)
    }
    ranked = sorted(probability_by_class.items(), key=lambda item: item[1], reverse=True)
    predicted_identity = str(predicted[0])
    confidence = float(probability_by_class[predicted_identity])
    accepted = confidence >= THRESHOLD

    result = {
        "csv_file": Path(csv_path).name,
        "csv_path": str(Path(csv_path).resolve()),
        "predicted_identity": predicted_identity,
        "confidence": confidence,
        "threshold": THRESHOLD,
        "acceptance": "Accept" if accepted else "Reject",
        "probabilities": ranked,
    }
    return result


def append_log(result: dict[str, Any]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "recognition_log.jsonl"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
    return log_path
