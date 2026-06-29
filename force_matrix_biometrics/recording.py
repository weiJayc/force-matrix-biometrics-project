from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SerialProfile
from .serial_io import open_serial, packet_to_grid, read_one_packet

FrameCallback = Callable[[np.ndarray, int], None]
ProgressCallback = Callable[[float, float, int], None]


@dataclass(frozen=True)
class RecordingResult:
    label: str
    csv_path: Path
    raw_frame_count: int
    normalized_frame_count: int
    rows: int
    cols: int


@dataclass(frozen=True)
class LoadedRecording:
    label: str
    csv_path: Path
    frames: list[np.ndarray]


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("Label cannot be empty")
    return cleaned


def normalize_frames(frames: list[np.ndarray], target_frame_count: int) -> list[np.ndarray]:
    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be greater than zero")
    if not frames:
        raise ValueError("At least one frame is required to normalize a recording")

    normalized = [np.array(frame, copy=True) for frame in frames[:target_frame_count]]
    frame_shape = normalized[0].shape
    frame_dtype = normalized[0].dtype

    while len(normalized) < target_frame_count:
        normalized.append(np.zeros(frame_shape, dtype=frame_dtype))

    return normalized


def build_recording_path(dataset_root: str | Path, label: str, timestamp: datetime | None = None) -> Path:
    safe_label = sanitize_label(label)
    timestamp = timestamp or datetime.now()
    return Path(dataset_root) / safe_label / f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.csv"


def capture_frames_for_duration(
    profile: SerialProfile,
    duration_seconds: float,
    frame_callback: FrameCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[np.ndarray]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")

    ser = open_serial(profile)
    frames: list[np.ndarray] = []
    start = time.monotonic()
    deadline = start + duration_seconds

    try:
        while time.monotonic() < deadline:
            packet = read_one_packet(ser, profile)
            if packet is None:
                if progress_callback:
                    elapsed = min(time.monotonic() - start, duration_seconds)
                    progress_callback(min(elapsed / duration_seconds, 1.0), elapsed, len(frames))
                continue

            grid = packet_to_grid(packet, profile)
            frames.append(np.array(grid, copy=True))

            elapsed = min(time.monotonic() - start, duration_seconds)
            if frame_callback:
                frame_callback(frames[-1], len(frames))
            if progress_callback:
                progress_callback(min(elapsed / duration_seconds, 1.0), elapsed, len(frames))
    finally:
        ser.close()

    if not frames:
        raise RuntimeError("No valid sensor frames were captured")

    if progress_callback:
        progress_callback(1.0, duration_seconds, len(frames))

    return frames


def save_recording_csv(
    frames: list[np.ndarray],
    csv_path: str | Path,
    label: str,
) -> Path:
    if not frames:
        raise ValueError("frames cannot be empty")

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = np.asarray(frames[0])
    rows, cols = first_frame.shape
    value_columns = [f"value_{index:02d}" for index in range(rows * cols)]
    header = ["label", "frame_index", *value_columns]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for frame_index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.shape != (rows, cols):
                raise ValueError("All frames must share the same shape")
            writer.writerow([label, frame_index, *[int(value) for value in array.reshape(-1)]])

    return output_path


def load_recording_csv(csv_path: str | Path) -> LoadedRecording:
    path = Path(csv_path)
    frames: list[np.ndarray] = []
    label = path.parent.name

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        value_columns = [name for name in (reader.fieldnames or []) if name.startswith("value_")]
        if not value_columns:
            raise ValueError(f"No frame values found in {path}")

        for row in reader:
            if row.get("label"):
                label = row["label"]
            values = [int(row[column]) for column in value_columns]
            edge = int(len(values) ** 0.5)
            if edge * edge != len(values):
                raise ValueError(f"Recording {path} does not contain a square frame")
            frames.append(np.array(values, dtype=np.uint16).reshape(edge, edge))

    if not frames:
        raise ValueError(f"Recording {path} does not contain any frames")

    return LoadedRecording(label=label, csv_path=path, frames=frames)


def discover_recording_csv_files(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    if not root.exists():
        return []

    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def capture_and_save_recording(
    profile: SerialProfile,
    dataset_root: str | Path,
    label: str,
    duration_seconds: float,
    target_frame_count: int,
    frame_callback: FrameCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RecordingResult:
    sanitized_label = sanitize_label(label)
    captured_frames = capture_frames_for_duration(
        profile=profile,
        duration_seconds=duration_seconds,
        frame_callback=frame_callback,
        progress_callback=progress_callback,
    )
    normalized_frames = normalize_frames(captured_frames, target_frame_count)
    recording_path = build_recording_path(dataset_root, sanitized_label)
    csv_path = save_recording_csv(normalized_frames, recording_path, sanitized_label)

    rows, cols = normalized_frames[0].shape
    return RecordingResult(
        label=sanitized_label,
        csv_path=csv_path,
        raw_frame_count=len(captured_frames),
        normalized_frame_count=len(normalized_frames),
        rows=rows,
        cols=cols,
    )