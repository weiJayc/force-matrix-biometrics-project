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
ProgressCallback = Callable[[float, int], None]


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


def capture_frames_by_count(
    profile: SerialProfile,
    target_frame_count: int,
    frame_callback: FrameCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[np.ndarray]:

    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be greater than zero")

    ser = open_serial(profile)
    frames: list[np.ndarray] = []

    try:
        while len(frames) < target_frame_count:

            packet = read_one_packet(ser, profile)

            if packet is None:
                continue

            grid = packet_to_grid(packet, profile)
            frames.append(np.array(grid, copy=True))

            # callback
            if frame_callback:
                frame_callback(frames[-1], len(frames))

            if progress_callback:
                progress_callback(
                    len(frames) / target_frame_count,  # progress (0~1)
                    len(frames),                        # frame count
                )

    finally:
        ser.close()

    if not frames:
        raise RuntimeError("No valid sensor frames were captured")

    return frames

def save_recording_csv(
    packets: list[bytes],
    csv_path: str | Path,
    label: str,
) -> Path:
    if not packets:
        raise ValueError("packets cannot be empty")

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["label", "frame_index", "hex_packet"]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for frame_index, packet in enumerate(packets):
            arr = np.asarray(packet).reshape(-1)   # flatten
            hex_str = " ".join(f"{int(b):02x}" for b in arr)
            writer.writerow([label, frame_index, hex_str])

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
    target_frame_count: int,
    frame_callback: FrameCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RecordingResult:
    sanitized_label = sanitize_label(label)

    print(f"[INFO] Start capture: {target_frame_count} frames")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Label: {sanitized_label}")

    captured_frames = capture_frames_by_count(
        profile=profile,
        target_frame_count=target_frame_count,
        frame_callback=frame_callback,
        progress_callback=progress_callback,
    )

    print(f"[INFO] Captured frames: {len(captured_frames)}")

    if len(captured_frames) == 0:
        raise RuntimeError("No frames captured → sensor / COM problem")

    normalized_frames = normalize_frames(captured_frames, target_frame_count)

    print(f"[INFO] Normalized frames: {len(normalized_frames)}")

    recording_path = build_recording_path(dataset_root, sanitized_label)
    print(f"[INFO] Saving to: {recording_path}")

    csv_path = save_recording_csv(captured_frames, recording_path, sanitized_label)

    print(f"[INFO] Saved CSV: {csv_path}")

    rows, cols = normalized_frames[0].shape


    return RecordingResult(
        label=sanitized_label,
        csv_path=csv_path,
        raw_frame_count=len(captured_frames),
        normalized_frame_count=len(normalized_frames),
        rows=rows,
        cols=cols,
    )

def save_recording_raw_hex(
    packets: list[bytes],
    csv_path: str | Path,
) -> Path:
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for packet in packets:
            hex_line = " ".join(f"{b:02x}" for b in packet)
            f.write(hex_line + "\n")

    return output_path