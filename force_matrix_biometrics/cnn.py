from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


# The people the CNN classifies; only these folders under the dataset root are read.
# dataset/background (nobody touching the sensor) is deliberately left out: whether someone
# is pressing is decided by the onset trigger, not by the CNN.
DEFAULT_LABELS = ("Amber", "Jay", "666", "Andy", "Apple", "Joanna", "Reyna", "Tiffany")
DEFAULT_TARGET_FRAMES = 50
DEFAULT_MAX_VALUE = 65535.0
DEFAULT_SPLIT_STRATEGY = "time"
PREPROCESS_CHOICES = ("resize", "onset")
DEFAULT_PREPROCESS = "resize"
# The sensor reports multiples of 256. Onset thresholds are written in raw / 256 units.
SENSOR_UNIT = 256.0

Frame = list[list[float]]


@dataclass(frozen=True)
class OnsetConfig:
    """Settings for the onset pipeline (baseline subtraction + fixed window at the press onset).

    window_frames: frames given to the model.
    pre_onset_frames: how many frames before the onset the window starts.
    onset_threshold: rise of the 16-cell total above baseline, in raw / 256 units.
    baseline_frames: frames averaged into the baseline.
    baseline_gap: frames skipped between the baseline and the frame being tested, so that a
        slow rise just before the trigger does not pull the baseline up.
    tail_frames: last frames of the recording used as the fallback baseline when the press
        was already under way at the start of the recording.
    early_baseline_tolerance: for an onset at earliest_onset, the first frames still count as
        a clean baseline unless their total is more than this far above the tail estimate
        (raw / 256 units). Above it, the press was already under way and the tail is used.
    baseline_jitter: training only; the baseline is scaled by a random factor in
        [1 - jitter, 1 + jitter] each time a sample is drawn.
    """

    window_frames: int = 28
    pre_onset_frames: int = 4
    onset_threshold: float = 60.0
    baseline_frames: int = 5
    baseline_gap: int = 3
    tail_frames: int = 5
    early_baseline_tolerance: float = 30.0
    baseline_jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.window_frames <= 0:
            raise ValueError("window_frames must be greater than zero")
        if self.pre_onset_frames < 0:
            raise ValueError("pre_onset_frames cannot be negative")
        if self.baseline_frames <= 0 or self.tail_frames <= 0:
            raise ValueError("baseline_frames and tail_frames must be greater than zero")
        if self.baseline_gap < 0:
            raise ValueError("baseline_gap cannot be negative")
        if not 0.0 <= self.baseline_jitter < 1.0:
            raise ValueError("baseline_jitter must be in [0, 1)")

    @property
    def earliest_onset(self) -> int:
        """First frame the trigger rule can test. An onset here means the press may have
        started inside the baseline frames, so their average cannot be trusted."""
        return self.baseline_frames + self.baseline_gap


DEFAULT_ONSET_CONFIG = OnsetConfig()


@dataclass(frozen=True)
class RecordingSample:
    path: Path
    label: str
    # Model input: resized frames (resize pipeline) or the baseline-subtracted window (onset).
    frames: list[Frame]
    # Onset pipeline only: what is needed to rebuild the window with a jittered baseline.
    raw_frames: list[Frame] | None = None
    onset: int | None = None
    window_start: int = 0
    baseline: Frame | None = None
    early_onset: bool = False
    tail_baseline_used: bool = False


@dataclass(frozen=True)
class DatasetBundle:
    train_samples: list[RecordingSample]
    validation_samples: list[RecordingSample]
    label_to_index: dict[str, int]
    rows: int
    cols: int
    target_frames: int
    train_counts: dict[str, int]
    validation_counts: dict[str, int]
    split_strategy: str
    labels: tuple[str, ...] = DEFAULT_LABELS
    preprocess: str = DEFAULT_PREPROCESS
    onset_config: OnsetConfig | None = None
    tail_correction: Frame | None = None
    preprocess_stats: dict[str, Any] = field(default_factory=dict)


def _require_torch() -> Any:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover - depends on the local environment.
        raise RuntimeError(
            "PyTorch is required for CNN training. Install it with: "
            "pip install -r requirements-cnn.txt"
        ) from exc

    return torch, nn, DataLoader, Dataset


def find_labeled_csv_files(dataset_root: str | Path, labels: tuple[str, ...] = DEFAULT_LABELS) -> dict[str, list[Path]]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    files_by_label: dict[str, list[Path]] = {}
    for label in labels:
        label_dir = root / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {label_dir}")
        files_by_label[label] = sorted(path for path in label_dir.glob("*.csv") if path.is_file())
        if not files_by_label[label]:
            raise ValueError(f"No CSV files found in {label_dir}")

    return files_by_label


def _value_column_sort_key(name: str) -> int:
    try:
        return int(name.removeprefix("value_"))
    except ValueError:
        return 0


def _infer_square_shape(value_count: int) -> tuple[int, int]:
    edge = int(value_count**0.5)
    if edge * edge != value_count:
        raise ValueError(f"Cannot infer square sensor grid from {value_count} values")
    return edge, edge


def load_recording_csv(path: str | Path, expected_label: str | None = None) -> RecordingSample:
    csv_path = Path(path)
    label = expected_label or csv_path.parent.name
    frames: list[list[list[float]]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        value_columns = sorted(
            [field for field in (reader.fieldnames or []) if field.startswith("value_")],
            key=_value_column_sort_key,
        )
        if not value_columns:
            raise ValueError(f"No value_* columns found in {csv_path}")

        rows, cols = _infer_square_shape(len(value_columns))
        for row in reader:
            if row.get("label"):
                label = row["label"].strip() or label
            values = [float(row[column]) for column in value_columns]
            frames.append([
                values[start : start + cols]
                for start in range(0, rows * cols, cols)
            ])

    if not frames:
        raise ValueError(f"No frames found in {csv_path}")

    if expected_label and label != expected_label:
        label = expected_label

    return RecordingSample(path=csv_path, label=label, frames=frames)


def resize_frames(frames: list[list[list[float]]], target_frames: int) -> list[list[list[float]]]:
    if target_frames <= 0:
        raise ValueError("target_frames must be greater than zero")
    if not frames:
        raise ValueError("frames cannot be empty")
    if len(frames) == target_frames:
        return [[row[:] for row in frame] for frame in frames]
    if len(frames) == 1:
        return [[row[:] for row in frames[0]] for _ in range(target_frames)]
    if target_frames == 1:
        return [[row[:] for row in frames[0]]]

    source_last = len(frames) - 1
    rows = len(frames[0])
    cols = len(frames[0][0])
    resized: list[list[list[float]]] = []

    for target_index in range(target_frames):
        source_position = target_index * source_last / (target_frames - 1)
        left_index = int(source_position)
        right_index = min(left_index + 1, source_last)
        blend = source_position - left_index

        frame: list[list[float]] = []
        for row_index in range(rows):
            row: list[float] = []
            for col_index in range(cols):
                left_value = frames[left_index][row_index][col_index]
                right_value = frames[right_index][row_index][col_index]
                row.append(left_value * (1.0 - blend) + right_value * blend)
            frame.append(row)
        resized.append(frame)

    return resized


def frame_totals(frames: list[Frame]) -> list[float]:
    """Sum of all cells in each frame, in raw / 256 units."""
    return [sum(sum(row) for row in frame) / SENSOR_UNIT for frame in frames]


def mean_frame(frames: list[Frame]) -> Frame:
    """Per-cell average over the given frames."""
    if not frames:
        raise ValueError("frames cannot be empty")
    count = len(frames)
    rows = len(frames[0])
    cols = len(frames[0][0])
    return [
        [sum(frame[row][col] for frame in frames) / count for col in range(cols)]
        for row in range(rows)
    ]


def find_onset(frames: list[Frame], config: OnsetConfig = DEFAULT_ONSET_CONFIG) -> int | None:
    """Return the frame where the press starts, or None if the trigger never fires.

    For each frame i, the baseline is the mean total of frames
    [i - baseline_gap - baseline_frames, i - baseline_gap). Frame i is the onset when both
    frame i and frame i + 1 exceed that baseline by more than onset_threshold. Written for
    whole recordings, but the rule only looks backwards plus one frame, so the live system
    can run the same check on its frame buffer.
    """
    totals = frame_totals(frames)
    count = config.baseline_frames
    gap = config.baseline_gap
    for index in range(config.earliest_onset, len(totals) - 1):
        start = index - gap - count
        baseline = sum(totals[start : start + count]) / count
        if (
            totals[index] - baseline > config.onset_threshold
            and totals[index + 1] - baseline > config.onset_threshold
        ):
            return index
    return None


def is_early_onset(onset: int, config: OnsetConfig = DEFAULT_ONSET_CONFIG) -> bool:
    return onset <= config.earliest_onset


def pre_onset_baseline(frames: list[Frame], onset: int, config: OnsetConfig = DEFAULT_ONSET_CONFIG) -> Frame:
    """Per-cell baseline from the same frames that find_onset compared against."""
    end = onset - config.baseline_gap
    start = end - config.baseline_frames
    if start < 0:
        raise ValueError(f"onset {onset} is too early for a pre-onset baseline")
    return mean_frame(frames[start:end])


def tail_baseline(frames: list[Frame], correction: Frame, config: OnsetConfig = DEFAULT_ONSET_CONFIG) -> Frame:
    """Fallback baseline for presses already under way when the recording starts: mean of the
    last frames plus a correction.

    The signal dips after release and has not fully recovered when the recording ends, so the
    raw tail reads lower than the true baseline. `correction` adds that gap back per cell.
    """
    tail = mean_frame(frames[-config.tail_frames :])
    return [
        [value + correction[row][col] for col, value in enumerate(tail_row)]
        for row, tail_row in enumerate(tail)
    ]


def extract_window(
    frames: list[Frame],
    start: int,
    baseline: Frame,
    window_frames: int,
    baseline_scale: float = 1.0,
) -> list[Frame]:
    """Subtract the baseline from each frame and take window_frames frames from start.

    Frames before the start or past the end of the recording are zero, which is the
    baseline level after subtraction. Negative values are kept.
    """
    rows = len(baseline)
    cols = len(baseline[0])
    scaled = [[value * baseline_scale for value in row] for row in baseline]
    window: list[Frame] = []
    for index in range(start, start + window_frames):
        if 0 <= index < len(frames):
            frame = frames[index]
            window.append([
                [frame[row][col] - scaled[row][col] for col in range(cols)]
                for row in range(rows)
            ])
        else:
            window.append([[0.0] * cols for _ in range(rows)])
    return window


def compute_tail_correction(samples: list[RecordingSample], config: OnsetConfig = DEFAULT_ONSET_CONFIG) -> Frame:
    """Per-cell median of (pre-onset baseline - tail mean) over recordings whose onset is
    late enough for a clean pre-onset baseline. Pass only training samples, so the
    validation split does not leak into the correction."""
    differences: list[Frame] = []
    for sample in samples:
        if sample.onset is None or is_early_onset(sample.onset, config):
            continue
        frames = sample.raw_frames if sample.raw_frames is not None else sample.frames
        pre = pre_onset_baseline(frames, sample.onset, config)
        tail = mean_frame(frames[-config.tail_frames :])
        differences.append([
            [pre[row][col] - tail[row][col] for col in range(len(pre[0]))]
            for row in range(len(pre))
        ])
    if not differences:
        raise ValueError("No recording with a clean pre-onset baseline to fit the tail correction")
    rows = len(differences[0])
    cols = len(differences[0][0])
    return [
        [statistics.median(difference[row][col] for difference in differences) for col in range(cols)]
        for row in range(rows)
    ]


def build_onset_sample(
    sample: RecordingSample,
    tail_correction: Frame,
    config: OnsetConfig = DEFAULT_ONSET_CONFIG,
) -> RecordingSample:
    """Turn a raw recording with a known onset into the model's baseline-subtracted window.

    Normal onset: baseline from the pre-onset frames, window from onset - pre_onset_frames.
    Early onset (at earliest_onset): window from frame 0. The baseline is still the first
    frames (the same ones the trigger compared against) unless they sit more than
    early_baseline_tolerance above the tail estimate, which means the press was already under
    way when recording started; only then the tail estimate is used.
    """
    if sample.onset is None:
        raise ValueError(f"{sample.path} has no onset")
    raw_frames = sample.raw_frames if sample.raw_frames is not None else sample.frames
    early = is_early_onset(sample.onset, config)
    baseline = pre_onset_baseline(raw_frames, sample.onset, config)
    used_tail = False
    if early:
        window_start = 0
        estimate = tail_baseline(raw_frames, tail_correction, config)
        rise = frame_totals([baseline])[0] - frame_totals([estimate])[0]
        if rise > config.early_baseline_tolerance:
            baseline = estimate
            used_tail = True
    else:
        window_start = sample.onset - config.pre_onset_frames
    return replace(
        sample,
        frames=extract_window(raw_frames, window_start, baseline, config.window_frames),
        raw_frames=raw_frames,
        window_start=window_start,
        baseline=baseline,
        early_onset=early,
        tail_baseline_used=used_tail,
    )


def _onset_stats(
    train_samples: list[RecordingSample],
    validation_samples: list[RecordingSample],
    skipped_no_onset: dict[str, int],
    config: OnsetConfig,
) -> dict[str, Any]:
    def padded(sample: RecordingSample) -> bool:
        length = len(sample.raw_frames or sample.frames)
        return sample.window_start + config.window_frames > length

    all_samples = train_samples + validation_samples
    onsets = [sample.onset for sample in all_samples if sample.onset is not None]
    early_by_label = Counter(sample.label for sample in all_samples if sample.early_onset)
    tail_by_label = Counter(sample.label for sample in all_samples if sample.tail_baseline_used)
    return {
        "skipped_no_onset": skipped_no_onset,
        "onset_frame": {
            "min": min(onsets),
            "median": statistics.median(onsets),
            "max": max(onsets),
        },
        "early_onset": {
            "train": sum(sample.early_onset for sample in train_samples),
            "validation": sum(sample.early_onset for sample in validation_samples),
            "by_label": dict(sorted(early_by_label.items())),
            # Early recordings already pressed at the start, so their baseline came from the tail.
            "tail_baseline": {
                "train": sum(sample.tail_baseline_used for sample in train_samples),
                "validation": sum(sample.tail_baseline_used for sample in validation_samples),
                "by_label": dict(sorted(tail_by_label.items())),
            },
        },
        "padded_past_recording_end": {
            "train": sum(padded(sample) for sample in train_samples),
            "validation": sum(padded(sample) for sample in validation_samples),
        },
    }


def build_dataset_bundle(
    dataset_root: str | Path = "dataset",
    labels: tuple[str, ...] = DEFAULT_LABELS,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    validation_ratio: float = 0.2,
    seed: int = 42,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    preprocess: str = DEFAULT_PREPROCESS,
    onset_config: OnsetConfig | None = None,
    tail_correction: Frame | None = None,
) -> DatasetBundle:
    """Load, preprocess and split the dataset.

    preprocess="resize": stretch every recording to target_frames (the original pipeline).
    preprocess="onset": find the press onset, subtract the baseline and cut a fixed window;
        target_frames is ignored in favour of onset_config.window_frames. tail_correction is
        fitted on the training split unless given (pass the saved one to rebuild a model's data).
    Only the folders named in labels are read. Pass a model's saved labels to rebuild its data.
    """
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if split_strategy not in {"random", "time"}:
        raise ValueError("split_strategy must be 'random' or 'time'")
    if preprocess not in PREPROCESS_CHOICES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_CHOICES}")
    config =(onset_config or DEFAULT_ONSET_CONFIG) if preprocess == "onset" else None

    files_by_label = find_labeled_csv_files(dataset_root, labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    rng = random.Random(seed)
    train_samples: list[RecordingSample] = []
    validation_samples: list[RecordingSample] = []
    train_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    skipped_no_onset: dict[str, int] = {}
    rows = 0
    cols = 0

    for label in labels:
        samples = [
            load_recording_csv(path, expected_label=label)
            for path in files_by_label[label]
        ]
        if config is None:
            samples = [
                replace(sample, frames=resize_frames(sample.frames, target_frames))
                for sample in samples
            ]
        else:
            located = []
            for sample in samples:
                onset = find_onset(sample.frames, config)
                if onset is not None:
                    located.append(replace(sample, raw_frames=sample.frames, onset=onset))
            skipped_no_onset[label] = len(samples) - len(located)
            if not located:
                raise ValueError(f"The trigger never fired in any '{label}' recording")
            samples = located
        rows = len(samples[0].frames[0])
        cols = len(samples[0].frames[0][0])

        validation_count = int(round(len(samples) * validation_ratio))
        if validation_ratio > 0.0 and len(samples) > 1:
            validation_count = max(1, min(validation_count, len(samples) - 1))

        if split_strategy == "random":
            rng.shuffle(samples)
            validation_label_samples = samples[:validation_count]
            train_label_samples = samples[validation_count:]
        else:
            validation_label_samples = samples[-validation_count:] if validation_count else []
            train_label_samples = samples[:-validation_count] if validation_count else samples

        validation_samples.extend(validation_label_samples)
        train_samples.extend(train_label_samples)
        validation_counts[label] = validation_count
        train_counts[label] = len(samples) - validation_count

    if split_strategy == "random":
        rng.shuffle(train_samples)
        rng.shuffle(validation_samples)

    preprocess_stats: dict[str, Any] = {}
    if config is not None:
        # Fitted after the split so that only training recordings shape the correction.
        if tail_correction is None:
            tail_correction = compute_tail_correction(train_samples, config)
        train_samples = [build_onset_sample(sample, tail_correction, config) for sample in train_samples]
        validation_samples = [build_onset_sample(sample, tail_correction, config) for sample in validation_samples]
        target_frames = config.window_frames
        preprocess_stats = _onset_stats(train_samples, validation_samples, skipped_no_onset, config)

    return DatasetBundle(
        train_samples=train_samples,
        validation_samples=validation_samples,
        label_to_index=label_to_index,
        rows=rows,
        cols=cols,
        target_frames=target_frames,
        train_counts=train_counts,
        validation_counts=validation_counts,
        split_strategy=split_strategy,
        labels=tuple(labels),
        preprocess=preprocess,
        onset_config=config,
        tail_correction=tail_correction if config is not None else None,
        preprocess_stats=preprocess_stats,
    )


def build_model(class_count: int) -> Any:
    _, nn, _, _ = _require_torch()

    class PressureCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 1, 1)),
                nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(32, class_count),
            )

        def forward(self, tensor: Any) -> Any:
            return self.classifier(self.features(tensor))

    return PressureCNN()


def summarize_dataset(dataset_root: str | Path = "dataset", labels: tuple[str, ...] = DEFAULT_LABELS) -> dict[str, int]:
    files_by_label = find_labeled_csv_files(dataset_root, labels)
    return {label: len(paths) for label, paths in files_by_label.items()}


def build_confusion_matrix(
    expected_indexes: list[int],
    predicted_indexes: list[int],
    class_count: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for expected, predicted in zip(expected_indexes, predicted_indexes):
        matrix[expected][predicted] += 1
    return matrix


def print_confusion_matrix(matrix: list[list[int]], labels: tuple[str, ...]) -> None:
    print("confusion_matrix rows=actual columns=predicted")
    header = "actual\\pred".ljust(14) + "".join(label.rjust(12) for label in labels)
    print(header)
    for label, row in zip(labels, matrix):
        values = "".join(str(value).rjust(12) for value in row)
        print(label.ljust(14) + values)


def train_cnn(
    dataset_root: str | Path = "dataset",
    output_dir: str | Path = "models/cnn",
    target_frames: int = DEFAULT_TARGET_FRAMES,
    validation_ratio: float = 0.2,
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    preprocess: str = DEFAULT_PREPROCESS,
    onset_config: OnsetConfig | None = None,
) -> dict[str, Any]:
    torch, _, DataLoader, Dataset = _require_torch()
    bundle = build_dataset_bundle(
        dataset_root=dataset_root,
        target_frames=target_frames,
        validation_ratio=validation_ratio,
        seed=seed,
        split_strategy=split_strategy,
        preprocess=preprocess,
        onset_config=onset_config,
    )
    baseline_jitter = bundle.onset_config.baseline_jitter if bundle.onset_config else 0.0

    class PressureDataset(Dataset):
        def __init__(self, samples: list[RecordingSample], jitter: float = 0.0, jitter_seed: int = 0) -> None:
            self.samples = samples
            self.jitter = jitter
            self.rng = random.Random(jitter_seed)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            sample = self.samples[index]
            frames = sample.frames
            if self.jitter > 0.0 and sample.raw_frames is not None and sample.baseline is not None:
                # Rebuild the window with a slightly wrong baseline, so the model does not
                # depend on the baseline estimate being exact.
                scale = 1.0 + self.rng.uniform(-self.jitter, self.jitter)
                frames = extract_window(
                    sample.raw_frames, sample.window_start, sample.baseline, len(sample.frames), scale
                )
            tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
            tensor = tensor / DEFAULT_MAX_VALUE
            label = torch.tensor(bundle.label_to_index[sample.label], dtype=torch.long)
            return tensor, label

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    train_loader = DataLoader(
        PressureDataset(bundle.train_samples, jitter=baseline_jitter, jitter_seed=seed),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(PressureDataset(bundle.validation_samples), batch_size=batch_size)
    model = build_model(class_count=len(bundle.label_to_index)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []

    print(f"labels={bundle.label_to_index}")
    print(f"train_samples={len(bundle.train_samples)} validation_samples={len(bundle.validation_samples)}")
    print(f"train_counts={bundle.train_counts}")
    print(f"validation_counts={bundle.validation_counts}")
    print(f"split_strategy={bundle.split_strategy}")
    print(f"preprocess={bundle.preprocess}")
    if bundle.onset_config is not None:
        print(f"onset_config={asdict(bundle.onset_config)}")
        print(f"preprocess_stats={json.dumps(bundle.preprocess_stats)}")
    print(f"input_shape=(1, {bundle.target_frames}, {bundle.rows}, {bundle.cols}) device={device}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            seen = labels.size(0)
            train_loss += float(loss.item()) * seen
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_seen += seen

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_seen = 0
        with torch.no_grad():
            for features, labels in validation_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                seen = labels.size(0)
                validation_loss += float(loss.item()) * seen
                validation_correct += int((logits.argmax(dim=1) == labels).sum().item())
                validation_seen += seen

        stats = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_seen),
            "train_accuracy": train_correct / max(1, train_seen),
            "validation_loss": validation_loss / max(1, validation_seen),
            "validation_accuracy": validation_correct / max(1, validation_seen),
        }
        history.append(stats)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={stats['train_loss']:.4f} "
            f"train_acc={stats['train_accuracy']:.3f} "
            f"val_loss={stats['validation_loss']:.4f} "
            f"val_acc={stats['validation_accuracy']:.3f}"
        )

    validation_expected: list[int] = []
    validation_predicted: list[int] = []
    model.eval()
    with torch.no_grad():
        for features, labels in validation_loader:
            features = features.to(device)
            logits = model(features)
            predictions = logits.argmax(dim=1).cpu().tolist()
            validation_predicted.extend(int(prediction) for prediction in predictions)
            validation_expected.extend(int(label) for label in labels.tolist())

    confusion_matrix = build_confusion_matrix(
        expected_indexes=validation_expected,
        predicted_indexes=validation_predicted,
        class_count=len(bundle.label_to_index),
    )
    print_confusion_matrix(confusion_matrix, bundle.labels)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "pressure_cnn.pt"
    metadata_path = output_path / "metadata.json"
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_path": str(model_path),
        "labels": list(bundle.labels),
        "label_to_index": bundle.label_to_index,
        "target_frames": bundle.target_frames,
        "rows": bundle.rows,
        "cols": bundle.cols,
        "max_value": DEFAULT_MAX_VALUE,
        "validation_ratio": validation_ratio,
        "split_strategy": bundle.split_strategy,
        "seed": seed,
        "train_samples": len(bundle.train_samples),
        "validation_samples": len(bundle.validation_samples),
        "train_counts": bundle.train_counts,
        "validation_counts": bundle.validation_counts,
        "preprocess": bundle.preprocess,
        "onset_config": asdict(bundle.onset_config) if bundle.onset_config else None,
        # Raw sensor units, one value per cell; corrects the tail baseline used for presses
        # already under way when recording started. Saved so re-evaluation uses the same one.
        "tail_correction": bundle.tail_correction,
        "preprocess_stats": bundle.preprocess_stats,
        "confusion_matrix": {
            "labels": list(bundle.labels),
            "rows": "actual",
            "columns": "predicted",
            "matrix": confusion_matrix,
        },
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved_model={model_path}")
    print(f"saved_metadata={metadata_path}")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the pressure-matrix CNN.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default="models/cnn")
    parser.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("random", "time"), default=DEFAULT_SPLIT_STRATEGY)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_CHOICES,
        default=DEFAULT_PREPROCESS,
        help="resize: stretch each recording to --target-frames (original pipeline). "
        "onset: subtract the pre-press baseline and cut a fixed window at the press onset.",
    )
    onset = parser.add_argument_group("onset pipeline (only used with --preprocess onset)")
    onset.add_argument("--window-frames", type=int, default=DEFAULT_ONSET_CONFIG.window_frames,
                       help="Frames given to the model.")
    onset.add_argument("--pre-onset-frames", type=int, default=DEFAULT_ONSET_CONFIG.pre_onset_frames,
                       help="How many frames before the onset the window starts.")
    onset.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_CONFIG.onset_threshold,
                       help="Rise of the 16-cell total above baseline that triggers, in raw/256 units.")
    onset.add_argument("--baseline-jitter", type=float, default=DEFAULT_ONSET_CONFIG.baseline_jitter,
                       help="Training only: scale the baseline by a random factor in [1-j, 1+j]. 0 turns it off.")
    parser.add_argument("--summary", action="store_true", help="Only print dataset counts.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.summary:
        counts = summarize_dataset(args.dataset_root)
        print(json.dumps(counts, indent=2))
        return

    metadata = train_cnn(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        target_frames=args.target_frames,
        validation_ratio=args.validation_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        split_strategy=args.split_strategy,
        preprocess=args.preprocess,
        onset_config=OnsetConfig(
            window_frames=args.window_frames,
            pre_onset_frames=args.pre_onset_frames,
            onset_threshold=args.onset_threshold,
            baseline_jitter=args.baseline_jitter,
        ),
    )

    from .training_plots import save_training_plots

    save_training_plots(metadata, args.output_dir)


if __name__ == "__main__":
    main()
