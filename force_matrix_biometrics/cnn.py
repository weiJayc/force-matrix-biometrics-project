from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LABELS = ("amber", "jay", "666", "background")
DEFAULT_TARGET_FRAMES = 50
DEFAULT_MAX_VALUE = 16384.0


@dataclass(frozen=True)
class RecordingSample:
    path: Path
    label: str
    frames: list[list[list[float]]]


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


def build_dataset_bundle(
    dataset_root: str | Path = "dataset",
    labels: tuple[str, ...] = DEFAULT_LABELS,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")

    files_by_label = find_labeled_csv_files(dataset_root, labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    rng = random.Random(seed)
    train_samples: list[RecordingSample] = []
    validation_samples: list[RecordingSample] = []
    train_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    rows = 0
    cols = 0

    for label in labels:
        samples = [
            load_recording_csv(path, expected_label=label)
            for path in files_by_label[label]
        ]
        samples = [
            RecordingSample(
                path=sample.path,
                label=sample.label,
                frames=resize_frames(sample.frames, target_frames),
            )
            for sample in samples
        ]
        rows = len(samples[0].frames[0])
        cols = len(samples[0].frames[0][0])

        rng.shuffle(samples)
        validation_count = int(round(len(samples) * validation_ratio))
        if validation_ratio > 0.0 and len(samples) > 1:
            validation_count = max(1, min(validation_count, len(samples) - 1))

        validation_samples.extend(samples[:validation_count])
        train_samples.extend(samples[validation_count:])
        validation_counts[label] = validation_count
        train_counts[label] = len(samples) - validation_count

    rng.shuffle(train_samples)
    rng.shuffle(validation_samples)

    return DatasetBundle(
        train_samples=train_samples,
        validation_samples=validation_samples,
        label_to_index=label_to_index,
        rows=rows,
        cols=cols,
        target_frames=target_frames,
        train_counts=train_counts,
        validation_counts=validation_counts,
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


def train_cnn(
    dataset_root: str | Path = "dataset",
    output_dir: str | Path = "models/cnn",
    target_frames: int = DEFAULT_TARGET_FRAMES,
    validation_ratio: float = 0.2,
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    torch, _, DataLoader, Dataset = _require_torch()
    bundle = build_dataset_bundle(
        dataset_root=dataset_root,
        target_frames=target_frames,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    class PressureDataset(Dataset):
        def __init__(self, samples: list[RecordingSample]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            sample = self.samples[index]
            tensor = torch.tensor(sample.frames, dtype=torch.float32).unsqueeze(0)
            tensor = tensor / DEFAULT_MAX_VALUE
            label = torch.tensor(bundle.label_to_index[sample.label], dtype=torch.long)
            return tensor, label

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    train_loader = DataLoader(PressureDataset(bundle.train_samples), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(PressureDataset(bundle.validation_samples), batch_size=batch_size)
    model = build_model(class_count=len(bundle.label_to_index)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []

    print(f"labels={bundle.label_to_index}")
    print(f"train_samples={len(bundle.train_samples)} validation_samples={len(bundle.validation_samples)}")
    print(f"train_counts={bundle.train_counts}")
    print(f"validation_counts={bundle.validation_counts}")
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "pressure_cnn.pt"
    metadata_path = output_path / "metadata.json"
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_path": str(model_path),
        "labels": list(DEFAULT_LABELS),
        "label_to_index": bundle.label_to_index,
        "target_frames": bundle.target_frames,
        "rows": bundle.rows,
        "cols": bundle.cols,
        "max_value": DEFAULT_MAX_VALUE,
        "validation_ratio": validation_ratio,
        "train_samples": len(bundle.train_samples),
        "validation_samples": len(bundle.validation_samples),
        "train_counts": bundle.train_counts,
        "validation_counts": bundle.validation_counts,
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", action="store_true", help="Only print dataset counts.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.summary:
        counts = summarize_dataset(args.dataset_root)
        print(json.dumps(counts, indent=2))
        return

    train_cnn(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        target_frames=args.target_frames,
        validation_ratio=args.validation_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
