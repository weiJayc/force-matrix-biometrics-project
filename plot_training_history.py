"""Plot CNN training history (loss/accuracy curves + confusion matrix) from metadata.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def plot_curves(metadata: dict, output_path: Path) -> None:
    history = metadata["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["validation_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["validation_accuracy"] for h in history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, train_loss, label="train")
    ax_loss.plot(epochs, val_loss, label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, train_acc, label="train")
    ax_acc.plot(epochs, val_acc, label="validation")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle(metadata.get("model_path", ""))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_confusion_matrix(metadata: dict, output_path: Path) -> None:
    cm = metadata["confusion_matrix"]
    labels = cm["labels"]
    matrix = np.array(cm["matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="models/cnn",
        help="Directory containing metadata.json (default: models/cnn)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively instead of only saving them",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metadata = load_metadata(model_dir)

    plot_curves(metadata, model_dir / "training_curves.png")
    plot_confusion_matrix(metadata, model_dir / "confusion_matrix.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
