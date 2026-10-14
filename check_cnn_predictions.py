"""Feed every recording through a trained CNN one at a time and compare the
result with the accuracy/loss recorded in metadata.json during training.

Usage:
    python check_cnn_predictions.py --model-dir models/cnn_train10
    python check_cnn_predictions.py --model-dir models/cnn_train10 --verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from force_matrix_biometrics.cnn import (
    DEFAULT_MAX_VALUE,
    RecordingSample,
    _require_torch,
    build_confusion_matrix,
    build_dataset_bundle,
    build_model,
    print_confusion_matrix,
)


def predict_one(torch, model, sample: RecordingSample, device) -> Any:
    tensor = torch.tensor(sample.frames, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = (tensor / DEFAULT_MAX_VALUE).to(device)  # shape (1, 1, T, H, W): batch of one
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0].cpu()


def evaluate(torch, model, samples, label_to_index, labels, device, verbose, title):
    expected, predicted, losses = [], [], []
    for sample in samples:
        probs = predict_one(torch, model, sample, device)
        target = label_to_index[sample.label]
        pred = int(probs.argmax())
        expected.append(target)
        predicted.append(pred)
        losses.append(-float(probs[target].clamp_min(1e-12).log()))
        if verbose:
            flag = "OK " if pred == target else "BAD"
            dist = " ".join(f"{p:.2f}" for p in probs.tolist())
            print(f"  [{flag}] {sample.path.parent.name}/{sample.path.name} "
                  f"actual={sample.label} pred={labels[pred]} probs=[{dist}]")

    n = max(1, len(samples))
    acc = sum(e == p for e, p in zip(expected, predicted)) / n
    loss = sum(losses) / n
    print(f"{title}: samples={len(samples)} accuracy={acc:.4f} loss={loss:.4f}")
    print_confusion_matrix(build_confusion_matrix(expected, predicted, len(labels)), tuple(labels))
    return acc, loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/cnn_train10")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed (default: read from metadata, else 42).")
    parser.add_argument("--verbose", action="store_true", help="Print every sample's prediction.")
    args = parser.parse_args()

    torch, _, _, _ = _require_torch()
    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = tuple(meta["labels"])

    seed = args.seed if args.seed is not None else meta.get("seed", 42)
    if args.seed is None and "seed" not in meta:
        print("NOTE: metadata has no seed (older model); assuming 42.")

    # Recreate exactly the same split that training used.
    bundle = build_dataset_bundle(
        dataset_root=args.dataset_root,
        labels=labels,
        target_frames=meta["target_frames"],
        validation_ratio=meta["validation_ratio"],
        seed=seed,
        split_strategy=meta["split_strategy"],
    )
    if bundle.train_counts != meta["train_counts"] or bundle.validation_counts != meta["validation_counts"]:
        print("WARNING: rebuilt split counts differ from metadata; dataset changed since training?")
        print(f"  rebuilt  train={bundle.train_counts} val={bundle.validation_counts}")
        print(f"  metadata train={meta['train_counts']} val={meta['validation_counts']}")

    device = torch.device("cpu")
    model = build_model(len(labels))
    model.load_state_dict(torch.load(model_dir / "pressure_cnn.pt", map_location=device))
    model.eval()

    print("=== TRAIN split, one sample at a time, eval mode ===")
    train_acc, train_loss = evaluate(torch, model, bundle.train_samples, bundle.label_to_index, labels, device, args.verbose, "train")
    print("\n=== VALIDATION split, one sample at a time, eval mode ===")
    val_acc, val_loss = evaluate(torch, model, bundle.validation_samples, bundle.label_to_index, labels, device, args.verbose, "val")

    if meta.get("history"):
        last = meta["history"][-1]
        print("\n=== Compare with last epoch in metadata.json ===")
        print(f"{'':10}{'recorded':>12}{'re-evaluated':>14}")
        print(f"{'train_acc':10}{last['train_accuracy']:12.4f}{train_acc:14.4f}")
        print(f"{'train_loss':10}{last['train_loss']:12.4f}{train_loss:14.4f}")
        print(f"{'val_acc':10}{last['validation_accuracy']:12.4f}{val_acc:14.4f}")
        print(f"{'val_loss':10}{last['validation_loss']:12.4f}{val_loss:14.4f}")
        print("\nNote: recorded train_* is measured in train mode (dropout on, BatchNorm using batch stats,")
        print("weights changing during the epoch), so it is expected to differ slightly from the eval-mode number.")


if __name__ == "__main__":
    main()
