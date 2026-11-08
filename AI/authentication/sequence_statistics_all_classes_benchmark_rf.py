from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / (
    "sequence_statistics_all_classes_rf_results"
)

# This is the only classifier-specific configuration in this benchmark.
RF_N_ESTIMATORS = 100
RF_CLASS_WEIGHT = None
RF_N_JOBS = -1

from data_loader import load_dataset
from sequence_statistics_contact_gate_benchmark_live import (
    NO_CONTACT_USER,
    RANDOM_SEED,
    REGISTRATION_COUNTS,
    REPEATS,
    TEST_COUNT,
    USERS,
    create_user_split,
    extract_dataset_statistics,
)


ALL_CLASSES = USERS + (NO_CONTACT_USER,)
RF_RESULTS_PATH = OUTPUT_DIR / (
    "sequence_statistics_all_classes_benchmark_rf_live.jsonl"
)
PER_SAMPLE_RESULTS_PATH = OUTPUT_DIR / (
    "sequence_statistics_all_classes_rf_per_sample.csv"
)
COMPARISON_REPORT_PATH = OUTPUT_DIR / (
    "sequence_statistics_all_classes_rf_benchmark_summary.txt"
)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load exactly the source benchmark dataset, rejecting legacy paths."""
    legacy_csvs = [
        path
        for path in DATA_DIR.rglob("*.csv")
        if any(part.lower() in {"legacy", "legency"} for part in path.parts)
    ]
    if legacy_csvs:
        raise ValueError(
            "Refusing to run because legacy CSV files were found: "
            + ", ".join(str(path) for path in legacy_csvs)
        )

    X, labels = load_dataset(DATA_DIR)
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    if X.ndim != 3 or X.shape[1:] != (50, 16):
        raise ValueError(
            f"Expected X with shape (samples, 50, 16), got {X.shape}"
        )
    return X, labels


def print_dataset_information(X: np.ndarray, labels: np.ndarray) -> None:
    print("\n" + "=" * 110)
    print("RANDOM FOREST DATASET INFORMATION")
    print("=" * 110)
    print(f"Dataset             : {DATA_DIR}")
    print(f"X shape             : {X.shape}")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"{str(label):<15}: {int(count)} samples")


def prepare_repeat_split(
    labels: np.ndarray,
    registration_count: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    registration_by_class: dict[str, np.ndarray] = {}
    test_by_class: dict[str, np.ndarray] = {}
    for class_name in ALL_CLASSES:
        indices = np.where(labels == class_name)[0]
        registration, test = create_user_split(indices, registration_count, rng)
        if np.intersect1d(registration, test).size:
            raise AssertionError(f"Split overlap detected for {class_name}.")
        registration_by_class[class_name] = registration
        test_by_class[class_name] = test
    return registration_by_class, test_by_class


def _classification_metrics(
    confusion: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    true_positive = np.diag(confusion).astype(np.int64)
    support = np.sum(confusion, axis=1).astype(np.int64)
    predicted = np.sum(confusion, axis=0).astype(np.int64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros(len(ALL_CLASSES)), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(len(ALL_CLASSES)), where=support != 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(ALL_CLASSES)),
        where=(precision + recall) != 0,
    )
    return {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
            "test_samples": int(support[index]),
            "correct_samples": int(true_positive[index]),
        }
        for index, class_name in enumerate(ALL_CLASSES)
    }


def evaluate_random_forest(
    X_features: np.ndarray,
    labels: np.ndarray,
    registration_count: int,
    repeat: int,
    registration_by_class: dict[str, np.ndarray],
    test_by_class: dict[str, np.ndarray],
) -> tuple[dict, list[dict[str, object]]]:
    """Fit one forest only on this repeat's registration recordings."""
    registration_indices = np.concatenate(
        [registration_by_class[class_name] for class_name in ALL_CLASSES]
    )
    test_indices = np.concatenate(
        [test_by_class[class_name] for class_name in ALL_CLASSES]
    )
    if np.intersect1d(registration_indices, test_indices).size:
        raise AssertionError("Training and test recordings overlap.")

    classifier = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_SEED,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=RF_N_JOBS,
    )
    classifier.fit(X_features[registration_indices], labels[registration_indices])
    predictions = classifier.predict(X_features[test_indices])

    class_to_index = {class_name: index for index, class_name in enumerate(ALL_CLASSES)}
    confusion = np.zeros((len(ALL_CLASSES), len(ALL_CLASSES)), dtype=np.int64)
    samples: list[dict[str, object]] = []
    for source_index, true_label, predicted_label in zip(
        test_indices, labels[test_indices], predictions
    ):
        true_class = str(true_label)
        predicted_class = str(predicted_label)
        confusion[class_to_index[true_class], class_to_index[predicted_class]] += 1
        samples.append(
            {
                "registration_count": registration_count,
                "repeat": repeat,
                "source_index": int(source_index),
                "true_class": true_class,
                "predicted_class": predicted_class,
            }
        )

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    background_index = class_to_index[NO_CONTACT_USER]
    background_total = int(confusion[background_index].sum())
    result = {
        "registration_count": registration_count,
        "repeat": repeat,
        "class_order": list(ALL_CLASSES),
        "classifier": "RandomForestClassifier",
        "rf_parameters": {
            "n_estimators": RF_N_ESTIMATORS,
            "random_state": RANDOM_SEED,
            "class_weight": RF_CLASS_WEIGHT,
            "n_jobs": RF_N_JOBS,
        },
        "train_samples": int(len(registration_indices)),
        "test_samples": int(len(test_indices)),
        "overall_accuracy": float(correct / total) if total else None,
        "correct_samples": correct,
        "total_samples": total,
        "background_accuracy": (
            float(confusion[background_index, background_index] / background_total)
            if background_total
            else None
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class_metrics": _classification_metrics(confusion),
    }
    return result, samples


def _save_heatmap(matrix: np.ndarray, title: str, path: Path, normalized: bool) -> None:
    display_matrix = matrix / matrix.sum(axis=1, keepdims=True)
    display_matrix = np.nan_to_num(display_matrix) if normalized else matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(display_matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, label="Percentage" if normalized else "Samples")
    ax.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(ALL_CLASSES)),
        yticks=np.arange(len(ALL_CLASSES)),
        xticklabels=ALL_CLASSES,
        yticklabels=ALL_CLASSES,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = display_matrix[row, column] * (100 if normalized else 1)
            text = f"{value:.1f}" if normalized else str(int(value))
            ax.text(column, row, text, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_artifacts(rows: list[dict], samples: list[dict[str, object]]) -> dict:
    selected = sorted(rows, key=lambda row: row["repeat"])
    matrices = np.asarray([row["confusion_matrix"] for row in selected], dtype=np.int64)
    aggregate = np.sum(matrices, axis=0)
    metrics = _classification_metrics(aggregate)
    repeat_accuracy = np.asarray([row["overall_accuracy"] for row in selected])
    paths = {
        "repeat_csv": OUTPUT_DIR / "sequence_statistics_all_classes_rf_repeat_statistics_reg80.csv",
        "per_class_csv": OUTPUT_DIR / "sequence_statistics_all_classes_rf_per_class_metrics_reg80.csv",
        "aggregate_csv": OUTPUT_DIR / "sequence_statistics_all_classes_rf_aggregate_confusion_matrix_reg80.csv",
        "normalized_csv": OUTPUT_DIR / "sequence_statistics_all_classes_rf_normalized_confusion_matrix_reg80.csv",
        "aggregate_png": OUTPUT_DIR / "sequence_statistics_all_classes_rf_aggregate_confusion_matrix_reg80.png",
        "normalized_png": OUTPUT_DIR / "sequence_statistics_all_classes_rf_normalized_confusion_matrix_reg80.png",
        "report": OUTPUT_DIR / "sequence_statistics_all_classes_rf_final_summary_reg80.txt",
    }
    with paths["repeat_csv"].open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["registration_count", "repeat", "train_samples", "test_samples",
                        "overall_accuracy", "correct_samples", "total_samples"],
        )
        writer.writeheader()
        writer.writerows({field: row[field] for field in writer.fieldnames} for row in selected)
    with paths["per_class_csv"].open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["registration_count", "class_name", "precision", "recall",
                        "f1_score", "test_samples", "correct_samples"],
        )
        writer.writeheader()
        writer.writerows(
            {"registration_count": 80, "class_name": name, **values}
            for name, values in metrics.items()
        )
    for key, matrix in (
        ("aggregate_csv", aggregate),
        ("normalized_csv", aggregate / aggregate.sum(axis=1, keepdims=True)),
    ):
        with paths[key].open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["true\\pred", *ALL_CLASSES])
            writer.writerows([name, *row.tolist()] for name, row in zip(ALL_CLASSES, matrix))
    _save_heatmap(aggregate, "Random Forest Aggregate Confusion Matrix", paths["aggregate_png"], False)
    _save_heatmap(aggregate, "Random Forest Row-Normalized Confusion Matrix", paths["normalized_png"], True)

    macro_precision = float(np.mean([value["precision"] for value in metrics.values()]))
    macro_recall = float(np.mean([value["recall"] for value in metrics.values()]))
    macro_f1 = float(np.mean([value["f1_score"] for value in metrics.values()]))
    summary = {
        "repeat_accuracies": [float(value) for value in repeat_accuracy],
        "repeat_accuracy_mean": float(np.mean(repeat_accuracy)),
        "repeat_accuracy_std": float(np.std(repeat_accuracy)),
        "aggregate_accuracy": float(np.trace(aggregate) / aggregate.sum()),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_metrics": metrics,
        "aggregate_confusion_matrix": aggregate.tolist(),
    }
    with paths["report"].open("w", encoding="utf-8") as file:
        file.write("Random Forest benchmark\n")
        file.write(f"RF parameters: {selected[0]['rf_parameters']}\n")
        file.write(f"Repeats: {len(selected)}\n")
        for key in ("repeat_accuracy_mean", "repeat_accuracy_std", "aggregate_accuracy",
                    "macro_precision", "macro_recall", "macro_f1"):
            file.write(f"{key}: {summary[key]:.8f}\n")
        file.write(f"Per-class metrics: {metrics}\n")
        file.write(f"Aggregate confusion matrix: {summary['aggregate_confusion_matrix']}\n")
    with PER_SAMPLE_RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    return {"paths": paths, **summary}


def _original_summary() -> dict | None:
    source = (
        Path(__file__).resolve().parent
        / "sequence_statistics_all_classes_results"
        / "sequence_statistics_all_classes_classification_live.jsonl"
    )
    if not source.exists():
        return None
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    matrices = np.asarray([row["confusion_matrix"] for row in rows], dtype=np.int64)
    aggregate = np.sum(matrices, axis=0)
    metrics = _classification_metrics(aggregate)
    return {
        "repeat_accuracy_mean": float(np.mean([row["overall_accuracy"] for row in rows])),
        "repeat_accuracies": [row["overall_accuracy"] for row in rows],
        "macro_f1": float(np.mean([value["f1_score"] for value in metrics.values()])),
        "macro_precision": float(np.mean([value["precision"] for value in metrics.values()])),
        "macro_recall": float(np.mean([value["recall"] for value in metrics.values()])),
        "per_class_metrics": metrics,
        "aggregate_confusion_matrix": aggregate.tolist(),
    }


def run_benchmark(X: np.ndarray, labels: np.ndarray) -> dict:
    print("\nExtracting Sequence Statistics...")
    X_features = extract_dataset_statistics(X)
    print(f"Sequence feature shape : {X_features.shape}")
    if np.isnan(X_features).any():
        raise ValueError("NaN detected in sequence feature dataset.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RF_RESULTS_PATH.write_text("", encoding="utf-8")
    all_rows: list[dict] = []
    all_samples: list[dict[str, object]] = []
    for registration_count in REGISTRATION_COUNTS:
        rng = np.random.default_rng(RANDOM_SEED)
        for repeat in range(1, REPEATS + 1):
            registration, test = prepare_repeat_split(labels, registration_count, rng)
            result, samples = evaluate_random_forest(
                X_features, labels, registration_count, repeat, registration, test
            )
            all_rows.append(result)
            all_samples.extend(samples)
            with RF_RESULTS_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"Reg={registration_count}, Repeat={repeat:02d}, "
                f"train={result['train_samples']}, test={result['test_samples']}, "
                f"accuracy={result['overall_accuracy']:.4f}"
            )
    return save_artifacts(all_rows, all_samples)


def main() -> None:
    print("=" * 110)
    print("SEQUENCE STATISTICS ALL-CLASSES RANDOM FOREST BENCHMARK")
    print("=" * 110)
    print(f"RF parameters: n_estimators={RF_N_ESTIMATORS}, random_state={RANDOM_SEED}, "
          f"class_weight={RF_CLASS_WEIGHT}, n_jobs={RF_N_JOBS}")
    print(f"Classes: {ALL_CLASSES}")
    print(f"Registration counts: {REGISTRATION_COUNTS}; test count: {TEST_COUNT}; repeats: {REPEATS}")
    X, labels = load_data()
    print_dataset_information(X, labels)
    rf_summary = run_benchmark(X, labels)
    original = _original_summary()
    with COMPARISON_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("Random Forest benchmark comparison\n")
        file.write(
            f"RF summary: {json.dumps(rf_summary, ensure_ascii=False, default=str)}\n"
        )
        file.write(
            f"Original summary: {json.dumps(original, ensure_ascii=False, default=str)}\n"
        )
        if original:
            file.write("\nComparison (original -> RF):\n")
            for key in ("repeat_accuracy_mean", "macro_f1", "macro_recall"):
                file.write(f"{key}: {original[key]:.8f} -> {rf_summary[key]:.8f}\n")
            file.write("\nPer-repeat accuracy (original, RF):\n")
            for repeat, (original_accuracy, rf_row) in enumerate(
                zip(original["repeat_accuracies"], rf_summary["repeat_accuracies"]),
                start=1,
            ):
                file.write(
                    f"  repeat {repeat:02d}: {original_accuracy:.8f}, {rf_row:.8f}\n"
                )
            file.write(f"Original per-class metrics: {original['per_class_metrics']}\n")
            file.write(f"RF per-class metrics: {rf_summary['per_class_metrics']}\n")
            file.write(f"Original confusion matrix: {original['aggregate_confusion_matrix']}\n")
            file.write(f"RF confusion matrix: {rf_summary['aggregate_confusion_matrix']}\n")
    print("\nRF summary:")
    print(f"  mean accuracy={rf_summary['repeat_accuracy_mean']:.6f}")
    print(f"  accuracy std={rf_summary['repeat_accuracy_std']:.6f}")
    print(f"  macro F1={rf_summary['macro_f1']:.6f}")
    print(f"  comparison report: {COMPARISON_REPORT_PATH}")


if __name__ == "__main__":
    main()
