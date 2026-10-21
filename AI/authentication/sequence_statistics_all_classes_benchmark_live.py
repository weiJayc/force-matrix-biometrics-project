from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "sequence_statistics_all_classes_results"
LIVE_RESULTS_PATH = OUTPUT_DIR / (
    "sequence_statistics_all_classes_benchmark_live.jsonl"
)
CLASSIFICATION_RESULTS_PATH = OUTPUT_DIR / (
    "sequence_statistics_all_classes_classification_live.jsonl"
)
CLASSIFICATION_OUTPUT_DIR = OUTPUT_DIR

from data_loader import load_dataset
from sequence_statistics_contact_gate_benchmark_live import (
    CONTACT_AREA_THRESHOLD,
    NO_CONTACT_USER,
    SEQUENCE_THRESHOLD_K,
    TEST_COUNT,
    USERS,
    build_sequence_template,
    calculate_sequence_distance,
    create_user_split,
    extract_dataset_statistics,
)


ALL_CLASSES = USERS + (NO_CONTACT_USER,)
REGISTRATION_COUNTS = (80,)
REPEATS = 20
RANDOM_SEED = 42


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X, labels = load_dataset(DATA_DIR)
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    if X.ndim != 3 or X.shape[1:] != (50, 16):
        raise ValueError(
            f"Expected X with shape (samples, 50, 16), got {X.shape}"
        )
    return X, labels


def print_dataset_information(X: np.ndarray, labels: np.ndarray) -> None:
    print()
    print("=" * 110)
    print("DATASET INFORMATION")
    print("=" * 110)
    print(f"Dataset             : {DATA_DIR}")
    print(f"X shape             : {X.shape}")
    print()
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"{str(label):<15}: {int(count)} samples")


def prepare_repeat_split(
    labels: np.ndarray,
    registration_count: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    registration_indices_by_class = {}
    test_indices_by_class = {}

    for class_name in ALL_CLASSES:
        indices = np.where(labels == class_name)[0]
        registration_indices, test_indices = create_user_split(
            indices,
            registration_count,
            rng,
        )
        registration_indices_by_class[class_name] = registration_indices
        test_indices_by_class[class_name] = test_indices

    return registration_indices_by_class, test_indices_by_class


def _rate_or_none(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_direct_classification(
    X_features: np.ndarray,
    registration_count: int,
    repeat: int,
    registration_indices_by_class: dict[str, np.ndarray],
    test_indices_by_class: dict[str, np.ndarray],
) -> dict:
    """Classify every test sample by the nearest class template."""
    templates = {
        class_name: build_sequence_template(
            X_features[registration_indices_by_class[class_name]]
        )
        for class_name in ALL_CLASSES
    }

    confusion = np.zeros((len(ALL_CLASSES), len(ALL_CLASSES)), dtype=np.int64)
    class_to_index = {class_name: index for index, class_name in enumerate(ALL_CLASSES)}
    total_samples = 0
    correct_samples = 0

    for true_class in ALL_CLASSES:
        test_indices = test_indices_by_class[true_class]
        if len(test_indices) == 0:
            continue

        distances = np.column_stack(
            [
                calculate_sequence_distance(
                    templates[predicted_class],
                    X_features[test_indices],
                )
                for predicted_class in ALL_CLASSES
            ]
        )
        predicted_indices = np.argmin(distances, axis=1)
        true_index = class_to_index[true_class]
        for predicted_index in predicted_indices:
            confusion[true_index, int(predicted_index)] += 1

        total_samples += len(test_indices)
        correct_samples += int(np.sum(predicted_indices == true_index))

    background_index = class_to_index[NO_CONTACT_USER]
    background_total = int(np.sum(confusion[background_index, :]))
    background_correct = int(confusion[background_index, background_index])
    background_false_user = int(
        np.sum(confusion[background_index, :background_index])
        + np.sum(confusion[background_index, background_index + 1 :])
    )
    row_totals = np.sum(confusion, axis=1)
    per_class_accuracy = {
        class_name: _rate_or_none(
            int(confusion[index, index]),
            int(row_totals[index]),
        )
        for index, class_name in enumerate(ALL_CLASSES)
    }

    return {
        "registration_count": registration_count,
        "repeat": repeat,
        "class_order": list(ALL_CLASSES),
        "overall_accuracy": _rate_or_none(correct_samples, total_samples),
        "background_accuracy": _rate_or_none(
            background_correct,
            background_total,
        ),
        "background_false_user_rate": _rate_or_none(
            background_false_user,
            background_total,
        ),
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": per_class_accuracy,
        "row_totals": row_totals.tolist(),
        "predicted_totals": np.sum(confusion, axis=0).tolist(),
    }


def run_one_experiment(
    X_features: np.ndarray,
    template_class: str,
    registration_count: int,
    repeat: int,
    registration_indices_by_class: dict[str, np.ndarray],
    test_indices_by_class: dict[str, np.ndarray],
) -> dict:
    registration_indices = registration_indices_by_class[template_class]
    genuine_indices = test_indices_by_class[template_class]
    sequence_template = build_sequence_template(
        X_features[registration_indices]
    )

    genuine_distances = calculate_sequence_distance(
        sequence_template,
        X_features[genuine_indices],
    )
    genuine_accept = genuine_distances <= sequence_template["threshold"]
    genuine_accept_count = int(np.sum(genuine_accept))

    impostor_accept_count = 0
    impostor_total = 0
    impostor_distances = []
    impostor_details = {}

    for impostor_class in ALL_CLASSES:
        if impostor_class == template_class:
            continue

        impostor_indices = test_indices_by_class[impostor_class]
        distances = calculate_sequence_distance(
            sequence_template,
            X_features[impostor_indices],
        )
        accepted = distances <= sequence_template["threshold"]
        accepted_count = int(np.sum(accepted))

        impostor_total += len(impostor_indices)
        impostor_accept_count += accepted_count
        impostor_distances.extend(distances.tolist())
        impostor_details[impostor_class] = {
            "accept": accepted_count,
            "reject": int(len(impostor_indices) - accepted_count),
            "contact": len(impostor_indices),
            "contact_rate": 1.0,
            "average_distance": (
                float(np.mean(distances)) if len(distances) else np.nan
            ),
        }

    gar = genuine_accept_count / TEST_COUNT
    far = impostor_accept_count / impostor_total if impostor_total else 0.0
    genuine_distance = (
        float(np.mean(genuine_distances))
        if len(genuine_distances)
        else np.nan
    )
    impostor_distance = (
        float(np.mean(impostor_distances))
        if impostor_distances
        else np.nan
    )

    return {
        "template_user": template_class,
        "registration_count": registration_count,
        "repeat": repeat,
        "contact_threshold": np.nan,
        "contact_rate": 1.0,
        "contact_miss_rate": 0.0,
        "no_contact_far": np.nan,
        "no_contact_reject": np.nan,
        "sequence_threshold": float(sequence_template["threshold"]),
        "gar": gar,
        "frr": 1.0 - gar,
        "far": far,
        "genuine_contact_count": TEST_COUNT,
        "genuine_accept": genuine_accept_count,
        "genuine_reject": TEST_COUNT - genuine_accept_count,
        "genuine_distance": genuine_distance,
        "impostor_contact_count": impostor_total,
        "impostor_accept": impostor_accept_count,
        "impostor_total": impostor_total,
        "impostor_distance": impostor_distance,
        "impostor_details": impostor_details,
        "standalone_gar": gar,
        "standalone_far": far,
        "standalone_frr": 1.0 - gar,
    }


def print_sequence_statistics_summary(rows: list[dict]) -> None:
    print()
    print("=" * 110)
    print("SEQUENCE STATISTICS ALL-CLASSES SUMMARY")
    print("=" * 110)
    print(
        f"{'Reg':<6}{'User':<15}{'GAR Mean':>14}{'GAR Std':>14}"
        f"{'FRR Mean':>14}{'FAR Mean':>14}{'FAR Std':>14}"
        f"{'Threshold':>18}"
    )
    print("-" * 110)
    for registration_count in REGISTRATION_COUNTS:
        for class_name in ALL_CLASSES:
            selected = [
                row
                for row in rows
                if row["registration_count"] == registration_count
                and row["template_user"] == class_name
            ]
            gar_values = np.asarray(
                [row["standalone_gar"] for row in selected],
                dtype=np.float64,
            )
            far_values = np.asarray(
                [row["standalone_far"] for row in selected],
                dtype=np.float64,
            )
            thresholds = np.asarray(
                [row["sequence_threshold"] for row in selected],
                dtype=np.float64,
            )
            print(
                f"{registration_count:<6}{class_name:<15}"
                f"{np.mean(gar_values) * 100:>10.2f}%"
                f"{np.std(gar_values) * 100:>10.2f}%"
                f"{(1.0 - np.mean(gar_values)) * 100:>10.2f}%"
                f"{np.mean(far_values) * 100:>10.2f}%"
                f"{np.std(far_values) * 100:>10.2f}%"
                f"{np.mean(thresholds):>18.4f}"
            )


def print_classification_summary(rows: list[dict]) -> None:
    print()
    print("=" * 110)
    print("NINE-CLASS DIRECT CLASSIFICATION SUMMARY")
    print("=" * 110)
    print(
        f"{'Reg':<8}{'Repeat':<10}{'Accuracy':>14}"
        f"{'Background Acc.':>20}{'Background False User':>24}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['registration_count']:<8}{row['repeat']:<10}"
            f"{(row['overall_accuracy'] * 100 if row['overall_accuracy'] is not None else np.nan):>13.2f}%"
            f"{(row['background_accuracy'] * 100 if row['background_accuracy'] is not None else np.nan):>19.2f}%"
            f"{(row['background_false_user_rate'] * 100 if row['background_false_user_rate'] is not None else np.nan):>23.2f}%"
        )

    for registration_count in REGISTRATION_COUNTS:
        selected = [
            row for row in rows if row["registration_count"] == registration_count
        ]
        print()
        print(f"Registration={registration_count} cross-repeat statistics")
        for field, label in (
            ("overall_accuracy", "Overall accuracy"),
            ("background_accuracy", "Background accuracy"),
            ("background_false_user_rate", "Background false user rate"),
        ):
            values = np.asarray(
                [
                    row[field]
                    for row in selected
                    if row[field] is not None
                ],
                dtype=np.float64,
            )
            mean = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values)) if len(values) else np.nan
            print(f"  {label:<30}: mean={mean * 100:.2f}%, std={std * 100:.2f}%")

        for row in selected:
            print(f"\nRepeat {row['repeat']} confusion matrix ({row['class_order']})")
            print("true\\pred".ljust(15) + "".join(
                f"{class_name:>12}" for class_name in row["class_order"]
            ))
            for class_name, values in zip(
                row["class_order"],
                row["confusion_matrix"],
            ):
                print(
                    f"{class_name:<15}"
                    + "".join(f"{value:>12}" for value in values)
                )


def _classification_metrics(confusion: np.ndarray) -> dict[str, dict[str, float | int]]:
    true_positive = np.diag(confusion).astype(np.int64)
    support = np.sum(confusion, axis=1).astype(np.int64)
    predicted = np.sum(confusion, axis=0).astype(np.int64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(len(ALL_CLASSES), dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(ALL_CLASSES), dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(ALL_CLASSES), dtype=np.float64),
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


def _classification_output_paths(registration_count: int) -> dict[str, Path]:
    suffix = f"_reg{registration_count}"
    return {
        "repeat_csv": CLASSIFICATION_OUTPUT_DIR / (
            f"sequence_statistics_all_classes_repeat_statistics{suffix}.csv"
        ),
        "per_class_csv": CLASSIFICATION_OUTPUT_DIR / (
            f"sequence_statistics_all_classes_per_class_metrics{suffix}.csv"
        ),
        "aggregate_csv": CLASSIFICATION_OUTPUT_DIR / (
            f"sequence_statistics_all_classes_aggregate_confusion_matrix{suffix}.csv"
        ),
        "normalized_csv": CLASSIFICATION_OUTPUT_DIR / (
            f"sequence_statistics_all_classes_normalized_confusion_matrix{suffix}.csv"
        ),
        "aggregate_png": CLASSIFICATION_OUTPUT_DIR / (
            f"sequence_statistics_all_classes_aggregate_confusion_matrix{suffix}.png"
        ),
        "normalized_png": CLASSIFICATION_OUTPUT_DIR / (
            f"sequence_statistics_all_classes_normalized_confusion_matrix{suffix}.png"
        ),
        "report": CLASSIFICATION_OUTPUT_DIR / (
            f"sequence_statistics_all_classes_final_summary{suffix}.txt"
        ),
    }


def _save_confusion_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    value_format: str,
    value_scale: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, label="Percentage" if value_scale != 1.0 else "Samples")
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
    threshold = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index] * value_scale
            value_text = (
                str(int(round(value)))
                if value_format == "d"
                else format(value, value_format)
            )
            ax.text(
                column_index,
                row_index,
                value_text,
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_classification_artifacts(rows: list[dict]) -> list[dict[str, object]]:
    """Save repeat-level and aggregate classification artifacts."""
    summaries = []
    for registration_count in REGISTRATION_COUNTS:
        selected = sorted(
            (
                row
                for row in rows
                if row["registration_count"] == registration_count
            ),
            key=lambda row: row["repeat"],
        )
        if not selected:
            continue

        paths = _classification_output_paths(registration_count)
        confusion_matrices = np.asarray(
            [row["confusion_matrix"] for row in selected],
            dtype=np.int64,
        )
        if confusion_matrices.shape[1:] != (len(ALL_CLASSES), len(ALL_CLASSES)):
            raise ValueError("Classification confusion matrices must be 9x9.")
        expected_samples = sum(int(row["total_samples"]) for row in selected)
        if not all(
            int(matrix.sum()) == int(row["total_samples"])
            for matrix, row in zip(confusion_matrices, selected)
        ):
            raise ValueError("A repeat confusion matrix total is inconsistent.")

        aggregate = np.sum(confusion_matrices, axis=0)
        if int(aggregate.sum()) != expected_samples:
            raise ValueError("Aggregate confusion matrix total is inconsistent.")
        row_totals = np.sum(aggregate, axis=1)
        normalized = np.divide(
            aggregate,
            row_totals[:, None],
            out=np.zeros_like(aggregate, dtype=np.float64),
            where=row_totals[:, None] != 0,
        )
        per_class = _classification_metrics(aggregate)

        with paths["repeat_csv"].open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "registration_count",
                    "repeat",
                    "overall_accuracy",
                    "background_accuracy",
                    "background_false_user_rate",
                    "total_samples",
                    "correct_samples",
                ],
            )
            writer.writeheader()
            writer.writerows(
                {
                    field: row[field]
                    for field in writer.fieldnames
                }
                for row in selected
            )

        with paths["per_class_csv"].open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "registration_count",
                    "class_name",
                    "precision",
                    "recall",
                    "f1_score",
                    "test_samples",
                    "correct_samples",
                ],
            )
            writer.writeheader()
            writer.writerows(
                {
                    "registration_count": registration_count,
                    "class_name": class_name,
                    **metrics,
                }
                for class_name, metrics in per_class.items()
            )

        with paths["aggregate_csv"].open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["true\\pred", *ALL_CLASSES])
            writer.writerows(
                [class_name, *matrix_row.tolist()]
                for class_name, matrix_row in zip(ALL_CLASSES, aggregate)
            )

        with paths["normalized_csv"].open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["true\\pred", *ALL_CLASSES])
            writer.writerows(
                [class_name, *matrix_row.tolist()]
                for class_name, matrix_row in zip(ALL_CLASSES, normalized)
            )

        _save_confusion_heatmap(
            aggregate,
            f"Aggregate Confusion Matrix (Registration={registration_count}, "
            f"{len(selected)} Repeats)",
            paths["aggregate_png"],
            "d",
        )
        _save_confusion_heatmap(
            normalized,
            f"Row-Normalized Confusion Matrix (Registration={registration_count}, "
            f"{len(selected)} Repeats)",
            paths["normalized_png"],
            ".1f",
            value_scale=100.0,
        )

        repeat_metrics = {
            field: np.asarray(
                [row[field] for row in selected if row[field] is not None],
                dtype=np.float64,
            )
            for field in (
                "overall_accuracy",
                "background_accuracy",
                "background_false_user_rate",
            )
        }
        aggregate_accuracy = float(np.trace(aggregate) / aggregate.sum()) if aggregate.sum() else None
        aggregate_background_total = int(row_totals[-1])
        aggregate_background_accuracy = _rate_or_none(
            int(aggregate[-1, -1]),
            aggregate_background_total,
        )
        aggregate_background_false_user = _rate_or_none(
            int(aggregate[-1, :-1].sum()),
            aggregate_background_total,
        )
        macro_precision = float(np.mean([metrics["precision"] for metrics in per_class.values()]))
        macro_recall = float(np.mean([metrics["recall"] for metrics in per_class.values()]))
        macro_f1 = float(np.mean([metrics["f1_score"] for metrics in per_class.values()]))

        with paths["report"].open("w", encoding="utf-8") as file:
            file.write(f"Registration count: {registration_count}\n")
            file.write(f"Repeat count: {len(selected)}\n")
            file.write(f"Class order (rows=true, columns=predicted): {list(ALL_CLASSES)}\n\n")
            file.write("Repeat metric mean and standard deviation:\n")
            for field, values in repeat_metrics.items():
                mean = float(np.mean(values)) if len(values) else np.nan
                std = float(np.std(values)) if len(values) else np.nan
                file.write(f"  {field}: mean={mean:.8f}, std={std:.8f}\n")
            file.write("\nAggregate confusion-matrix metrics:\n")
            file.write(f"  overall_accuracy: {aggregate_accuracy}\n")
            file.write(f"  background_accuracy: {aggregate_background_accuracy}\n")
            file.write(f"  background_false_user_rate: {aggregate_background_false_user}\n")
            file.write(f"  macro_precision: {macro_precision:.8f}\n")
            file.write(f"  macro_recall: {macro_recall:.8f}\n")
            file.write(f"  macro_f1: {macro_f1:.8f}\n\n")
            file.write("Per-class metrics from aggregate confusion matrix:\n")
            for class_name, metrics in per_class.items():
                file.write(f"  {class_name}: {metrics}\n")
            file.write("\nAggregate confusion matrix:\n")
            file.write("true\\pred," + ",".join(ALL_CLASSES) + "\n")
            for class_name, matrix_row in zip(ALL_CLASSES, aggregate):
                file.write(f"{class_name}," + ",".join(map(str, matrix_row.tolist())) + "\n")
            file.write("\nRow-normalized confusion matrix:\n")
            file.write("true\\pred," + ",".join(ALL_CLASSES) + "\n")
            for class_name, matrix_row in zip(ALL_CLASSES, normalized):
                file.write(f"{class_name}," + ",".join(f"{value:.8f}" for value in matrix_row) + "\n")

        summaries.append(
            {
                "registration_count": registration_count,
                "repeat_count": len(selected),
                "paths": paths,
                "repeat_metrics": repeat_metrics,
                "aggregate_accuracy": aggregate_accuracy,
                "aggregate_background_accuracy": aggregate_background_accuracy,
                "aggregate_background_false_user_rate": aggregate_background_false_user,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }
        )
    return summaries


def print_benchmark_information() -> None:
    print()
    print("=" * 110)
    print("SEQUENCE STATISTICS ALL-CLASSES BENCHMARK")
    print("=" * 110)
    print()
    print("Pipeline:")
    print("    Raw Pressure 50 x 16")
    print("          -> Sequence Statistics")
    print("          -> Euclidean Identity Authentication")
    print()
    print(f"Dataset                     : {DATA_DIR}")
    print(f"Classes                     : {ALL_CLASSES}")
    print(f"Registration counts         : {REGISTRATION_COUNTS}")
    print(f"Test count                  : {TEST_COUNT}")
    print(f"Repeats                     : {REPEATS}")
    print(f"Sequence threshold K        : {SEQUENCE_THRESHOLD_K}")
    print(f"Contact area feature cutoff: {CONTACT_AREA_THRESHOLD}")


def run_benchmark(X: np.ndarray, labels: np.ndarray) -> list[dict]:
    print()
    print("Extracting Sequence Statistics...")
    X_features = extract_dataset_statistics(X)
    print(f"Sequence feature shape : {X_features.shape}")

    if np.isnan(X_features).any():
        raise ValueError("NaN detected in sequence feature dataset.")

    all_rows = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_RESULTS_PATH.write_text("", encoding="utf-8")
    CLASSIFICATION_RESULTS_PATH.write_text("", encoding="utf-8")

    for registration_count in REGISTRATION_COUNTS:
        print()
        print("=" * 110)
        print(f"REGISTRATION = {registration_count}")
        print("=" * 110)
        rng = np.random.default_rng(RANDOM_SEED)

        for repeat in range(1, REPEATS + 1):
            print(f"\nRunning Reg={registration_count}, Repeat={repeat:02d}")
            registration_indices, test_indices = prepare_repeat_split(
                labels,
                registration_count,
                rng,
            )

            classification_result = evaluate_direct_classification(
                X_features=X_features,
                registration_count=registration_count,
                repeat=repeat,
                registration_indices_by_class=registration_indices,
                test_indices_by_class=test_indices,
            )
            with CLASSIFICATION_RESULTS_PATH.open(
                "a",
                encoding="utf-8",
            ) as classification_file:
                classification_file.write(
                    json.dumps(
                        classification_result,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                classification_file.flush()

            for template_class in ALL_CLASSES:
                print(f"    Template={template_class}")
                result = run_one_experiment(
                    X_features=X_features,
                    template_class=template_class,
                    registration_count=registration_count,
                    repeat=repeat,
                    registration_indices_by_class=registration_indices,
                    test_indices_by_class=test_indices,
                )
                all_rows.append(result)
                with LIVE_RESULTS_PATH.open("a", encoding="utf-8") as live_file:
                    live_file.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            allow_nan=True,
                        )
                        + "\n"
                    )
                    live_file.flush()

    return all_rows


def main() -> None:
    print_benchmark_information()
    X, labels = load_data()
    print_dataset_information(X, labels)
    results = run_benchmark(X, labels)
    print_sequence_statistics_summary(results)
    with CLASSIFICATION_RESULTS_PATH.open("r", encoding="utf-8") as classification_file:
        classification_rows = [
            json.loads(line)
            for line in classification_file
            if line.strip()
        ]
    print_classification_summary(classification_rows)
    artifact_summaries = save_classification_artifacts(classification_rows)
    print()
    print("=" * 110)
    print("NINE-CLASS FINAL AGGREGATE STATISTICS")
    print("=" * 110)
    for summary in artifact_summaries:
        repeat_metrics = summary["repeat_metrics"]
        print(
            f"Registration={summary['registration_count']}, "
            f"Repeats={summary['repeat_count']}"
        )
        for field in (
            "overall_accuracy",
            "background_accuracy",
            "background_false_user_rate",
        ):
            values = repeat_metrics[field]
            mean = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values)) if len(values) else np.nan
            print(f"  {field}: repeat mean={mean:.4f}, std={std:.4f}")
        print(
            f"  aggregate overall_accuracy={summary['aggregate_accuracy']}\n"
            f"  aggregate background_accuracy="
            f"{summary['aggregate_background_accuracy']}\n"
            f"  aggregate background_false_user_rate="
            f"{summary['aggregate_background_false_user_rate']}\n"
            f"  aggregate macro_precision={summary['macro_precision']:.4f}\n"
            f"  aggregate macro_recall={summary['macro_recall']:.4f}\n"
            f"  aggregate macro_f1={summary['macro_f1']:.4f}"
        )
        print("  artifacts:")
        for artifact_name, artifact_path in summary["paths"].items():
            print(f"    {artifact_name}: {artifact_path}")
    print()
    print("=" * 110)
    print("Benchmark Finished")
    print("=" * 110)


if __name__ == "__main__":
    main()
