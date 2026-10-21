from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
AI_ROOT = HERE.parents[1]
PROJECT_ROOT = HERE.parents[2]
OUTPUT_DIR = HERE / "contact_gate_confusion_matrix_results"
DETAILS_PATH = OUTPUT_DIR / "sample_decisions.jsonl"
RESULTS_PATH = OUTPUT_DIR / "contact_gate_results.jsonl"

if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

import sequence_statistics_contact_gate_benchmark_live as contact_gate
import sequence_statistics_all_classes_benchmark_live as direct_classifier
from data_loader import load_dataset


ALL_CLASSES = contact_gate.USERS + (contact_gate.NO_CONTACT_USER,)
OUTCOME_CLASSES = ALL_CLASSES + ("rejected", "multiple_accept")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_matrix(path: Path, matrix: np.ndarray, columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *columns])
        writer.writerows(
            [row_name, *row.tolist()]
            for row_name, row in zip(ALL_CLASSES, matrix)
        )


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=totals != 0,
    )


def _heatmap(
    matrix: np.ndarray,
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
    title: str,
    path: Path,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, label="Percentage" if percent else "Samples")
    ax.set(
        title=title,
        xlabel="Predicted outcome",
        ylabel="True class",
        xticks=np.arange(len(column_labels)),
        yticks=np.arange(len(row_labels)),
        xticklabels=column_labels,
        yticklabels=row_labels,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    cutoff = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index] * (100.0 if percent else 1.0)
            text = f"{value:.1f}%" if percent else str(int(round(value)))
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > cutoff else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _side_by_side_heatmap(
    original_matrix: np.ndarray,
    direct_matrix: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    for ax, matrix, columns, title in (
        (
            axes[0],
            original_matrix,
            OUTCOME_CLASSES,
            "Original Contact Gate outcomes",
        ),
        (
            axes[1],
            direct_matrix,
            ALL_CLASSES,
            "Nine-class direct classification",
        ),
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set(
            title=title,
            xlabel="Predicted outcome",
            ylabel="True class",
            xticks=np.arange(len(columns)),
            yticks=np.arange(len(ALL_CLASSES)),
            xticklabels=columns,
            yticklabels=ALL_CLASSES,
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        cutoff = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = int(matrix[row_index, column_index])
                ax.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > cutoff else "black",
                )
        fig.colorbar(image, ax=ax, label="Samples")
    fig.suptitle(
        "Aggregate confusion matrices (rows=true, columns=predicted)",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_models(
    X: np.ndarray,
    X_features: np.ndarray,
    registration_indices_by_user: dict[str, np.ndarray],
    background_calibration_indices: np.ndarray,
    registration_count: int,
    repeat: int,
) -> dict[str, tuple[object, dict]]:
    models = {}
    for user in contact_gate.USERS:
        registration_indices = registration_indices_by_user[user]
        gate = contact_gate.calibrate_gate(
            template_user=user,
            registration_count=registration_count,
            repeat=repeat,
            registration_samples=X[registration_indices],
            background_calibration=X[background_calibration_indices],
        )
        template = contact_gate.build_sequence_template(
            X_features[registration_indices]
        )
        models[user] = (gate, template)
    return models


def _evaluate_repeat(
    X: np.ndarray,
    X_features: np.ndarray,
    labels: np.ndarray,
    registration_count: int,
    repeat: int,
    registration_indices_by_user: dict[str, np.ndarray],
    test_indices_by_user: dict[str, np.ndarray],
    background_calibration_indices: np.ndarray,
    no_contact_test_indices: np.ndarray,
) -> tuple[dict, list[dict]]:
    models = _build_models(
        X,
        X_features,
        registration_indices_by_user,
        background_calibration_indices,
        registration_count,
        repeat,
    )
    matrix = np.zeros((len(ALL_CLASSES), len(OUTCOME_CLASSES)), dtype=np.int64)
    binary = {
        user: np.zeros((2, 2), dtype=np.int64)
        for user in contact_gate.USERS
    }
    details = []
    test_indices_by_class = {
        **test_indices_by_user,
        contact_gate.NO_CONTACT_USER: no_contact_test_indices,
    }

    for true_class in ALL_CLASSES:
        for sample_index in test_indices_by_class[true_class]:
            sample = X[sample_index]
            sample_features = X_features[sample_index]
            template_results = {}
            accepted_users = []
            for user, (gate, template) in models.items():
                contact_score, contact = contact_gate.contact_decision(gate, sample)
                distance = float(
                    contact_gate.calculate_sequence_distance(
                        template,
                        sample_features.reshape(1, -1),
                    )[0]
                )
                sequence_accept = bool(
                    distance <= float(template["threshold"])
                )
                accepted = bool(contact and sequence_accept)
                template_results[user] = {
                    "contact_score": contact_score,
                    "contact": contact,
                    "distance": distance,
                    "sequence_threshold": float(template["threshold"]),
                    "sequence_accept": sequence_accept,
                    "accepted": accepted,
                }
                accepted_users.append(user) if accepted else None

                actual_positive = true_class == user
                binary[user][int(actual_positive), int(accepted)] += 1

            if len(accepted_users) == 1:
                outcome = accepted_users[0]
            elif not accepted_users:
                outcome = "rejected"
            else:
                outcome = "multiple_accept"
            matrix[ALL_CLASSES.index(true_class), OUTCOME_CLASSES.index(outcome)] += 1
            details.append(
                {
                    "registration_count": registration_count,
                    "repeat": repeat,
                    "sample_index": int(sample_index),
                    "true_class": true_class,
                    "contact_gate_decisions": {
                        user: values["contact"]
                        for user, values in template_results.items()
                    },
                    "template_results": template_results,
                    "accepted_users": accepted_users,
                    "final_outcome": outcome,
                }
            )

    original_rows = []
    for user in contact_gate.USERS:
        result = contact_gate.run_one_experiment(
            X=X,
            X_features=X_features,
            labels=labels,
            template_user=user,
            registration_count=registration_count,
            repeat=repeat,
            registration_indices_by_user=registration_indices_by_user,
            test_indices_by_user=test_indices_by_user,
            background_calibration_indices=background_calibration_indices,
            no_contact_test_indices=no_contact_test_indices,
        )
        original_rows.append(result)

    return (
        {
            "registration_count": registration_count,
            "repeat": repeat,
            "class_order": list(ALL_CLASSES),
            "outcome_order": list(OUTCOME_CLASSES),
            "confusion_matrix": matrix.tolist(),
            "total_samples": int(matrix.sum()),
            "rejected_samples": int(matrix[:, OUTCOME_CLASSES.index("rejected")].sum()),
            "multiple_accept_samples": int(
                matrix[:, OUTCOME_CLASSES.index("multiple_accept")].sum()
            ),
            "binary_by_template": {
                user: values.tolist() for user, values in binary.items()
            },
            "original_rows": original_rows,
        },
        details,
    )


def _compare_split_conditions(labels: np.ndarray) -> dict:
    original_rng = np.random.default_rng(contact_gate.RANDOM_SEED)
    direct_rng = np.random.default_rng(direct_classifier.RANDOM_SEED)
    original = contact_gate.prepare_repeat_split(
        np.zeros((len(labels), 50, 16), dtype=np.float32),
        labels,
        contact_gate.REGISTRATION_COUNTS[0],
        original_rng,
    )
    direct = direct_classifier.prepare_repeat_split(
        labels,
        direct_classifier.REGISTRATION_COUNTS[0],
        direct_rng,
    )
    original_tests = {
        **original[1],
        contact_gate.NO_CONTACT_USER: original[3],
    }
    direct_tests = direct[1]
    return {
        class_name: bool(
            np.array_equal(
                np.sort(original_tests[class_name]),
                np.sort(direct_tests[class_name]),
            )
        )
        for class_name in ALL_CLASSES
    }


def _write_comparison(
    original_aggregate: np.ndarray,
    direct_aggregate: np.ndarray,
    split_match: dict[str, bool],
) -> None:
    output = OUTPUT_DIR / "contact_gate_vs_direct_classification_comparison.txt"
    direct_total = int(direct_aggregate.sum())
    original_total = int(original_aggregate.sum())
    with output.open("w", encoding="utf-8") as file:
        file.write("Contact Gate original method vs nine-class direct classification\n")
        file.write("The Contact Gate method does not define a unique multiclass prediction.\n")
        file.write("Exactly one accepted template is shown as that identity; zero is rejected; ")
        file.write("more than one is multiple_accept.\n\n")
        file.write(f"Class order: {list(ALL_CLASSES)}\n")
        file.write(f"Original outcome columns: {list(OUTCOME_CLASSES)}\n\n")
        file.write("Repeat/test split comparison with the current direct benchmark:\n")
        for class_name, matched in split_match.items():
            file.write(f"  {class_name}: {'same' if matched else 'different'}\n")
        file.write(
            "\nThe formal user test samples match. The background test samples differ "
            "because the original method reserves 20 background calibration samples, "
            "while the direct benchmark uses 80 background registration samples.\n"
        )
        file.write(
            f"\nOriginal aggregate outcome total: {original_total}\n"
            f"Direct aggregate nine-class total: {direct_total}\n"
        )
        file.write(
            "\nDo not compare the original outcome accuracy directly with direct "
            "classification accuracy without accounting for rejected/multiple_accept "
            "outcomes and the background split difference.\n"
        )

    _heatmap(
        original_aggregate,
        ALL_CLASSES,
        OUTCOME_CLASSES,
        "Original Contact Gate Outcome Matrix",
        OUTPUT_DIR / "contact_gate_outcome_matrix.png",
    )
    _heatmap(
        _row_normalize(original_aggregate),
        ALL_CLASSES,
        OUTCOME_CLASSES,
        "Original Contact Gate Outcome Matrix (Row-Normalized)",
        OUTPUT_DIR / "contact_gate_outcome_matrix_normalized.png",
        percent=True,
    )
    _heatmap(
        direct_aggregate,
        ALL_CLASSES,
        ALL_CLASSES,
        "Nine-Class Direct Classification Matrix",
        OUTPUT_DIR / "direct_classification_matrix_side.png",
    )
    _side_by_side_heatmap(
        original_aggregate,
        direct_aggregate,
        OUTPUT_DIR / "contact_gate_vs_direct_side_by_side.png",
    )


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, labels = contact_gate.load_data()
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    X_features = contact_gate.extract_dataset_statistics(X)
    DETAILS_PATH.write_text("", encoding="utf-8")
    RESULTS_PATH.write_text("", encoding="utf-8")

    all_results = []
    rng = np.random.default_rng(contact_gate.RANDOM_SEED)
    for registration_count in contact_gate.REGISTRATION_COUNTS:
        for repeat in range(1, contact_gate.REPEATS + 1):
            split = contact_gate.prepare_repeat_split(
                X,
                labels,
                registration_count,
                rng,
            )
            result, details = _evaluate_repeat(
                X,
                X_features,
                labels,
                registration_count,
                repeat,
                *split,
            )
            all_results.append(result)
            with RESULTS_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, allow_nan=True) + "\n")
            with DETAILS_PATH.open("a", encoding="utf-8") as file:
                for detail in details:
                    file.write(json.dumps(detail, allow_nan=True) + "\n")

    for registration_count in contact_gate.REGISTRATION_COUNTS:
        selected = [
            row for row in all_results
            if row["registration_count"] == registration_count
        ]
        aggregate = np.sum(
            np.asarray([row["confusion_matrix"] for row in selected], dtype=np.int64),
            axis=0,
        )
        _write_matrix(
            OUTPUT_DIR / "contact_gate_outcome_matrix.csv",
            aggregate,
            OUTCOME_CLASSES,
        )
        _write_matrix(
            OUTPUT_DIR / "contact_gate_outcome_matrix_normalized.csv",
            _row_normalize(aggregate),
            OUTCOME_CLASSES,
        )
        with (
            OUTPUT_DIR / "contact_gate_binary_confusion_by_template.csv"
        ).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "template_user",
                "true_positive",
                "false_negative",
                "false_positive",
                "true_negative",
                "test_samples",
            ])
            binary_totals = {
                user: np.zeros((2, 2), dtype=np.int64)
                for user in contact_gate.USERS
            }
            for row in selected:
                for user, values in row["binary_by_template"].items():
                    binary_totals[user] += np.asarray(values, dtype=np.int64)
            for user, values in binary_totals.items():
                writer.writerow([
                    user,
                    int(values[1, 1]),
                    int(values[1, 0]),
                    int(values[0, 1]),
                    int(values[0, 0]),
                    int(values.sum()),
                ])
        repeat_stats_path = OUTPUT_DIR / "contact_gate_repeat_statistics.csv"
        with repeat_stats_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "registration_count",
                "repeat",
                "total_samples",
                "rejected_samples",
                "multiple_accept_samples",
                "unique_accept_samples",
            ])
            for row in selected:
                writer.writerow([
                    row["registration_count"],
                    row["repeat"],
                    row["total_samples"],
                    row["rejected_samples"],
                    row["multiple_accept_samples"],
                    row["total_samples"]
                    - row["rejected_samples"]
                    - row["multiple_accept_samples"],
                ])

        direct_path = direct_classifier.CLASSIFICATION_RESULTS_PATH
        direct_rows = [
            json.loads(line)
            for line in direct_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        direct_aggregate = np.sum(
            np.asarray([row["confusion_matrix"] for row in direct_rows], dtype=np.int64),
            axis=0,
        )
        split_match = _compare_split_conditions(labels)
        _write_comparison(aggregate, direct_aggregate, split_match)

        report = OUTPUT_DIR / "contact_gate_confusion_matrix_report.txt"
        with report.open("w", encoding="utf-8") as file:
            file.write("Original Contact Gate + Sequence Statistics report\n")
            file.write(f"Registration count: {registration_count}\n")
            file.write(f"Repeat count: {len(selected)}\n")
            file.write(f"Class order: {list(ALL_CLASSES)}\n")
            file.write(f"Outcome order: {list(OUTCOME_CLASSES)}\n")
            file.write("Rows are true classes; columns are final diagnostic outcomes.\n\n")
            file.write(
                "This is not a replacement multiclass classifier. The original "
                "algorithm evaluates each user template independently.\n\n"
            )
            file.write(f"Aggregate matrix total: {int(aggregate.sum())}\n")
            file.write(f"Rejected total: {int(aggregate[:, -2].sum())}\n")
            file.write(f"Multiple-accept total: {int(aggregate[:, -1].sum())}\n\n")
            for user in contact_gate.USERS:
                values = [row["original_rows"] for row in selected]
                user_rows = [
                    item for repeat_rows in values for item in repeat_rows
                    if item["template_user"] == user
                ]
                file.write(
                    f"{user}: GAR mean={np.mean([r['gar'] for r in user_rows]):.8f}, "
                    f"FRR mean={np.mean([r['frr'] for r in user_rows]):.8f}, "
                    f"FAR mean={np.mean([r['far'] for r in user_rows]):.8f}, "
                    f"Threshold mean={np.mean([r['sequence_threshold'] for r in user_rows]):.8f}\n"
                )


if __name__ == "__main__":
    run()
