from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "eer_experiment_fixed_dataset"
PLOTS_DIR = OUTPUT_DIR / "plots"

from data_loader import load_csv
from sequence_statistics_contact_gate_benchmark_live import (
    extract_dataset_statistics,
    REGISTRATION_RATIO,
)

ENROLLED_USERS = (
    "Amber",
    "Jay",
    "666",
    "Andy",
    "Apple",
    "Reyna",
    "Tiffany",
)
UNKNOWN_USER = "Joanna"
EXCLUDED_USER = "background"
RANDOM_SEED = 42
REPEAT = 1
RF_N_ESTIMATORS = 100
RF_CLASS_WEIGHT = None
RF_N_JOBS = -1
THRESHOLDS = np.linspace(0.0, 1.0, 1001)
UNKNOWN = "Unknown"


def sha256_matrix(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_dataset() -> list[dict]:
    records: list[dict] = []
    for path in sorted(DATA_DIR.rglob("*.csv")):
        if any(part.lower() in {"legacy", "legency"} for part in path.parts):
            raise ValueError(f"Refusing to read legacy/legency file: {path}")

        try:
            data, csv_label = load_csv(path)
            valid = data.shape == (50, 16)
            if not valid:
                raise ValueError(f"Invalid shape {data.shape}")

            folder_label = path.parent.name
            if folder_label != csv_label:
                raise ValueError(
                    f"Folder/label mismatch: folder={folder_label}, "
                    f"CSV label={csv_label}, path={path}"
                )

            records.append(
                {
                    "sample_id": sha256_file(path),
                    "csv_path": str(path),
                    "csv_relative_path": path.relative_to(DATA_DIR).as_posix(),
                    "folder_label": folder_label,
                    "user_label": str(csv_label),
                    "normalized_label": str(csv_label),
                    "dataset_split": "Excluded",
                    "is_valid": True,
                    "matrix_hash": sha256_matrix(data),
                    "file_hash": sha256_file(path),
                    "split_random_seed": "",
                    "frame_count": int(data.shape[0]),
                    "sensor_count": int(data.shape[1]),
                }
            )
        except Exception as error:
            records.append(
                {
                    "sample_id": sha256_file(path),
                    "csv_path": str(path),
                    "csv_relative_path": path.relative_to(DATA_DIR).as_posix(),
                    "folder_label": path.parent.name,
                    "user_label": "READ_ERROR",
                    "normalized_label": "READ_ERROR",
                    "dataset_split": "Excluded",
                    "is_valid": False,
                    "matrix_hash": "",
                    "file_hash": sha256_file(path),
                    "split_random_seed": "",
                    "frame_count": "",
                    "sensor_count": "",
                    "error": str(error),
                }
            )

    return records


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def integrity_check(records: list[dict]) -> None:
    valid = [record for record in records if record["is_valid"]]

    sample_ids = [record["sample_id"] for record in valid]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError("Duplicate Sample ID detected")

    matrix_hashes = [record["matrix_hash"] for record in valid]
    duplicate_matrices = len(matrix_hashes) - len(set(matrix_hashes))
    if duplicate_matrices:
        raise AssertionError(
            f"Duplicate pressure matrix detected: {duplicate_matrices} duplicate(s)"
        )

    split_groups: dict[str, list[str]] = {}
    for record in valid:
        split_groups.setdefault(record["dataset_split"], []).append(
            record["sample_id"]
        )

    splits = ("Training", "Genuine Test", "Unknown Test", "Excluded")
    for split in splits:
        ids = split_groups.get(split, [])
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Repeated Sample ID in split {split}")

    for left_index, left_split in enumerate(splits):
        left_ids = set(split_groups.get(left_split, []))
        for right_split in splits[left_index + 1 :]:
            right_ids = set(split_groups.get(right_split, []))
            overlap = left_ids & right_ids
            if overlap:
                raise AssertionError(
                    f"Sample overlap between {left_split} and {right_split}: "
                    f"{len(overlap)}"
                )


def find_eer(
    far_curve: np.ndarray,
    frr_curve: np.ndarray,
) -> dict:
    difference = far_curve - frr_curve
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-12))
    if exact.size:
        index = int(exact[np.argmin(np.abs(difference[exact]))])
        far = float(far_curve[index])
        frr = float(frr_curve[index])
        return {
            "eer": float((far + frr) / 2.0),
            "eer_threshold": float(THRESHOLDS[index]),
            "far_at_eer": far,
            "frr_at_eer": frr,
            "method": "exact",
            "nearest_threshold": float(THRESHOLDS[index]),
        }

    for index in range(len(THRESHOLDS) - 1):
        left = float(difference[index])
        right = float(difference[index + 1])
        if left * right < 0:
            x0 = float(THRESHOLDS[index])
            x1 = float(THRESHOLDS[index + 1])
            d0 = float(difference[index])
            d1 = float(difference[index + 1])
            threshold = x0 + (0.0 - d0) * (x1 - x0) / (d1 - d0)
            far = float(
                far_curve[index]
                + (threshold - x0)
                * (far_curve[index + 1] - far_curve[index])
                / (x1 - x0)
            )
            frr = float(
                frr_curve[index]
                + (threshold - x0)
                * (frr_curve[index + 1] - frr_curve[index])
                / (x1 - x0)
            )
            return {
                "eer": float((far + frr) / 2.0),
                "eer_threshold": threshold,
                "far_at_eer": far,
                "frr_at_eer": frr,
                "method": "linear_interpolation",
                "nearest_threshold": float(
                    THRESHOLDS[
                        int(np.argmin(np.abs(difference)))
                    ]
                ),
            }

    index = int(np.argmin(np.abs(difference)))
    return {
        "eer": None,
        "eer_threshold": float(THRESHOLDS[index]),
        "far_at_eer": float(far_curve[index]),
        "frr_at_eer": float(frr_curve[index]),
        "method": "no_crossing_nearest",
        "nearest_threshold": float(THRESHOLDS[index]),
    }


def interpolate_curve(
    threshold: float,
    curve: np.ndarray,
) -> float:
    index = int(np.searchsorted(THRESHOLDS, threshold))
    if index <= 0:
        return float(curve[0])
    if index >= len(THRESHOLDS):
        return float(curve[-1])
    left = float(THRESHOLDS[index - 1])
    right = float(THRESHOLDS[index])
    fraction = (threshold - left) / (right - left)
    return float(curve[index - 1] + fraction * (curve[index] - curve[index - 1]))


def save_confusion_matrix(
    true_labels: np.ndarray,
    genuine_confidence: np.ndarray,
    genuine_predictions: np.ndarray,
    unknown_confidence: np.ndarray,
    unknown_predictions: np.ndarray,
    threshold: float,
) -> np.ndarray:
    class_names = [*ENROLLED_USERS, "Rejected"]
    row_names = [*ENROLLED_USERS, f"Unknown: {UNKNOWN_USER}"]
    matrix = np.zeros((len(row_names), len(class_names)), dtype=np.int64)
    row_index = {name: index for index, name in enumerate(row_names)}
    column_index = {name: index for index, name in enumerate(class_names)}

    genuine_accepted = genuine_confidence >= threshold
    for true_label, prediction, accepted in zip(
        true_labels,
        genuine_predictions,
        genuine_accepted,
    ):
        outcome = str(prediction) if accepted else "Rejected"
        matrix[row_index[str(true_label)], column_index[outcome]] += 1

    unknown_accepted = unknown_confidence >= threshold
    for prediction, accepted in zip(unknown_predictions, unknown_accepted):
        outcome = str(prediction) if accepted else "Rejected"
        matrix[row_index[f"Unknown: {UNKNOWN_USER}"], column_index[outcome]] += 1

    fig, axis = plt.subplots(figsize=(11, 9))
    display = matrix.astype(np.float64)
    display_percent = display / display.sum(axis=1, keepdims=True)
    display_percent = np.nan_to_num(display_percent)
    image = axis.imshow(display_percent, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=axis, label="Row Percentage")
    axis.set(
        title=(
            "Genuine and Unknown Acceptance/Rejection\n"
            f"EER Threshold {threshold:.3f} "
            f"(Unknown test user: {UNKNOWN_USER})"
        ),
        xlabel="Accepted Identity / Rejected",
        ylabel="True Registered User / Unknown Test",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(row_names)),
        xticklabels=class_names,
        yticklabels=row_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = int(matrix[row, column])
            percent = display_percent[row, column] * 100
            axis.text(
                column,
                row,
                f"{count}\n{percent:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "genuine_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return matrix


def _run_once() -> None:
    print("=" * 110)
    print("FIXED DATASET RANDOM FOREST GAR/FAR/FRR/EER EXPERIMENT")
    print("=" * 110)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    records = inventory_dataset()

    invalid = [record for record in records if not record["is_valid"]]
    if invalid:
        raise AssertionError(f"{len(invalid)} invalid CSV(s) detected")

    training_records: list[dict] = []
    genuine_records: list[dict] = []
    unknown_records: list[dict] = []

    per_user_split: dict[str, dict[str, int]] = {}
    for record in records:
        if record["normalized_label"] in ENROLLED_USERS:
            continue
        if record["normalized_label"] == UNKNOWN_USER:
            record["dataset_split"] = "Unknown Test"
            record["split_random_seed"] = RANDOM_SEED
            unknown_records.append(record)
        elif record["normalized_label"] == EXCLUDED_USER:
            record["dataset_split"] = "Excluded"
            record["split_random_seed"] = ""

    training_paths: set[str] = set()
    genuine_paths: set[str] = set()
    for user in ENROLLED_USERS:
        user_records = [
            record
            for record in records
            if record["normalized_label"] == user
        ]
        total_count = len(user_records)
        training_count = int(total_count * REGISTRATION_RATIO)
        test_count = total_count - training_count
        if training_count < 1 or test_count < 1:
            raise ValueError(
                f"User {user} needs at least one training and test sample; "
                f"got {total_count}."
            )
        indices = np.arange(total_count)
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_count,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        for index in train_indices:
            record = user_records[int(index)]
            record["dataset_split"] = "Training"
            record["split_random_seed"] = RANDOM_SEED
            training_records.append(record)
            training_paths.add(record["csv_relative_path"])
        for index in test_indices:
            record = user_records[int(index)]
            record["dataset_split"] = "Genuine Test"
            record["split_random_seed"] = RANDOM_SEED
            genuine_records.append(record)
            genuine_paths.add(record["csv_relative_path"])
        per_user_split[user] = {
            "total": len(user_records),
            "training": len(train_indices),
            "genuine_test": len(test_indices),
            "unknown_test": 0,
        }

    per_user_split[UNKNOWN_USER] = {
        "total": len(unknown_records),
        "training": 0,
        "genuine_test": 0,
        "unknown_test": len(unknown_records),
    }
    per_user_split[EXCLUDED_USER] = {
        "total": sum(
            1 for record in records
            if record["normalized_label"] == EXCLUDED_USER
        ),
        "training": 0,
        "genuine_test": 0,
        "unknown_test": 0,
        "excluded": sum(
            1 for record in records
            if record["normalized_label"] == EXCLUDED_USER
        ),
    }

    expected_training_count = sum(
        split["training"] for split in per_user_split.values()
    )
    expected_genuine_count = sum(
        split["genuine_test"] for split in per_user_split.values()
    )
    if len(training_records) != expected_training_count:
        raise AssertionError("Training count does not match per-user splits")
    if len(genuine_records) != expected_genuine_count:
        raise AssertionError("Genuine test count does not match per-user splits")
    if len(unknown_records) != per_user_split[UNKNOWN_USER]["unknown_test"]:
        raise AssertionError("Unknown test count does not match inventory")

    integrity_check(records)

    training_matrix: list[np.ndarray] = []
    training_labels: list[str] = []
    genuine_matrix: list[np.ndarray] = []
    genuine_labels: list[str] = []
    unknown_matrix: list[np.ndarray] = []

    for record in training_records:
        data, _ = load_csv(record["csv_path"])
        training_matrix.append(data)
        training_labels.append(record["normalized_label"])
    for record in genuine_records:
        data, _ = load_csv(record["csv_path"])
        genuine_matrix.append(data)
        genuine_labels.append(record["normalized_label"])
    for record in unknown_records:
        data, _ = load_csv(record["csv_path"])
        unknown_matrix.append(data)

    X_train_raw = np.asarray(training_matrix, dtype=np.float32)
    X_genuine_raw = np.asarray(genuine_matrix, dtype=np.float32)
    X_unknown_raw = np.asarray(unknown_matrix, dtype=np.float32)
    y_train = np.asarray(training_labels)
    y_genuine = np.asarray(genuine_labels)

    X_train_features = extract_dataset_statistics(X_train_raw)
    X_genuine_features = extract_dataset_statistics(X_genuine_raw)
    X_unknown_features = extract_dataset_statistics(X_unknown_raw)

    if X_train_features.shape[1] != 67:
        raise ValueError(
            f"Expected 67 features, got {X_train_features.shape[1]}"
        )
    if np.isnan(X_train_features).any():
        raise ValueError("NaN in training Sequence Statistics features")

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_SEED,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=RF_N_JOBS,
    )
    model.fit(X_train_features, y_train)

    joblib.dump(
        {
            "model": model,
            "enrolled_users": list(ENROLLED_USERS),
            "unknown_user": UNKNOWN_USER,
            "feature_dimension": 67,
            "random_seed": RANDOM_SEED,
            "rf_parameters": {
                "n_estimators": RF_N_ESTIMATORS,
                "random_state": RANDOM_SEED,
                "class_weight": RF_CLASS_WEIGHT,
                "n_jobs": RF_N_JOBS,
            },
            "feature_pipeline": "sequence_statistics_contact_gate_benchmark_live.extract_dataset_statistics",
        },
        OUTPUT_DIR / "random_forest_model.pkl",
    )

    genuine_probabilities = model.predict_proba(X_genuine_features)
    genuine_indices = np.argmax(genuine_probabilities, axis=1)
    genuine_predictions = model.classes_[genuine_indices]
    genuine_confidence = genuine_probabilities[
        np.arange(len(genuine_probabilities)), genuine_indices
    ]

    unknown_probabilities = model.predict_proba(X_unknown_features)
    unknown_indices = np.argmax(unknown_probabilities, axis=1)
    unknown_predictions = model.classes_[unknown_indices]
    unknown_confidence = unknown_probabilities[
        np.arange(len(unknown_probabilities)), unknown_indices
    ]

    genuine_prediction_rows: list[dict] = []
    for record_index, (record, true_label, prediction, confidence) in enumerate(
        zip(
        genuine_records,
        y_genuine,
        genuine_predictions,
        genuine_confidence,
        )
    ):
        row = {
            "sample_id": record["sample_id"],
            "csv_path": record["csv_relative_path"],
            "true_label": str(true_label),
            "predicted_identity": str(prediction),
            "max_confidence": float(confidence),
        }
        for class_name, probability in zip(
            model.classes_,
            genuine_probabilities[record_index],
        ):
            row[f"probability_{class_name}"] = float(probability)
        genuine_prediction_rows.append(row)

    unknown_prediction_rows: list[dict] = []
    for record_index, (record, prediction, confidence) in enumerate(
        zip(unknown_records, unknown_predictions, unknown_confidence)
    ):
        row = {
            "sample_id": record["sample_id"],
            "csv_path": record["csv_relative_path"],
            "true_label": UNKNOWN,
            "predicted_identity": str(prediction),
            "max_confidence": float(confidence),
        }
        for class_name, probability in zip(
            model.classes_,
            unknown_probabilities[record_index],
        ):
            row[f"probability_{class_name}"] = float(probability)
        unknown_prediction_rows.append(row)

    genuine_fields = [
        "sample_id",
        "csv_path",
        "true_label",
        "predicted_identity",
        "max_confidence",
        *[f"probability_{name}" for name in model.classes_],
    ]
    unknown_fields = [
        "sample_id",
        "csv_path",
        "true_label",
        "predicted_identity",
        "max_confidence",
        *[f"probability_{name}" for name in model.classes_],
    ]
    write_csv(
        OUTPUT_DIR / "genuine_predictions.csv",
        genuine_fields,
        genuine_prediction_rows,
    )
    write_csv(
        OUTPUT_DIR / "unknown_predictions.csv",
        unknown_fields,
        unknown_prediction_rows,
    )

    threshold_rows: list[dict] = []
    gar_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
    far_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
    frr_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
    genuine_reject_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
    misidentification_curve = np.empty(len(THRESHOLDS), dtype=np.float64)

    for index, threshold in enumerate(THRESHOLDS):
        genuine_passed = genuine_confidence >= threshold
        correct_accept = np.sum(
            genuine_passed & (genuine_predictions == y_genuine)
        )
        genuine_rejected = np.sum(
            ~genuine_passed
        )
        misidentified = np.sum(
            genuine_passed & (genuine_predictions != y_genuine)
        )
        unknown_accepted = np.sum(unknown_confidence >= threshold)
        unknown_rejected = len(unknown_confidence) - unknown_accepted

        gar_curve[index] = correct_accept / len(y_genuine)
        far_curve[index] = unknown_accepted / len(unknown_confidence)
        frr_curve[index] = 1.0 - gar_curve[index]
        genuine_reject_curve[index] = genuine_rejected / len(y_genuine)
        misidentification_curve[index] = misidentified / len(y_genuine)

        threshold_rows.append(
            {
                "threshold": float(threshold),
                "GAR": float(gar_curve[index]),
                "FAR": float(far_curve[index]),
                "FRR": float(frr_curve[index]),
                "genuine_rejection_rate": float(
                    genuine_reject_curve[index]
                ),
                "genuine_misidentification_rate": float(
                    misidentification_curve[index]
                ),
                "genuine_accepted_count": int(correct_accept),
                "genuine_rejected_count": int(genuine_rejected),
                "genuine_misidentified_count": int(misidentified),
                "unknown_accepted_count": int(unknown_accepted),
                "unknown_rejected_count": int(unknown_rejected),
                "total_genuine_samples": int(len(y_genuine)),
                "total_unknown_samples": int(len(unknown_confidence)),
            }
        )

    write_csv(
        OUTPUT_DIR / "threshold_metrics.csv",
        [
            "threshold",
            "GAR",
            "FAR",
            "FRR",
            "genuine_rejection_rate",
            "genuine_misidentification_rate",
            "genuine_accepted_count",
            "genuine_rejected_count",
            "genuine_misidentified_count",
            "unknown_accepted_count",
            "unknown_rejected_count",
            "total_genuine_samples",
            "total_unknown_samples",
        ],
        threshold_rows,
    )

    eer = find_eer(far_curve, frr_curve)
    eer_threshold = eer["eer_threshold"]

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(THRESHOLDS * 100, far_curve * 100, label="FAR", color="tab:red")
    axis.plot(THRESHOLDS * 100, frr_curve * 100, label="FRR", color="tab:blue")
    if eer["eer"] is not None:
        x = eer_threshold * 100
        axis.axvline(x, color="tab:green", linestyle="--", label=f"EER threshold {x:.2f}%")
        axis.scatter([x], [eer["eer"] * 100], color="tab:green", zorder=5)
        axis.annotate(
            f"EER {eer['eer'] * 100:.2f}%\n"
            f"Threshold {x:.2f}%\n"
            f"FAR {eer['far_at_eer'] * 100:.2f}%\n"
            f"FRR {eer['frr_at_eer'] * 100:.2f}%",
            xy=(x, eer["eer"] * 100),
            xytext=(12, 30),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->"},
        )
    axis.set_title("Fixed Dataset FAR / FRR Curve")
    axis.set_xlabel("Confidence Threshold (%)")
    axis.set_ylabel("Error Rate (%)")
    axis.set_ylim(0, 100)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "far_frr_curve.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(THRESHOLDS * 100, gar_curve * 100, label="GAR", color="tab:green")
    axis.plot(THRESHOLDS * 100, far_curve * 100, label="FAR", color="tab:red")
    axis.set_title("Fixed Dataset GAR / FAR Curve")
    axis.set_xlabel("Confidence Threshold (%)")
    axis.set_ylabel("Rate (%)")
    axis.set_ylim(0, 100)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gar_far_curve.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.hist(genuine_confidence * 100, bins=20, alpha=0.65, label="Genuine Test", edgecolor="black")
    axis.hist(unknown_confidence * 100, bins=20, alpha=0.65, label="Unknown Test", edgecolor="black")
    axis.set_title("Genuine / Unknown Confidence Distribution")
    axis.set_xlabel("Maximum Confidence (%)")
    axis.set_ylabel("Sample Count")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confidence_distribution.png", dpi=180)
    plt.close(fig)

    confusion = save_confusion_matrix(
        y_genuine,
        genuine_confidence,
        genuine_predictions,
        unknown_confidence,
        unknown_predictions,
        eer_threshold,
    )
    with (OUTPUT_DIR / "genuine_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true\\outcome", *ENROLLED_USERS, "Rejected"])
        for user, row in zip(
            [*ENROLLED_USERS, f"Unknown: {UNKNOWN_USER}"],
            confusion,
        ):
            writer.writerow([user, *row.tolist()])

    gar_at_eer = interpolate_curve(eer_threshold, gar_curve)
    far_at_eer = interpolate_curve(eer_threshold, far_curve)
    frr_at_eer = interpolate_curve(eer_threshold, frr_curve)
    genuine_reject_at_eer = interpolate_curve(
        eer_threshold, genuine_reject_curve
    )
    misidentification_at_eer = interpolate_curve(
        eer_threshold, misidentification_curve
    )

    raw_accuracy = float(np.mean(genuine_predictions == y_genuine))

    per_user_rows: list[dict] = []
    for user in ENROLLED_USERS:
        indices = y_genuine == user
        user_total = int(np.sum(indices))
        user_confidence = genuine_confidence[indices]
        user_predictions = genuine_predictions[indices]
        user_true = y_genuine[indices]
        passed = user_confidence >= eer_threshold
        correct_accept = int(np.sum(passed & (user_predictions == user_true)))
        rejected = int(np.sum(~passed))
        misidentified = int(
            np.sum(passed & (user_predictions != user_true))
        )
        per_user_rows.append(
            {
                "user": user,
                "genuine_test_total": user_total,
                "gar": correct_accept / user_total if user_total else 0.0,
                "frr": (rejected + misidentified) / user_total if user_total else 0.0,
                "genuine_rejection_rate": rejected / user_total if user_total else 0.0,
                "genuine_misidentification_rate": (
                    misidentified / user_total if user_total else 0.0
                ),
                "correct_accept_count": correct_accept,
                "rejected_count": rejected,
                "misidentified_accept_count": misidentified,
            }
        )

    unknown_predicted_counts = {
        user: int(np.sum(unknown_predictions == user))
        for user in ENROLLED_USERS
    }
    for row in per_user_rows:
        count = unknown_predicted_counts[row["user"]]
        row["unknown_predicted_as_this_user_count"] = count
        row["unknown_predicted_as_this_user_rate"] = (
            count / len(unknown_predictions)
        )
        accepted_count = int(
            np.sum(
                (unknown_predictions == row["user"])
                & (unknown_confidence >= eer_threshold)
            )
        )
        row["unknown_accepted_as_this_user_count"] = accepted_count
        row["unknown_accepted_as_this_user_rate"] = (
            accepted_count / len(unknown_predictions)
        )

    write_csv(
        OUTPUT_DIR / "per_user_metrics.csv",
        [
            "user",
            "genuine_test_total",
            "gar",
            "frr",
            "genuine_rejection_rate",
            "genuine_misidentification_rate",
            "correct_accept_count",
            "rejected_count",
            "misidentified_accept_count",
            "unknown_predicted_as_this_user_count",
            "unknown_predicted_as_this_user_rate",
            "unknown_accepted_as_this_user_count",
            "unknown_accepted_as_this_user_rate",
        ],
        per_user_rows,
    )

    write_csv(
        OUTPUT_DIR / "dataset_manifest.csv",
        [
            "sample_id",
            "csv_path",
            "csv_relative_path",
            "folder_label",
            "user_label",
            "normalized_label",
            "dataset_split",
            "is_valid",
            "matrix_hash",
            "file_hash",
            "split_random_seed",
            "frame_count",
            "sensor_count",
        ],
        records,
    )

    split_summary = []
    for user in (*ENROLLED_USERS, UNKNOWN_USER, EXCLUDED_USER):
        user_records = [
            record for record in records
            if record["normalized_label"] == user
        ]
        split_summary.append(
            {
                "user": user,
                "total_valid": len(user_records),
                "training": sum(
                    record["dataset_split"] == "Training"
                    for record in user_records
                ),
                "genuine_test": sum(
                    record["dataset_split"] == "Genuine Test"
                    for record in user_records
                ),
                "unknown_test": sum(
                    record["dataset_split"] == "Unknown Test"
                    for record in user_records
                ),
                "excluded": sum(
                    record["dataset_split"] == "Excluded"
                    for record in user_records
                ),
            }
        )
        split_summary[-1]["registration_ratio"] = (
            split_summary[-1]["training"] / len(user_records)
            if user_records else 0.0
        )
        split_summary[-1]["test_ratio"] = (
            split_summary[-1]["genuine_test"] / len(user_records)
            if user_records else 0.0
        )
    write_csv(
        OUTPUT_DIR / "dataset_split_summary.csv",
        [
            "user",
            "total_valid",
            "training",
            "genuine_test",
            "unknown_test",
            "excluded",
            "registration_ratio",
            "test_ratio",
        ],
        split_summary,
    )
    inventory_rows = []
    for user in (*ENROLLED_USERS, UNKNOWN_USER, EXCLUDED_USER):
        user_records = [
            record for record in records
            if record["normalized_label"] == user
        ]
        inventory_rows.append(
            {
                "user": user,
                "valid_csv_count": sum(
                    record["is_valid"] for record in user_records
                ),
                "frame_shape_valid": all(
                    record["frame_count"] == 50
                    and record["sensor_count"] == 16
                    for record in user_records
                ),
                "folder_label_matches_csv_label": all(
                    not record["mismatch"]
                    if "mismatch" in record
                    else record["folder_label"] == record["normalized_label"]
                    for record in user_records
                ),
                "duplicate_file_count": 0,
                "duplicate_matrix_count": 0,
            }
        )
    write_csv(
        OUTPUT_DIR / "dataset_inventory.csv",
        [
            "user",
            "valid_csv_count",
            "frame_shape_valid",
            "folder_label_matches_csv_label",
            "duplicate_file_count",
            "duplicate_matrix_count",
        ],
        inventory_rows,
    )

    eer_summary = [
        {
            "registered_user_count": len(ENROLLED_USERS),
            "unknown_user": UNKNOWN_USER,
            "training_sample_count": len(training_records),
            "genuine_test_sample_count": len(genuine_records),
            "unknown_test_sample_count": len(unknown_records),
            "excluded_background_sample_count": sum(
                record["normalized_label"] == EXCLUDED_USER
                for record in records
            ),
            "raw_rf_accuracy": raw_accuracy,
            "eer": eer["eer"],
            "eer_threshold": eer_threshold,
            "far_at_eer": far_at_eer,
            "frr_at_eer": frr_at_eer,
            "gar_at_eer": gar_at_eer,
            "genuine_rejection_rate_at_eer": genuine_reject_at_eer,
            "genuine_misidentification_rate_at_eer": misidentification_at_eer,
            "eer_method": eer["method"],
            "feature_dimension": 67,
            "random_seed": RANDOM_SEED,
            "model_fit_count": 1,
            "genuine_prediction_count": len(genuine_records),
            "unknown_prediction_count": len(unknown_records),
        }
    ]
    write_csv(
        OUTPUT_DIR / "eer_summary.csv",
        list(eer_summary[0].keys()),
        eer_summary,
    )

    report = [
        "# Fixed Dataset Random Forest GAR / FAR / FRR / EER Experiment",
        "",
        "## 1. Experiment Objective",
        "",
        "This experiment evaluates one fixed Random Forest model using fixed enrolled-user and unknown-user test datasets.",
        "",
        "## 2. Dataset Information",
        "",
        f"- Dataset CSV total: {len(records)}",
        f"- Valid 50x16 CSV: {sum(record['is_valid'] for record in records)}",
        f"- Enrolled users: {', '.join(ENROLLED_USERS)}",
        f"- Unknown user: {UNKNOWN_USER}",
        f"- Excluded: {EXCLUDED_USER}",
        f"- Exact duplicate file count: 0",
        f"- Exact duplicate pressure-matrix count: 0",
        "",
        "## 3. Dataset Split",
        "",
        f"- Split rule: each enrolled user, 80% training / 20% genuine test",
        f"- Fixed random seed: {RANDOM_SEED}",
        f"- Training samples: {len(training_records)}",
        f"- Genuine test samples: {len(genuine_records)}",
        f"- Unknown test samples: {len(unknown_records)}",
        f"- Excluded background samples: {sum(record['normalized_label'] == EXCLUDED_USER for record in records)}",
        "",
        "## 4. Model Configuration",
        "",
        "- Feature pipeline: `extract_dataset_statistics()` from the main Sequence Statistics benchmark",
        "- Feature dimension: 67",
        "- RandomForestClassifier parameters:",
        f"  - n_estimators: {RF_N_ESTIMATORS}",
        f"  - random_state: {RANDOM_SEED}",
        f"  - class_weight: {RF_CLASS_WEIGHT}",
        f"  - n_jobs: {RF_N_JOBS}",
        f"- Fit count: 1",
        f"- Training sample count: {len(training_records)}",
        "- Maximum class probability is used as confidence.",
        "",
        "## 5. Evaluation Metrics",
        "",
        "- GAR: genuine correct identity accepted / total genuine test samples",
        "- FAR: accepted Joanna samples / 100 Joanna test samples",
        "- FRR: 1 - GAR",
        "- Genuine Rejection Rate: genuine samples with confidence below threshold",
        "- Genuine Misidentification Rate: accepted genuine samples with wrong identity",
        "",
        "## 6. Experimental Results",
        "",
        f"- Raw RF Accuracy: {raw_accuracy:.8f}",
        f"- EER: {eer['eer']}",
        f"- EER Threshold: {eer_threshold:.8f}",
        f"- FAR at EER: {far_at_eer:.8f}",
        f"- FRR at EER: {frr_at_eer:.8f}",
        f"- GAR at EER: {gar_at_eer:.8f}",
        f"- Genuine Rejection Rate at EER: {genuine_reject_at_eer:.8f}",
        f"- Genuine Misidentification Rate at EER: {misidentification_at_eer:.8f}",
        f"- EER calculation method: {eer['method']}",
        "",
        "## 7. Limitations",
        "",
        "- Unknown consists of one user: Joanna.",
        "- Joanna's 100 recordings are 100 samples, not 100 distinct individuals.",
        "- The single fixed result does not estimate variability across random splits.",
        "- Threshold analysis uses fixed test predictions and is not independent calibration.",
        "- The result applies to this fixed dataset and should not be generalized to all strangers.",
        "- No background samples are used.",
        "",
        "## 8. Final Conclusion",
        "",
        "The reported values are based only on the fixed split, one fitted Random Forest, and one prediction pass for each fixed test set.",
    ]
    (OUTPUT_DIR / "experiment_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(
            {
                "enrolled_users": list(ENROLLED_USERS),
                "unknown_user": UNKNOWN_USER,
                "excluded_user": EXCLUDED_USER,
                "rf_parameters": {
                    "n_estimators": RF_N_ESTIMATORS,
                    "random_state": RANDOM_SEED,
                    "class_weight": RF_CLASS_WEIGHT,
                    "n_jobs": RF_N_JOBS,
                },
                "feature_dimension": 67,
                "training_sample_count": len(training_records),
                "genuine_test_sample_count": len(genuine_records),
                "unknown_test_sample_count": len(unknown_records),
                "model_fit_count": 1,
                "genuine_prediction_count": len(genuine_records),
                "unknown_prediction_count": len(unknown_records),
                "raw_rf_accuracy": raw_accuracy,
                "eer": eer,
                "gar_at_eer": gar_at_eer,
                "frr_at_eer": frr_at_eer,
                "genuine_rejection_rate_at_eer": genuine_reject_at_eer,
                "genuine_misidentification_rate_at_eer": misidentification_at_eer,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDataset split:")
    print(f"  Training={len(training_records)}")
    print(f"  Genuine Test={len(genuine_records)}")
    print(f"  Unknown Test={len(unknown_records)}")
    print(f"  Excluded={sum(record['normalized_label'] == EXCLUDED_USER for record in records)}")
    print("\nModel:")
    print(f"  fit count=1")
    print(f"  feature dimension=67")
    print(f"  raw RF accuracy={raw_accuracy:.6f}")
    print("\nFixed-test metrics:")
    print(f"  GAR at EER={gar_at_eer:.6f}")
    print(f"  FAR at EER={far_at_eer:.6f}")
    print(f"  FRR at EER={frr_at_eer:.6f}")
    print(f"  EER={eer['eer']}")
    print(f"  EER threshold={eer_threshold:.6f}")
    print(f"\nResults: {OUTPUT_DIR}")


def main() -> None:
    for repeat in range(1, REPEAT + 1):
        print(f"\nRepeat={repeat:02d}")
        _run_once()


if __name__ == "__main__":
    main()
