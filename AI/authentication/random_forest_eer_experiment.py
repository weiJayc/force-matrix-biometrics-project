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
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "eer_experiment"

from data_loader import load_dataset
from sequence_statistics_contact_gate_benchmark_live import (
    RANDOM_SEED,
    REGISTRATION_COUNTS,
    REPEATS,
    TEST_COUNT,
    USERS,
    create_user_split,
    extract_dataset_statistics,
)

RF_N_ESTIMATORS = 100
RF_CLASS_WEIGHT = None
RF_N_JOBS = -1
THRESHOLDS = np.linspace(0.0, 1.0, 1001)
UNKNOWN = "Unknown"


def load_data() -> tuple[np.ndarray, np.ndarray]:
    legacy_csvs = [
        path
        for path in DATA_DIR.rglob("*.csv")
        if any(part.lower() in {"legacy", "legency"} for part in path.parts)
    ]
    if legacy_csvs:
        raise ValueError(
            "Refusing to read legacy or legency data: "
            + ", ".join(str(path) for path in legacy_csvs)
        )

    X, labels = load_dataset(DATA_DIR)
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    if X.ndim != 3 or X.shape[1:] != (50, 16):
        raise ValueError(f"Expected X with shape (N, 50, 16), got {X.shape}")

    required = REGISTRATION_COUNTS[0] + TEST_COUNT
    for user in USERS:
        count = int(np.sum(labels == user))
        if count < required:
            raise ValueError(
                f"{user} needs {required} samples, but only {count} are available"
            )
    return X, labels


def fit_predict(
    X_features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    genuine_test_indices: np.ndarray,
    stranger_test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    overlap = np.intersect1d(train_indices, genuine_test_indices)
    if overlap.size:
        raise AssertionError(
            f"Genuine training/test overlap: {overlap.tolist()}"
        )
    overlap = np.intersect1d(train_indices, stranger_test_indices)
    if overlap.size:
        raise AssertionError(
            f"Stranger training/test overlap: {overlap.tolist()}"
        )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_SEED,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=RF_N_JOBS,
    )
    model.fit(X_features[train_indices], labels[train_indices])

    def predict(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = model.predict_proba(X_features[indices])
        probability_indices = np.argmax(probabilities, axis=1)
        predictions = model.classes_[probability_indices]
        max_probabilities = probabilities[
            np.arange(len(probabilities)), probability_indices
        ]
        return np.asarray(predictions), max_probabilities

    genuine_predictions, genuine_probabilities = predict(genuine_test_indices)
    stranger_predictions, stranger_probabilities = predict(stranger_test_indices)
    return (
        genuine_predictions,
        genuine_probabilities,
        stranger_predictions,
        stranger_probabilities,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_eer(
    far_values: np.ndarray,
    frr_values: np.ndarray,
) -> dict:
    difference = far_values - frr_values
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-12))
    if exact.size:
        index = int(exact[np.argmin(np.abs(difference[exact]))])
        far = float(far_values[index])
        frr = float(frr_values[index])
        return {
            "eer": float((far + frr) / 2.0),
            "eer_threshold": float(THRESHOLDS[index]),
            "far_at_eer": far,
            "frr_at_eer": frr,
            "method": "exact",
        }

    for index in range(len(THRESHOLDS) - 1):
        left = difference[index]
        right = difference[index + 1]
        if left * right < 0:
            x0 = float(THRESHOLDS[index])
            x1 = float(THRESHOLDS[index + 1])
            d0 = float(difference[index])
            d1 = float(difference[index + 1])
            threshold = x0 + (0.0 - d0) * (x1 - x0) / (d1 - d0)
            far = float(
                far_values[index]
                + (threshold - x0)
                * (far_values[index + 1] - far_values[index])
                / (x1 - x0)
            )
            frr = float(
                frr_values[index]
                + (threshold - x0)
                * (frr_values[index + 1] - frr_values[index])
                / (x1 - x0)
            )
            return {
                "eer": float((far + frr) / 2.0),
                "eer_threshold": threshold,
                "far_at_eer": far,
                "frr_at_eer": frr,
                "method": "linear_interpolation",
            }

    index = int(np.argmin(np.abs(difference)))
    far = float(far_values[index])
    frr = float(frr_values[index])
    return {
        "eer": None,
        "eer_threshold": float(THRESHOLDS[index]),
        "far_at_eer": far,
        "frr_at_eer": frr,
        "method": "no_crossing_nearest",
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan")
    return float(np.mean(array)), float(np.std(array))


def run_original_experiment(
    X_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, np.ndarray, list[float]]:
    repeat_rows: list[dict] = []
    sample_rows: list[dict] = []
    gar_curves: list[np.ndarray] = []
    mis_curves: list[np.ndarray] = []
    raw_accuracies: list[float] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for repeat in range(1, REPEATS + 1):
        registration_by_user: dict[str, np.ndarray] = {}
        test_by_user: dict[str, np.ndarray] = {}

        for user in USERS:
            registration, test = create_user_split(
                np.where(labels == user)[0],
                REGISTRATION_COUNTS[0],
                rng,
            )
            if np.intersect1d(registration, test).size:
                raise AssertionError(f"Original split overlap for {user}")
            registration_by_user[user] = registration
            test_by_user[user] = test

        train_indices = np.concatenate(
            [registration_by_user[user] for user in USERS]
        )
        test_indices = np.concatenate([test_by_user[user] for user in USERS])
        if np.intersect1d(train_indices, test_indices).size:
            raise AssertionError("Original 8-class train/test overlap")

        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_SEED,
            class_weight=RF_CLASS_WEIGHT,
            n_jobs=RF_N_JOBS,
        )
        model.fit(X_features[train_indices], labels[train_indices])
        probabilities = model.predict_proba(X_features[test_indices])
        probability_indices = np.argmax(probabilities, axis=1)
        predictions = model.classes_[probability_indices]
        max_probabilities = probabilities[
            np.arange(len(probabilities)), probability_indices
        ]
        true_labels = labels[test_indices]

        raw_accuracy = float(np.mean(true_labels == predictions))
        raw_accuracies.append(raw_accuracy)
        repeat_rows.append(
            {
                "repeat": repeat,
                "raw_accuracy": raw_accuracy,
                "train_samples": int(len(train_indices)),
                "test_samples": int(len(test_indices)),
            }
        )

        gar_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
        mis_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
        for index, threshold in enumerate(THRESHOLDS):
            passed = max_probabilities >= threshold
            gar_curve[index] = np.sum(
                passed & (true_labels == predictions)
            ) / len(test_indices)
            mis_curve[index] = np.sum(
                passed & (true_labels != predictions)
            ) / len(test_indices)
        gar_curves.append(gar_curve)
        mis_curves.append(mis_curve)

        for source_index, true_label, prediction, probability in zip(
            test_indices, true_labels, predictions, max_probabilities
        ):
            sample_rows.append(
                {
                    "experiment": "original_8_class",
                    "repeat": repeat,
                    "excluded_subject": "",
                    "source_index": int(source_index),
                    "actual_label": str(true_label),
                    "predicted_label": str(prediction),
                    "max_probability": float(probability),
                }
            )

    return (
        repeat_rows,
        sample_rows,
        np.asarray(gar_curves),
        np.asarray(mis_curves),
        np.asarray([1.0 - curve for curve in gar_curves]),
        raw_accuracies,
    )


def run_loso_experiment(
    X_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    model_rows: list[dict] = []
    held_out_rows: list[dict] = []
    sample_rows: list[dict] = []
    curve_records: list[dict] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for repeat in range(1, REPEATS + 1):
        for held_out_user in USERS:
            known_users = [user for user in USERS if user != held_out_user]
            train_parts: list[np.ndarray] = []
            genuine_test_parts: list[np.ndarray] = []

            for known_user in known_users:
                registration, genuine_test = create_user_split(
                    np.where(labels == known_user)[0],
                    REGISTRATION_COUNTS[0],
                    rng,
                )
                train_parts.append(registration)
                genuine_test_parts.append(genuine_test)

            train_indices = np.concatenate(train_parts)
            genuine_test_indices = np.concatenate(genuine_test_parts)

            if held_out_user in labels[train_indices]:
                raise AssertionError(
                    f"Held-out user {held_out_user} entered LOSO training"
                )
            if np.intersect1d(train_indices, genuine_test_indices).size:
                raise AssertionError(
                    f"Genuine training/test overlap for {held_out_user}"
                )

            stranger_indices = np.where(labels == held_out_user)[0]
            stranger_test_indices = rng.permutation(stranger_indices)[:TEST_COUNT]
            if np.intersect1d(train_indices, stranger_test_indices).size:
                raise AssertionError(
                    f"Stranger training/test overlap for {held_out_user}"
                )

            genuine_true = labels[genuine_test_indices]
            (
                genuine_predictions,
                genuine_probabilities,
                stranger_predictions,
                stranger_probabilities,
            ) = fit_predict(
                X_features,
                labels,
                train_indices,
                genuine_test_indices,
                stranger_test_indices,
            )

            gar_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
            far_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
            mis_curve = np.empty(len(THRESHOLDS), dtype=np.float64)
            genuine_reject_curve = np.empty(len(THRESHOLDS), dtype=np.float64)

            for index, threshold in enumerate(THRESHOLDS):
                genuine_passed = genuine_probabilities >= threshold
                genuine_correct = genuine_passed & (
                    genuine_true == genuine_predictions
                )
                genuine_misidentified = genuine_passed & (
                    genuine_true != genuine_predictions
                )
                genuine_rejected = ~genuine_passed

                gar_curve[index] = np.sum(genuine_correct) / len(genuine_true)
                mis_curve[index] = (
                    np.sum(genuine_misidentified) / len(genuine_true)
                )
                genuine_reject_curve[index] = (
                    np.sum(genuine_rejected) / len(genuine_true)
                )
                far_curve[index] = (
                    np.sum(stranger_probabilities >= threshold)
                    / len(stranger_test_indices)
                )

            frr_curve = 1.0 - gar_curve
            eer = find_eer(far_curve, frr_curve)

            model_rows.append(
                {
                    "repeat": repeat,
                    "held_out_user": held_out_user,
                    "eer": eer["eer"],
                    "eer_threshold": eer["eer_threshold"],
                    "far_at_eer": eer["far_at_eer"],
                    "frr_at_eer": eer["frr_at_eer"],
                    "method": eer["method"],
                    "train_samples": int(len(train_indices)),
                    "genuine_test_samples": int(len(genuine_test_indices)),
                    "stranger_test_samples": int(len(stranger_test_indices)),
                }
            )
            held_out_rows.append(
                {
                    "repeat": repeat,
                    "held_out_user": held_out_user,
                    "train_samples": int(len(train_indices)),
                    "genuine_test_samples": int(len(genuine_test_indices)),
                    "stranger_test_samples": int(len(stranger_test_indices)),
                }
            )
            curve_records.append(
                {
                    "repeat": repeat,
                    "held_out_user": held_out_user,
                    "gar": gar_curve,
                    "far": far_curve,
                    "frr": frr_curve,
                    "mis": mis_curve,
                    "genuine_reject": genuine_reject_curve,
                }
            )

            for source_index, prediction, probability in zip(
                genuine_test_indices,
                genuine_predictions,
                genuine_probabilities,
            ):
                sample_rows.append(
                    {
                        "experiment": "LOSO_genuine",
                        "repeat": repeat,
                        "excluded_subject": held_out_user,
                        "source_index": int(source_index),
                        "actual_label": str(labels[source_index]),
                        "predicted_label": str(prediction),
                        "max_probability": float(probability),
                    }
                )
            for source_index, prediction, probability in zip(
                stranger_test_indices,
                stranger_predictions,
                stranger_probabilities,
            ):
                sample_rows.append(
                    {
                        "experiment": "LOSO_simulated_stranger",
                        "repeat": repeat,
                        "excluded_subject": held_out_user,
                        "source_index": int(source_index),
                        "actual_label": UNKNOWN,
                        "predicted_label": str(prediction),
                        "max_probability": float(probability),
                    }
                )

    return model_rows, held_out_rows, sample_rows, curve_records


def save_plots(
    mean_far: np.ndarray,
    mean_frr: np.ndarray,
    aggregated_eer: dict,
    model_rows: list[dict],
) -> None:
    mean_curve_eer = find_eer(mean_far, mean_frr)
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        THRESHOLDS * 100,
        mean_far * 100,
        label="Mean LOSO FAR",
        color="tab:red",
    )
    axis.plot(
        THRESHOLDS * 100,
        mean_frr * 100,
        label="Mean LOSO FRR",
        color="tab:blue",
    )
    if mean_curve_eer["eer"] is not None:
        threshold = mean_curve_eer["eer_threshold"] * 100
        axis.axvline(
            threshold,
            linestyle="--",
            color="tab:green",
            label=f"Mean-Curve EER Threshold {threshold:.2f}% ",
        )
        axis.scatter(
            [threshold],
            [mean_curve_eer["eer"] * 100],
            color="tab:green",
            zorder=5,
        )
        axis.annotate(
            f"Mean-Curve EER {mean_curve_eer['eer'] * 100:.2f}%\n"
            f"Threshold {threshold:.2f}%\n"
            f"FAR {mean_curve_eer['far_at_eer'] * 100:.2f}%\n"
            f"FRR {mean_curve_eer['frr_at_eer'] * 100:.2f}%",
            xy=(threshold, mean_curve_eer["eer"] * 100),
            xytext=(12, 30),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->"},
        )
    axis.set_title("LOSO Mean FAR / FRR Threshold Curve")
    axis.set_xlabel("Confidence Threshold (%)")
    axis.set_ylabel("Rate (%)")
    axis.set_ylim(0, 100)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_mean_far_frr_curve.png", dpi=180)
    plt.close(fig)

    eer_values = [
        float(row["eer"]) for row in model_rows if row["eer"] is not None
    ]
    threshold_values = [
        float(row["eer_threshold"])
        for row in model_rows
        if row["eer"] is not None
    ]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(np.asarray(eer_values) * 100, bins=20, edgecolor="black")
    axis.set_title("LOSO EER Distribution")
    axis.set_xlabel("EER (%)")
    axis.set_ylabel("Model Count")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_eer_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(np.asarray(threshold_values) * 100, bins=20, edgecolor="black")
    axis.set_title("LOSO EER Threshold Distribution")
    axis.set_xlabel("EER Threshold (%)")
    axis.set_ylabel("Model Count")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_eer_threshold_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    print("=" * 110)
    print("RANDOM FOREST LOSO EER EXPERIMENT")
    print("=" * 110)

    X, labels = load_data()
    X_features = extract_dataset_statistics(X)
    if X_features.shape[1] != 67:
        raise ValueError(f"Expected 67 features, got {X_features.shape}")
    if np.isnan(X_features).any():
        raise ValueError("NaN detected in Sequence Statistics features")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {DATA_DIR}")
    print(f"Sequence feature shape: {X_features.shape}")
    print("Threshold scan: 0.000 to 1.000, step 0.001")

    (
        original_repeat_rows,
        original_samples,
        original_gar_matrix,
        original_mis_matrix,
        original_frr_matrix,
        original_raw_accuracies,
    ) = run_original_experiment(X_features, labels)

    model_rows, held_out_rows, loso_samples, curve_records = (
        run_loso_experiment(X_features, labels)
    )

    write_csv(
        OUTPUT_DIR / "original_8_user_repeat_results.csv",
        ["repeat", "raw_accuracy", "train_samples", "test_samples"],
        original_repeat_rows,
    )
    write_csv(
        OUTPUT_DIR / "original_8_user_threshold_curve.csv",
        ["threshold", "GAR", "FRR", "Genuine_Misidentification_Rate"],
        [
            {
                "threshold": float(threshold),
                "GAR": float(np.mean(original_gar_matrix[:, index])),
                "FRR": float(np.mean(original_frr_matrix[:, index])),
                "Genuine_Misidentification_Rate": float(
                    np.mean(original_mis_matrix[:, index])
                ),
            }
            for index, threshold in enumerate(THRESHOLDS)
        ],
    )
    write_csv(
        OUTPUT_DIR / "original_8_user_per_sample_predictions.csv",
        [
            "experiment",
            "repeat",
            "excluded_subject",
            "source_index",
            "actual_label",
            "predicted_label",
            "max_probability",
        ],
        original_samples,
    )

    mean_far = np.mean(
        np.asarray([record["far"] for record in curve_records]),
        axis=0,
    )
    mean_frr = np.mean(
        np.asarray([record["frr"] for record in curve_records]),
        axis=0,
    )
    mean_gar = np.mean(
        np.asarray([record["gar"] for record in curve_records]),
        axis=0,
    )
    mean_mis = np.mean(
        np.asarray([record["mis"] for record in curve_records]),
        axis=0,
    )
    mean_genuine_reject = np.mean(
        np.asarray([record["genuine_reject"] for record in curve_records]),
        axis=0,
    )

    far_correct_sum = np.zeros(len(THRESHOLDS), dtype=np.int64)
    stranger_total = 0
    gar_correct_sum = np.zeros(len(THRESHOLDS), dtype=np.int64)
    genuine_total = 0
    for record in curve_records:
        stranger_total += TEST_COUNT
        genuine_total += 140

    for index in range(len(THRESHOLDS)):
        for record in curve_records:
            gar_correct_sum[index] += int(
                np.rint(record["gar"][index] * 140)
            )
            far_correct_sum[index] += int(
                np.rint(record["far"][index] * TEST_COUNT)
            )

    pooled_far = far_correct_sum / max(stranger_total, 1)
    pooled_gar = gar_correct_sum / max(genuine_total, 1)
    pooled_frr = 1.0 - pooled_gar
    aggregated_eer = find_eer(pooled_far, pooled_frr)

    write_csv(
        OUTPUT_DIR / "loso_mean_far_frr_gar_curve.csv",
        [
            "threshold",
            "mean_FAR",
            "mean_FRR",
            "mean_GAR",
            "mean_Genuine_Misidentification_Rate",
            "mean_Genuine_Rejection_Rate",
        ],
        [
            {
                "threshold": float(threshold),
                "mean_FAR": float(mean_far[index]),
                "mean_FRR": float(mean_frr[index]),
                "mean_GAR": float(mean_gar[index]),
                "mean_Genuine_Misidentification_Rate": float(mean_mis[index]),
                "mean_Genuine_Rejection_Rate": float(
                    mean_genuine_reject[index]
                ),
            }
            for index, threshold in enumerate(THRESHOLDS)
        ],
    )
    write_csv(
        OUTPUT_DIR / "loso_aggregated_far_frr_gar_curve.csv",
        ["threshold", "pooled_FAR", "pooled_FRR", "pooled_GAR"],
        [
            {
                "threshold": float(threshold),
                "pooled_FAR": float(pooled_far[index]),
                "pooled_FRR": float(pooled_frr[index]),
                "pooled_GAR": float(pooled_gar[index]),
            }
            for index, threshold in enumerate(THRESHOLDS)
        ],
    )
    write_csv(
        OUTPUT_DIR / "loso_eer_per_model.csv",
        [
            "repeat",
            "held_out_user",
            "eer",
            "eer_threshold",
            "far_at_eer",
            "frr_at_eer",
            "method",
            "train_samples",
            "genuine_test_samples",
            "stranger_test_samples",
        ],
        model_rows,
    )

    repeat_rows: list[dict] = []
    held_out_rows_summary: list[dict] = []
    for repeat in range(1, REPEATS + 1):
        selected = [row for row in model_rows if row["repeat"] == repeat]
        eer_mean, eer_std = mean_std(
            [row["eer"] for row in selected if row["eer"] is not None]
        )
        threshold_mean, threshold_std = mean_std(
            [
                row["eer_threshold"]
                for row in selected
                if row["eer"] is not None
            ]
        )
        repeat_rows.append(
            {
                "repeat": repeat,
                "mean_eer": eer_mean,
                "eer_std_within_repeat": eer_std,
                "mean_eer_threshold": threshold_mean,
                "eer_threshold_std_within_repeat": threshold_std,
            }
        )

    write_csv(
        OUTPUT_DIR / "loso_eer_per_repeat.csv",
        [
            "repeat",
            "mean_eer",
            "eer_std_within_repeat",
            "mean_eer_threshold",
            "eer_threshold_std_within_repeat",
        ],
        repeat_rows,
    )

    for held_out_user in USERS:
        selected = [
            row for row in model_rows if row["held_out_user"] == held_out_user
        ]
        eer_mean, eer_std = mean_std(
            [row["eer"] for row in selected if row["eer"] is not None]
        )
        threshold_mean, threshold_std = mean_std(
            [
                row["eer_threshold"]
                for row in selected
                if row["eer"] is not None
            ]
        )
        held_out_rows_summary.append(
            {
                "held_out_user": held_out_user,
                "eer_mean": eer_mean,
                "eer_std": eer_std,
                "eer_threshold_mean": threshold_mean,
                "eer_threshold_std": threshold_std,
            }
        )
    write_csv(
        OUTPUT_DIR / "loso_eer_held_out_summary.csv",
        [
            "held_out_user",
            "eer_mean",
            "eer_std",
            "eer_threshold_mean",
            "eer_threshold_std",
        ],
        held_out_rows_summary,
    )

    write_csv(
        OUTPUT_DIR / "loso_held_out_test_splits.csv",
        [
            "repeat",
            "held_out_user",
            "train_samples",
            "genuine_test_samples",
            "stranger_test_samples",
        ],
        held_out_rows,
    )
    write_csv(
        OUTPUT_DIR / "loso_per_sample_predictions.csv",
        [
            "experiment",
            "repeat",
            "excluded_subject",
            "source_index",
            "actual_label",
            "predicted_label",
            "max_probability",
        ],
        loso_samples,
    )

    complete_eers = [
        row["eer"] for row in model_rows if row["eer"] is not None
    ]
    complete_thresholds = [
        row["eer_threshold"] for row in model_rows if row["eer"] is not None
    ]
    mean_eer, std_eer = mean_std(complete_eers)
    mean_threshold, std_threshold = mean_std(complete_thresholds)
    raw_accuracy_mean, raw_accuracy_std = mean_std(
        original_raw_accuracies
    )

    summary_rows = [
        {
            "metric": "mean_eer_across_individual_loso_models",
            "value": mean_eer,
            "std": std_eer,
        },
        {
            "metric": "mean_eer_threshold_across_individual_loso_models",
            "value": mean_threshold,
            "std": std_threshold,
        },
        {
            "metric": "aggregated_curve_eer",
            "value": aggregated_eer["eer"],
            "std": None,
        },
        {
            "metric": "aggregated_curve_eer_threshold",
            "value": aggregated_eer["eer_threshold"],
            "std": None,
        },
        {
            "metric": "raw_accuracy_original_8_user",
            "value": raw_accuracy_mean,
            "std": raw_accuracy_std,
        },
    ]
    write_csv(
        OUTPUT_DIR / "loso_eer_summary.csv",
        ["metric", "value", "std"],
        summary_rows,
    )

    save_plots(mean_far, mean_frr, aggregated_eer, model_rows)

    report = [
        "Random Forest LOSO EER Experiment",
        "",
        f"Sequence features: {X_features.shape[1]}",
        f"RF parameters: n_estimators={RF_N_ESTIMATORS}, random_state={RANDOM_SEED}, "
        f"class_weight={RF_CLASS_WEIGHT}, n_jobs={RF_N_JOBS}",
        f"Registration: {REGISTRATION_COUNTS[0]}, test: {TEST_COUNT}, repeats: {REPEATS}",
        "Threshold scan: 0.000 to 1.000, step 0.001",
        "",
        "Original 8-user experiment:",
        f"Raw RF accuracy mean: {raw_accuracy_mean:.8f}",
        f"Raw RF accuracy std: {raw_accuracy_std:.8f}",
        "This experiment has no EER.",
        "Its GAR, FRR=1-GAR, and Genuine Misidentification curves are independent.",
        "",
        "LOSO computation:",
        "For each repeat and held-out user, 7 users provide 80 registration "
        "samples and 20 genuine test samples each.",
        "The held-out user provides 20 stranger test samples and no training samples.",
        "FAR and FRR are calculated from the same fitted model.",
        "",
        f"Mean EER across 160 held-out/repeat models: {mean_eer:.8f}",
        f"EER std: {std_eer:.8f}",
        f"Mean EER threshold: {mean_threshold:.8f}",
        f"EER threshold std: {std_threshold:.8f}",
        f"Aggregated curve EER: {aggregated_eer['eer']}",
        f"Aggregated curve threshold: {aggregated_eer['eer_threshold']}",
        f"Aggregated curve FAR: {aggregated_eer['far_at_eer']}",
        f"Aggregated curve FRR: {aggregated_eer['frr_at_eer']}",
        f"Aggregated curve method: {aggregated_eer['method']}",
        "",
        "FAR = held-out stranger data accepted as any registered identity.",
        "GAR = genuine data correctly identified and accepted.",
        "FRR = 1 - GAR.",
        "Genuine misidentification is retained separately.",
        "This is simulated LOSO FAR, not measured real-stranger FAR.",
    ]
    (OUTPUT_DIR / "eer_experiment_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "eer_experiment_summary.json").write_text(
        json.dumps(
            {
                "original_8_user": {
                    "raw_accuracy_mean": raw_accuracy_mean,
                    "raw_accuracy_std": raw_accuracy_std,
                    "eer_calculated": False,
                },
                "loso": {
                    "mean_eer": mean_eer,
                    "std_eer": std_eer,
                    "mean_eer_threshold": mean_threshold,
                    "std_eer_threshold": std_threshold,
                    "aggregated_curve": aggregated_eer,
                    "model_count": len(model_rows),
                    "repeat_count": REPEATS,
                    "held_out_count": len(USERS),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nOriginal 8-user:")
    print(
        f"  Raw accuracy mean={raw_accuracy_mean:.6f}, "
        f"std={raw_accuracy_std:.6f}"
    )
    print("  EER is not calculated.")
    print("\nLOSO:")
    print(f"  Individual-model mean EER={mean_eer:.6f}, std={std_eer:.6f}")
    print(
        f"  Individual-model mean threshold={mean_threshold:.6f}, "
        f"std={std_threshold:.6f}"
    )
    print(
        f"  Aggregated curve EER={aggregated_eer['eer']:.6f}, "
        f"threshold={aggregated_eer['eer_threshold']:.6f}"
    )
    print(f"\nResults: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
