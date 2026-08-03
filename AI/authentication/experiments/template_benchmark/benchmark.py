from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from .common import (
    DEFAULT_USERS,
    TemplateBenchmarkRecord,
    TemplateDistributionSet,
    build_mean_template,
    build_median_template,
    build_robust_mean_template,
    build_trimmed_mean_template,
    compute_far_frr_acceptance,
    compute_euclidean_distances,
    compute_roc_metrics,
    format_template_table,
    load_template_benchmark_dataset,
    select_user_samples,
    summarize_template_records,
    build_user_pipeline_vectors,
)

DATA_DIR = Path(__file__).resolve().parents[4] / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent


TEMPLATE_METHODS: tuple[str, ...] = (
    "Mean Template",
    "Median Template",
    "Trimmed Mean Template",
    "Robust Mean",
)


def _build_template(vectors: np.ndarray, template_method: str) -> np.ndarray:
    if template_method == "Mean Template":
        return build_mean_template(vectors)
    if template_method == "Median Template":
        return build_median_template(vectors)
    if template_method == "Trimmed Mean Template":
        return build_trimmed_mean_template(vectors, trim_fraction=0.1)
    if template_method == "Robust Mean":
        return build_robust_mean_template(vectors)
    raise ValueError(f"Unknown template method: {template_method}")


def run_template_benchmark(
    X: np.ndarray | None = None,
    raw_y: np.ndarray | None = None,
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
    k_value: float = 2.0,
) -> tuple[list[TemplateBenchmarkRecord], list[TemplateDistributionSet]]:
    if X is None or raw_y is None:
        X, raw_y = load_template_benchmark_dataset()

    records: list[TemplateBenchmarkRecord] = []
    distributions: list[TemplateDistributionSet] = []

    for template_method in TEMPLATE_METHODS:
        genuine_pool: list[np.ndarray] = []
        impostor_pool: list[np.ndarray] = []

        for user_id in users:
            registration_samples, validation_genuine_samples, impostor_samples = select_user_samples(
                X,
                raw_y,
                user_id,
                registration_count,
            )

            registration_vectors, genuine_vectors, impostor_vectors, _ = build_user_pipeline_vectors(
                registration_samples,
                validation_genuine_samples,
                impostor_samples,
            )

            template_vector = _build_template(registration_vectors, template_method)
            registration_distances = compute_euclidean_distances(registration_vectors, template_vector)
            genuine_distances = compute_euclidean_distances(genuine_vectors, template_vector)
            impostor_distances = compute_euclidean_distances(impostor_vectors, template_vector)

            threshold = float(np.mean(registration_distances) + k_value * np.std(registration_distances))
            far, frr, acceptance_rate = compute_far_frr_acceptance(genuine_distances, impostor_distances, threshold)
            auc, eer, best_threshold, best_far, best_frr, best_acceptance_rate, _, _, _ = compute_roc_metrics(
                genuine_distances,
                impostor_distances,
            )

            records.append(
                TemplateBenchmarkRecord(
                    user_id=user_id,
                    template_method=template_method,
                    threshold=threshold,
                    far=far,
                    frr=frr,
                    acceptance_rate=acceptance_rate,
                    auc=auc,
                    eer=eer,
                    average_genuine_distance=float(np.mean(genuine_distances)),
                    average_impostor_distance=float(np.mean(impostor_distances)),
                )
            )

            genuine_pool.append(genuine_distances)
            impostor_pool.append(impostor_distances)

        distributions.append(
            TemplateDistributionSet(
                template_method=template_method,
                genuine_distances=np.concatenate(genuine_pool),
                impostor_distances=np.concatenate(impostor_pool),
            )
        )

    summary = summarize_template_records(records)
    return summary, distributions


def _plot_distance_distributions(distributions: Sequence[TemplateDistributionSet]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for axis, item in zip(axes, distributions):
        axis.hist(item.genuine_distances, bins=28, alpha=0.68, label="Genuine", color="#2b8a3e", density=True)
        axis.hist(item.impostor_distances, bins=28, alpha=0.68, label="Impostor", color="#c92a2a", density=True)
        axis.set_title(item.template_method)
        axis.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.suptitle("Template Distance Distributions")
    fig.supxlabel("Euclidean Distance")
    fig.supylabel("Density")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUT_DIR / "template_distance_distribution.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_roc_curves(distributions: Sequence[TemplateDistributionSet]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))

    for item in distributions:
        auc, eer, best_threshold, best_far, best_frr, best_acceptance_rate, far_values, tpr_values, _ = compute_roc_metrics(
            item.genuine_distances,
            item.impostor_distances,
        )
        ax.plot(far_values, tpr_values, linewidth=2, label=f"{item.template_method} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("Template ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "template_roc.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    records, distributions = run_template_benchmark()
    roc_path = _plot_roc_curves(distributions)
    dist_path = _plot_distance_distributions(distributions)

    print("\n========== Template Benchmark ==========")
    print(format_template_table(records))
    print(f"\nROC plot: {roc_path}")
    print(f"Distance distribution plot: {dist_path}")


if __name__ == "__main__":
    main()
