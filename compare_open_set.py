"""Compare models on the open-set test (rejecting a person the model never saw).

Reads the scores that evaluate_open_set.py saved for each model and lines the models up fold
by fold. Every model is trained on the same folds (same stranger, held-out day and seed), so
differences are paired: each fold's number for a model minus the same fold's number for the
first model.

Usage:
    python compare_open_set.py
    python compare_open_set.py --runs "Classifier=experiments/open_set" "ArcFace=experiments/open_set_arcface"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_multi_press import fused_metrics, load_folds
from evaluate_open_set import (
    GRID,
    INK,
    INK_SECONDARY,
    SCORER_NAMES,
    SCORERS,
    SURFACE,
    mean_sd,
    pct_sd,
)

DEFAULT_RUNS = [
    "Classifier=experiments/open_set",
    "ArcFace=experiments/open_set_arcface",
    "ArcFace + cross-day=experiments/open_set_arcface_crossday",
]
# Categorical slots 1-3 of the validated default palette, light mode.
MODEL_COLORS = ["#257de0", "#f47230", "#00b894"]
MAIN_SCORER = "centroid"


def parse_runs(items: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for item in items:
        name, separator, folder = item.partition("=")
        if not separator:
            raise SystemExit(f"--runs entries look like Name=folder, got {item!r}")
        runs.append((name, Path(folder)))
    return runs


def describe(folder: Path) -> str:
    meta = json.loads(next(folder.glob("*/metadata.json")).read_text(encoding="utf-8"))
    if (meta.get("loss") or "cross_entropy") == "cross_entropy":
        return f"cross-entropy classifier, lr {meta['learning_rate']:g}, batch {meta['batch_size']}"
    arc = meta["arcface"]
    return (
        f"ArcFace (dim {arc['embedding_dim']}, scale {arc['scale']:g}, margin {arc['margin']:g}), "
        f"cross-day weight {arc['cross_day_weight']:g}, lr {meta['learning_rate']:g}, "
        f"{arc['samples_per_person']} presses per person per batch"
    )


def paired_text(values: list[float], reference: list[float], lower_is_better: bool) -> str:
    """Mean paired difference in points, and in how many folds the model was better."""
    differences = [v - r for v, r in zip(values, reference)]
    mean, sd = mean_sd(differences)
    better = sum((d < 0) if lower_is_better else (d > 0) for d in differences)
    return f"{mean * 100:+.1f} ± {sd * 100:.1f} ({better}/{len(differences)} folds better)"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="Name=folder pairs; the first is the reference for paired differences.")
    parser.add_argument("--presses", nargs="+", type=int, default=[1, 5],
                        help="Presses per attempt to report (see evaluate_multi_press.py).")
    parser.add_argument("--output", default="experiments/open_set_comparison",
                        help="Writes <output>.md and <output>.png.")
    args = parser.parse_args(argv)
    runs = parse_runs(args.runs)
    if len(runs) > len(MODEL_COLORS):
        raise SystemExit(f"At most {len(MODEL_COLORS)} runs can be compared in one chart.")

    loaded = {name: dict(load_folds(folder)) for name, folder in runs}
    common = sorted(set.intersection(*(set(folds) for folds in loaded.values())))
    if not common:
        raise SystemExit("The runs share no folds.")
    days = sorted({fold.split("_day")[1][:8] for fold in common})
    # metrics[name][presses][fold] -> fused_metrics result
    metrics = {
        name: {n: {fold: fused_metrics(loaded[name][fold], n) for fold in common} for n in args.presses}
        for name, _ in runs
    }

    def values(name: str, n: int, scorer: str, key: str, folds: list[str] | None = None) -> list[float]:
        return [metrics[name][n][fold][scorer][key] for fold in (folds or common)]

    reference = runs[0][0]
    lines = [
        "# Open-set comparison: rejecting a person the model never saw",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `compare_open_set.py` from {len(common)} folds that "
        "every model was trained on (each fold: one person left out as the stranger, one day held out; "
        "see each folder's `summary.md`). Thresholds are picked on each fold's own test scores. "
        "Numbers are mean ± standard deviation over folds.",
        "",
        "Models:",
        "",
    ]
    lines += [f"- **{name}** (`{folder.as_posix()}`): {describe(folder)}" for name, folder in runs]

    for n in args.presses:
        attempt = "1 press" if n == 1 else f"{n} presses in a row, scores averaged"
        lines += [
            "",
            f"## Decision from {attempt}",
            "",
            "| Model | Score | Closed-set accuracy | EER (known vs stranger) | Accepted & named right at 1% strangers in "
            "| Accepted & named right at 5% strangers in | Verification EER |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, _ in runs:
            for scorer in SCORERS:
                lines.append(
                    f"| {name} | {SCORER_NAMES[scorer]} | {pct_sd(values(name, n, scorer, 'accuracy'))} | "
                    f"{pct_sd(values(name, n, scorer, 'eer'))} | {pct_sd(values(name, n, scorer, 'dir@0.01'))} | "
                    f"{pct_sd(values(name, n, scorer, 'dir@0.05'))} | {pct_sd(values(name, n, scorer, 'verify_eer'))} |"
                )

    if len(runs) > 1:
        lines += [
            "",
            f"## Paired difference from {reference} ({SCORER_NAMES[MAIN_SCORER].lower()})",
            "",
            "Points, model minus the reference on the same fold. EER: negative is better. "
            "Accepted & named right: positive is better.",
            "",
            "| Model | Presses | EER | Accepted & named right at 1% strangers in | Closed-set accuracy |",
            "|---|---|---|---|---|",
        ]
        for name, _ in runs[1:]:
            for n in args.presses:
                lines.append(
                    f"| {name} | {n} | "
                    f"{paired_text(values(name, n, MAIN_SCORER, 'eer'), values(reference, n, MAIN_SCORER, 'eer'), True)} | "
                    f"{paired_text(values(name, n, MAIN_SCORER, 'dir@0.01'), values(reference, n, MAIN_SCORER, 'dir@0.01'), False)} | "
                    f"{paired_text(values(name, n, MAIN_SCORER, 'accuracy'), values(reference, n, MAIN_SCORER, 'accuracy'), False)} |"
                )

    lines += [
        "",
        f"## How far apart are known and stranger scores? ({SCORER_NAMES[MAIN_SCORER].lower()}, 1 press)",
        "",
        "Each recording's best cosine similarity to an enrolled person, averaged per fold. d′ is the gap "
        "between the known and stranger means in units of their pooled standard deviation (higher = easier "
        "to separate). If both means rise together, the model is pulling everything, strangers included, "
        "onto the enrolled people.",
        "",
        "| Model | Known test: best similarity | Stranger: best similarity | d′ |",
        "|---|---|---|---|",
    ]
    for name, _ in runs:
        known_means, stranger_means, d_primes = [], [], []
        for fold in common:
            known = loaded[name][fold][f"{MAIN_SCORER}_known"].max(axis=1)
            stranger = loaded[name][fold][f"{MAIN_SCORER}_stranger"].max(axis=1)
            known_means.append(float(known.mean()))
            stranger_means.append(float(stranger.mean()))
            d_primes.append(float((known.mean() - stranger.mean()) / np.sqrt((known.var() + stranger.var()) / 2)))
        lines.append(
            f"| {name} | {np.mean(known_means):.3f} | {np.mean(stranger_means):.3f} | "
            f"{np.mean(d_primes):.2f} ± {np.std(d_primes, ddof=1):.2f} |"
        )

    lines += [
        "",
        f"## By held-out day ({SCORER_NAMES[MAIN_SCORER].lower()})",
        "",
        "Closed-set accuracy and EER on each held-out day, mean over the strangers.",
        "",
        "| Held-out day | " + " | ".join(
            f"{name}: accuracy | {name}: EER, {n} press{'es' if n > 1 else ''}"
            for name, _ in runs for n in args.presses[:1]
        ) + " | " + " | ".join(
            f"{name}: EER, {n} presses" for name, _ in runs for n in args.presses[1:]
        ) + " |",
        "|---|" + "---|" * (2 * len(runs) + len(runs) * len(args.presses[1:])),
    ]
    for day in days:
        folds = [fold for fold in common if f"_day{day}_" in fold]
        first = args.presses[0]
        cells = []
        for name, _ in runs:
            cells += [pct_sd(values(name, first, MAIN_SCORER, "accuracy", folds)),
                      pct_sd(values(name, first, MAIN_SCORER, "eer", folds))]
        cells += [pct_sd(values(name, n, MAIN_SCORER, "eer", folds)) for name, _ in runs for n in args.presses[1:]]
        lines.append(f"| {day[:4]}-{day[4:6]}-{day[6:]} | " + " | ".join(cells) + " |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines += ["", f"Chart: `{output.with_suffix('.png').name}`."]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(runs, metrics, common, days, args.presses, output.with_suffix(".png"))
    print(f"Wrote {output.with_suffix('.md')} and {output.with_suffix('.png')}")


def plot(runs, metrics, common, days, presses_list, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [presses_list[0], presses_list[-1]] if len(presses_list) > 1 else presses_list
    fig, axes = plt.subplots(1, len(shown), figsize=(5.6 * len(shown), 4.4), facecolor=SURFACE, squeeze=False)
    categories = [f"{day[4:6]}-{day[6:]}" for day in days] + ["All days"]
    offsets = {name: (index - (len(runs) - 1) / 2) * 0.22 for index, (name, _) in enumerate(runs)}
    for ax, n in zip(axes[0], shown):
        for (name, _), color in zip(runs, MODEL_COLORS):
            means, sds = [], []
            for day in days:
                chosen = [metrics[name][n][fold][MAIN_SCORER]["eer"] for fold in common if f"_day{day}_" in fold]
                mean, sd = mean_sd(chosen)
                means.append(mean * 100)
                sds.append(sd * 100)
            mean, sd = mean_sd([metrics[name][n][fold][MAIN_SCORER]["eer"] for fold in common])
            means.append(mean * 100)
            sds.append(sd * 100)
            xs = [index + offsets[name] for index in range(len(categories))]
            ax.errorbar(xs, means, yerr=sds, fmt="o", markersize=8, color=color, ecolor=color, elinewidth=2,
                        capsize=0, markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
        # Exact values are in the .md tables; labels here would collide with the error bars.
        ax.set_xticks(range(len(categories)), categories)
        ax.set_ylim(0, 70)
        ax.set_xlabel("Held-out day", color=INK_SECONDARY)
        ax.set_ylabel("EER, known vs stranger (%)", color=INK_SECONDARY)
        attempt = "1 press" if n == 1 else f"{n} presses averaged"
        ax.set_title(f"Equal error rate, {attempt} (lower is better)", color=INK, loc="left", fontsize=10)
        ax.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=8, ncols=len(runs))
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
