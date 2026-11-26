"""Does deciding from several presses reject strangers better than deciding from one?

Reads the scores that evaluate_open_set.py saved for each fold and fuses n consecutive
presses into one attempt: the per-person score vectors of the n presses are averaged, then
the attempt is accepted or rejected exactly as a single press would be.

An attempt is n presses in a row by the same person on the same day, in recording order
(the time at the start of each file name; presses in a session are about 4 s apart).
Consecutive presses resemble each other more than presses picked at random, so this does
not overstate the gain. Every run of n consecutive presses counts as an attempt (a sliding
window), so attempts overlap and are not independent; they only smooth the estimate.

Usage:
    python evaluate_multi_press.py                       # n = 1 2 3 5
    python evaluate_multi_press.py --presses 1 3 5 10
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_open_set import (
    FAR_TARGETS,
    GRID,
    INK,
    INK_SECONDARY,
    SCORER_COLORS,
    SCORER_NAMES,
    SCORERS,
    SHORT_NAMES,
    SURFACE,
    auroc,
    dir_at_far,
    eer,
    mean_sd,
    pct_sd,
)


def attempt_windows(groups: np.ndarray, files: np.ndarray, presses: int) -> np.ndarray:
    """Index windows of `presses` consecutive recordings within each group, in file-name order."""
    windows = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        ordered = members[np.argsort(files[members])]
        windows.extend(ordered[start : start + presses] for start in range(len(ordered) - presses + 1))
    return np.array(windows, dtype=int).reshape(-1, presses)


def fused_metrics(data: Any, presses: int) -> dict[str, dict[str, float]]:
    known_y = data["known_y"]
    known_windows = attempt_windows(known_y, data["known_file"], presses)
    stranger_windows = attempt_windows(data["stranger_day"], data["stranger_file"], presses)
    attempt_y = known_y[known_windows[:, 0]]
    result: dict[str, dict[str, float]] = {}
    for scorer in SCORERS:
        known = data[f"{scorer}_known"][known_windows].mean(axis=1)
        stranger = data[f"{scorer}_stranger"][stranger_windows].mean(axis=1)
        top_known = known.max(axis=1)
        top_stranger = stranger.max(axis=1)
        correct = known.argmax(axis=1) == attempt_y
        result[scorer] = {
            "accuracy": float(correct.mean()),
            "auroc": auroc(top_known, top_stranger),
            "eer": eer(top_known, top_stranger),
            **{f"dir@{far:g}": dir_at_far(top_known, correct, top_stranger, far) for far in FAR_TARGETS},
            "verify_eer": eer(known[np.arange(len(attempt_y)), attempt_y], stranger.ravel()),
            "known_attempts": len(known_windows),
            "stranger_attempts": len(stranger_windows),
        }
    return result


def load_folds(results_dir: Path) -> list[tuple[str, Any]]:
    folds = []
    for path in sorted(results_dir.glob("*/scores.npz")):
        with np.load(path) as data:
            if "stranger_file" not in data.files:
                raise SystemExit(f"{path} has no file names; run evaluate_open_set.py again to re-score it.")
            folds.append((path.parent.name, {key: data[key] for key in data.files}))
    return folds


def summarize(results_dir: Path, presses_list: list[int]) -> None:
    folds = load_folds(results_dir)
    if not folds:
        raise SystemExit(f"No scores.npz under {results_dir}; run evaluate_open_set.py first.")
    by_n = {n: [fused_metrics(data, n) for _, data in folds] for n in presses_list}

    def values(n: int, scorer: str, key: str) -> list[float]:
        return [fold[scorer][key] for fold in by_n[n]]

    example = json.loads((results_dir / folds[0][0] / "metadata.json").read_text(encoding="utf-8"))
    lines = [
        "# Deciding from several presses",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `evaluate_multi_press.py` from the {len(folds)} folds of "
        f"`evaluate_open_set.py` in this folder (preprocessing: {example['preprocess']}). See `summary.md` for how "
        "the folds are built.",
        "",
        "An attempt is n presses in a row by the same person on the same day. Their per-person scores are "
        "averaged, and the attempt is accepted when the best averaged score clears the threshold. Every run "
        "of n consecutive presses counts, so attempts overlap. As in `summary.md`, thresholds are picked on "
        "each fold's own test scores, and numbers are mean ± standard deviation over folds.",
        "",
        "## 1. Equal error rate, known vs stranger",
        "",
        "| Presses | " + " | ".join(SCORER_NAMES[s] for s in SCORERS) + " |",
        "|---|" + "---|" * len(SCORERS),
    ]
    for n in presses_list:
        lines.append(f"| {n} | " + " | ".join(pct_sd(values(n, s, "eer")) for s in SCORERS) + " |")

    for far in FAR_TARGETS:
        lines += [
            "",
            f"## {2 + FAR_TARGETS.index(far)}. Attempts accepted under the right name when {far * 100:g}% of "
            "stranger attempts get in",
            "",
            "| Presses | " + " | ".join(SCORER_NAMES[s] for s in SCORERS) + " |",
            "|---|" + "---|" * len(SCORERS),
        ]
        for n in presses_list:
            lines.append(f"| {n} | " + " | ".join(pct_sd(values(n, s, f"dir@{far:g}")) for s in SCORERS) + " |")

    section = 2 + len(FAR_TARGETS)
    lines += [
        "",
        f"## {section}. Closed-set accuracy and verification EER",
        "",
        "Closed-set accuracy: the best averaged score names the right person, with no rejection. "
        "Verification EER: the person claims a name first.",
        "",
        "| Presses | Known attempts per fold | Stranger attempts per fold | "
        + " | ".join(f"{SHORT_NAMES[s]} accuracy" for s in SCORERS) + " | "
        + " | ".join(f"{SHORT_NAMES[s]} verification EER" for s in SCORERS) + " |",
        "|---|---|---|" + "---|" * (2 * len(SCORERS)),
    ]
    for n in presses_list:
        known_attempts = int(np.median(values(n, SCORERS[0], "known_attempts")))
        stranger_attempts = int(np.median(values(n, SCORERS[0], "stranger_attempts")))
        lines.append(
            f"| {n} | {known_attempts} (median) | {stranger_attempts} (median) | "
            + " | ".join(pct_sd(values(n, s, "accuracy")) for s in SCORERS) + " | "
            + " | ".join(pct_sd(values(n, s, "verify_eer")) for s in SCORERS) + " |"
        )

    lines += ["", "Chart: `multi_press.png`."]
    (results_dir / "multi_press.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(by_n, presses_list, results_dir / "multi_press.png")
    print(f"Wrote {results_dir / 'multi_press.md'} and {results_dir / 'multi_press.png'}")


def plot(by_n: dict[int, list[dict[str, Any]]], presses_list: list[int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    panels = [
        (left, "eer", "EER, known vs stranger (%)", "Equal error rate (lower is better)"),
        (right, f"dir@{FAR_TARGETS[0]:g}", "Known attempts let in under the right name (%)",
         f"Right person let in when {FAR_TARGETS[0] * 100:g}% of strangers get in (higher is better)"),
    ]
    for ax, key, ylabel, title in panels:
        ends = []
        for scorer in SCORERS:
            means = [mean_sd([fold[scorer][key] for fold in by_n[n]])[0] * 100 for n in presses_list]
            ax.plot(presses_list, means, color=SCORER_COLORS[scorer], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=SCORER_NAMES[scorer], zorder=3)
            ends.append([means[-1], scorer])
        # Direct labels at the right end, nudged apart so they do not overlap.
        ends.sort()
        for index in range(1, len(ends)):
            ends[index][0] = max(ends[index][0], ends[index - 1][0] + 3.5)
        for y, scorer in ends:
            ax.annotate(SHORT_NAMES[scorer], (presses_list[-1], y), textcoords="offset points", xytext=(10, -3),
                        color=INK, fontsize=9, annotation_clip=False)
        ax.set_xticks(presses_list)
        ax.set_xlim(presses_list[0] - 0.3, presses_list[-1] + 0.3)
        ax.set_ylim(0, 100 if key != "eer" else 50)
        ax.set_xlabel("Presses per attempt", color=INK_SECONDARY)
        ax.set_ylabel(ylabel, color=INK_SECONDARY)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY)
    left.legend(frameon=False, loc="lower left", labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="experiments/open_set",
                        help="Output folder of evaluate_open_set.py.")
    parser.add_argument("--presses", nargs="+", type=int, default=[1, 2, 3, 5])
    args = parser.parse_args(argv)
    presses_list = sorted(set(args.presses))
    if presses_list[0] < 1:
        raise SystemExit("--presses must be 1 or more")
    summarize(Path(args.results_dir), presses_list)


if __name__ == "__main__":
    main()
