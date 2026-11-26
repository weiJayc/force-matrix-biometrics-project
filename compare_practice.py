"""Does leaving people's practice presses out of training help on a new day?

On their first day with the sensor, people take about 40 presses to settle into a consistent
grip. This trains the onset model on the held-out-day split three ways and compares them:

- Baseline: every recording (the onset runs of compare_methods.py, reused, same settings).
- Practice skipped: each person's first --skip-practice recordings of their first day are
  left out of training (--skip-practice N --skip-practice-from start).
- Control: the LAST N recordings of that same day are left out instead, so the same amount of
  data from the same day goes; any gain beyond the control is due to the practice presses.

Validation is never changed. Only held-out days whose training set loses recordings are
trained; when the held-out day is everyone's first day, the practice presses are validation
data and training is identical to the baseline.

Usage:
    python compare_practice.py --dry-run
    python compare_practice.py
    python compare_practice.py --summarize-only
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from compare_methods import GRID, INK, INK_SECONDARY, LAST_EPOCHS, SURFACE, load_metadata, mean_sd, pct_sd
from force_matrix_biometrics.cnn import (
    DEFAULT_LABELS,
    find_labeled_csv_files,
    list_recording_days,
    practice_recordings,
    recording_day,
)

PROJECT_ROOT = Path(__file__).resolve().parent
VARIANTS = ("baseline", "practice", "control")
VARIANT_NAMES = {
    "baseline": "All recordings",
    "practice": "Practice skipped",
    "control": "Control (last N of the day skipped)",
}
# Categorical slots 1-3 of the validated default palette, light mode.
VARIANT_COLORS = {"baseline": "#257de0", "practice": "#f47230", "control": "#00b894"}


@dataclass(frozen=True)
class Run:
    variant: str
    day: str
    seed: int

    def folder(self, args: argparse.Namespace) -> Path:
        if self.variant == "baseline":
            return Path(args.baseline_dir) / f"onset_day{self.day}_seed{self.seed}"
        return Path(args.output_dir) / f"{self.variant}{args.skip_practice}_day{self.day}_seed{self.seed}"


def affected_days(dataset_root: str, count: int) -> tuple[list[str], list[str]]:
    """Held-out days whose training set loses recordings, and the days where nothing changes."""
    files = find_labeled_csv_files(dataset_root, DEFAULT_LABELS)
    days = list_recording_days(dataset_root, DEFAULT_LABELS)
    first_days = {recording_day(practice_recordings(paths, count)[0]) for paths in files.values()}
    changed = [day for day in days if any(first != day for first in first_days)]
    return changed, [day for day in days if day not in changed]


def train_command(run: Run, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(PROJECT_ROOT / "train_cnn.py"),
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--output-dir", str(run.folder(args).resolve()),
        "--preprocess", "onset",
        "--split-strategy", "day",
        "--validation-day", run.day,
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--seed", str(run.seed),
        "--skip-practice", str(args.skip_practice),
        "--skip-practice-from", "start" if run.variant == "practice" else "end",
    ]


def settings_mismatch(run: Run, meta: dict[str, Any], args: argparse.Namespace) -> list[str]:
    expected = {
        "preprocess": "onset",
        "split_strategy": "day",
        "validation_day": run.day,
        "seed": run.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "labels": list(DEFAULT_LABELS),
        "skip_practice": 0 if run.variant == "baseline" else args.skip_practice,
    }
    if run.variant != "baseline":
        expected["skip_practice_from"] = "start" if run.variant == "practice" else "end"
    actual = dict(meta, skip_practice=meta.get("skip_practice", 0))
    return [f"{key}={actual.get(key)!r} (want {value!r})" for key, value in expected.items() if actual.get(key) != value]


def execute(runs: list[Run], args: argparse.Namespace) -> None:
    todo = []
    for run in runs:
        meta = load_metadata(run.folder(args))
        if meta is None:
            if run.variant == "baseline":
                raise SystemExit(f"Baseline run {run.folder(args)} is missing; run compare_methods.py first.")
            todo.append(run)
            continue
        mismatch = settings_mismatch(run, meta, args)
        if mismatch:
            raise SystemExit(f"{run.folder(args)} was trained with different settings: {', '.join(mismatch)}.")
    print(f"Training {len(todo)} runs ({len(runs) - len(todo)} already done or reused).", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def work(run: Run) -> tuple[Run, int, float]:
        folder = run.folder(args)
        folder.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with (folder / "train.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(train_command(run, args), cwd=PROJECT_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, env=env)
        return run, result.returncode, time.time() - start

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(work, run) for run in todo]
        for done, future in enumerate(as_completed(futures), 1):
            run, code, seconds = future.result()
            status = "ok" if code == 0 else f"FAILED (exit {code}, see train.log)"
            print(f"[{done}/{len(todo)}] {run.folder(args).name}: {status} in {seconds:.0f}s", flush=True)


def record(run: Run, args: argparse.Namespace) -> dict[str, Any] | None:
    meta = load_metadata(run.folder(args))
    if meta is None or settings_mismatch(run, meta, args):
        return None
    matrix = meta["confusion_matrix"]["matrix"]
    labels = meta["labels"]
    return {
        "final": meta["history"][-1]["validation_accuracy"],
        "last10": statistics.mean(e["validation_accuracy"] for e in meta["history"][-LAST_EPOCHS:]),
        "n": meta["validation_samples"],
        "correct": {label: matrix[i][i] for i, label in enumerate(labels)},
        "total": {label: sum(matrix[i]) for i, label in enumerate(labels)},
        "train": meta["train_samples"],
    }


def summarize(days: list[str], unchanged: list[str], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    results = {(v, d, s): record(Run(v, d, s), args) for v in VARIANTS for d in days for s in args.seeds}
    missing = [k for k, r in results.items() if r is None]

    def values(variant: str, day: str, key: str) -> list[float]:
        return [results[(variant, day, s)][key] for s in args.seeds if results[(variant, day, s)]]

    def pooled(variant: str, seed: int, key: str) -> float | None:
        chosen = [results[(variant, d, seed)] for d in days]
        if any(r is None for r in chosen):
            return None
        return sum(r[key] * r["n"] for r in chosen) / sum(r["n"] for r in chosen)

    def paired(variant: str, key: str, day: str | None = None) -> str:
        diffs = []
        for s in args.seeds:
            if day is None:
                a, b = pooled(variant, s, key), pooled("baseline", s, key)
            else:
                ra, rb = results[(variant, day, s)], results[("baseline", day, s)]
                a, b = (ra[key], rb[key]) if ra and rb else (None, None)
            if a is not None and b is not None:
                diffs.append(a - b)
        mean, sd = mean_sd(diffs)
        return "–" if mean != mean else f"{mean * 100:+.1f} ± {sd * 100:.1f}"

    example = next(r for r in results.values() if r)
    lines = [
        "# Leaving practice presses out of training",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `compare_practice.py`. Onset preprocessing, held-out-day split, "
        f"{args.epochs} epochs, learning rate {args.learning_rate:g}, batch size {args.batch_size}, "
        f"seeds {', '.join(map(str, args.seeds))}. N = {args.skip_practice}: each person's first {args.skip_practice} "
        "recordings of their first day are skipped (practice), or the last "
        f"{args.skip_practice} of that day (control). Baseline runs are reused from `{Path(args.baseline_dir).as_posix()}`.",
        "",
        "Validation is never changed. "
        + (f"Held-out {', '.join(f'{d[:4]}-{d[4:6]}-{d[6:]}' for d in unchanged)} is not listed: the practice "
           "presses are on that day, so they are validation data there and training is the same as the baseline. "
           if unchanged else "")
        + "Accuracy is mean ± standard deviation over seeds; last-10 averages the final 10 epochs. "
        "Differences are paired by seed, in points.",
        "",
    ]
    if missing:
        lines += [f"**Incomplete:** {len(missing)} runs missing.", ""]
    lines += [
        "## 1. Accuracy on the held-out day (last-10)",
        "",
        "| Held-out day | Training recordings (baseline → skipped) | "
        + " | ".join(VARIANT_NAMES[v] for v in VARIANTS) + " | Practice − baseline | Control − baseline |",
        "|---|---|" + "---|" * (len(VARIANTS) + 2),
    ]
    for day in days:
        trains = [results[(v, day, args.seeds[0])]["train"] for v in ("baseline", "practice") if results[(v, day, args.seeds[0])]]
        lines.append(
            f"| {day[:4]}-{day[4:6]}-{day[6:]} | {' → '.join(map(str, trains))} | "
            + " | ".join(pct_sd(values(v, day, "last10")) for v in VARIANTS)
            + f" | {paired('practice', 'last10', day)} | {paired('control', 'last10', day)} |"
        )
    pooled_cells = []
    for v in VARIANTS:
        per_seed = [x for x in (pooled(v, s, "last10") for s in args.seeds) if x is not None]
        pooled_cells.append(pct_sd(per_seed))
    lines.append(
        f"| **These days pooled** | | " + " | ".join(pooled_cells)
        + f" | {paired('practice', 'last10')} | {paired('control', 'last10')} |"
    )

    lines += [
        "",
        "## 2. Same, accuracy after the last epoch",
        "",
        "| Held-out day | " + " | ".join(VARIANT_NAMES[v] for v in VARIANTS) + " | Practice − baseline | Control − baseline |",
        "|---|" + "---|" * (len(VARIANTS) + 2),
    ]
    for day in days:
        lines.append(
            f"| {day[:4]}-{day[4:6]}-{day[6:]} | " + " | ".join(pct_sd(values(v, day, "final")) for v in VARIANTS)
            + f" | {paired('practice', 'final', day)} | {paired('control', 'final', day)} |"
        )

    # Who keeps other days to train on? A person whose only training day is their first day
    # loses part of their ONLY training data when practice presses are skipped.
    files = find_labeled_csv_files(args.dataset_root, DEFAULT_LABELS)
    first_day = {label: min(recording_day(p) for p in paths) for label, paths in files.items()}
    only_first = {
        (day, label) for day in days for label, paths in files.items()
        if {recording_day(p) for p in paths} - {day} == {first_day[label]}
    }

    def group_accuracy(variant: str, seed: int, single: bool) -> float | None:
        correct = total = 0
        for day in days:
            r = results[(variant, day, seed)]
            if r is None:
                return None
            for label, n in r["total"].items():
                if n and ((day, label) in only_first) == single:
                    correct += r["correct"][label]
                    total += n
        return correct / total if total else None

    lines += [
        "",
        "## 3. People with other training days vs people whose only training day is their first day",
        "",
        "After the last epoch, all held-out days above pooled, per seed, then mean ± SD. Skipping practice presses "
        "costs the second group part of their only training data.",
        "",
        "| Group | Who (held-out day) | " + " | ".join(VARIANT_NAMES[v] for v in VARIANTS)
        + " | Practice − baseline | Control − baseline | Practice − control |",
        "|---|---|" + "---|" * (len(VARIANTS) + 3),
    ]
    for single, title in ((False, "Other training days too"), (True, "Only their first day")):
        who = sorted(f"{label} ({day[4:6]}-{day[6:]})" for day, label in only_first) if single else ["everyone else"]
        per_variant = {v: [group_accuracy(v, s, single) for s in args.seeds] for v in VARIANTS}
        if any(x is None for xs in per_variant.values() for x in xs):
            continue

        def diff(a: str, b: str) -> str:
            mean, sd = mean_sd([x - y for x, y in zip(per_variant[a], per_variant[b])])
            return f"{mean * 100:+.1f} ± {sd * 100:.1f}"

        lines.append(
            f"| {title} | {', '.join(who)} | " + " | ".join(pct_sd(per_variant[v]) for v in VARIANTS)
            + f" | {diff('practice', 'baseline')} | {diff('control', 'baseline')} | {diff('practice', 'control')} |"
        )

    lines += [
        "",
        "## 4. Per person on each held-out day (after the last epoch, mean over seeds)",
        "",
        "| Held-out day | Person | Recordings | " + " | ".join(VARIANT_NAMES[v] for v in VARIANTS) + " |",
        "|---|---|---|" + "---|" * len(VARIANTS),
    ]
    for day in days:
        for label in example["total"]:
            cells = []
            total = 0
            for v in VARIANTS:
                accs = []
                for s in args.seeds:
                    r = results[(v, day, s)]
                    if r and r["total"].get(label):
                        accs.append(r["correct"][label] / r["total"][label])
                        total = r["total"][label]
                cells.append(pct_sd(accs) if accs else "–")
            if total:
                lines.append(f"| {day[:4]}-{day[4:6]}-{day[6:]} | {label} | {total} | " + " | ".join(cells) + " |")

    lines += ["", "Chart: `practice_comparison.png`."]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(days, values, pooled, args, out_dir / "practice_comparison.png")
    print(f"Wrote {out_dir / 'summary.md'} and {out_dir / 'practice_comparison.png'}")
    if missing:
        print(f"Incomplete: {len(missing)} runs missing.")


def plot(days, values, pooled, args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    categories = [f"{d[4:6]}-{d[6:]}" for d in days] + ["Pooled"]
    offsets = {v: (i - (len(VARIANTS) - 1) / 2) * 0.22 for i, v in enumerate(VARIANTS)}
    lowest = 1.0
    for v in VARIANTS:
        means, sds = [], []
        for d in days:
            mean, sd = mean_sd(values(v, d, "last10"))
            means.append(mean)
            sds.append(sd)
        mean, sd = mean_sd([x for x in (pooled(v, s, "last10") for s in args.seeds) if x is not None])
        means.append(mean)
        sds.append(sd)
        xs = [i + offsets[v] for i in range(len(categories))]
        ax.errorbar(xs, [m * 100 for m in means], yerr=[s * 100 for s in sds], fmt="o", markersize=8,
                    color=VARIANT_COLORS[v], ecolor=VARIANT_COLORS[v], elinewidth=2, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=VARIANT_NAMES[v], zorder=3)
        lowest = min([lowest] + [m - s for m, s in zip(means, sds) if m == m])
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylim(max(0, (lowest * 100) // 5 * 5 - 5), 100)
    ax.set_xlabel("Held-out day", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy on the held-out day, last-10 (%)", color=INK_SECONDARY)
    ax.set_title(f"Leaving each person's first {args.skip_practice} presses out of training",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK, fontsize=8)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default="experiments/practice_comparison")
    parser.add_argument("--baseline-dir", default="experiments/method_comparison",
                        help="compare_methods.py output holding the onset_day<day>_seed<n> baseline runs.")
    parser.add_argument("--skip-practice", type=int, default=40, help="Recordings skipped per person.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    days, unchanged = affected_days(args.dataset_root, args.skip_practice)
    runs = [Run(v, d, s) for v in VARIANTS for d in days for s in args.seeds]
    if args.dry_run:
        print(f"Held-out days whose training changes: {', '.join(days)}; unchanged: {', '.join(unchanged) or 'none'}")
        for run in runs:
            state = "reused" if run.variant == "baseline" else (
                "done" if (run.folder(args) / "metadata.json").exists() else "to train")
            print(f"  {run.folder(args).as_posix():60s} {state}")
        return
    if not args.summarize_only:
        execute(runs, args)
    summarize(days, unchanged, args)


if __name__ == "__main__":
    main()
