"""Compare the old (resize) and new (onset) preprocessing fairly.

Both methods are trained with the same settings and the same seeds on the same people:

- Held-out day: each recording day is held out once and used only for validation, so the
  model is tested on a session it never saw. This is the main comparison, and the closest
  to a door handle meeting a new day.
- Random split: recordings shuffled across days, so validation shares sessions with
  training. Only used to measure how much each method drops from a same-session test to a
  new-day test; a large drop means the model leans on session cues such as the baseline.

Finished runs are skipped, so an interrupted run can be resumed by running the script again.
Results go to the output folder: one sub-folder per run, plus results.csv, summary.md and
comparison.png.

Usage:
    python compare_methods.py --dry-run          # list the runs, train nothing
    python compare_methods.py                    # train every missing run, then summarize
    python compare_methods.py --summarize-only   # rebuild the summary from finished runs
"""
from __future__ import annotations

import argparse
import csv
import json
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

from force_matrix_biometrics.cnn import DEFAULT_LABELS, list_recording_days

PROJECT_ROOT = Path(__file__).resolve().parent
METHODS = ("resize", "onset")
METHOD_NAMES = {"resize": "Old (resize)", "onset": "New (onset)"}
LAST_EPOCHS = 10  # "last-10" accuracy: mean validation accuracy over the final epochs

# Chart colors: categorical slots 1-2 (validated pair) and chart chrome, light mode.
METHOD_COLORS = {"resize": "#2a78d6", "onset": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"


@dataclass(frozen=True)
class Run:
    method: str
    split: str  # "day" or "random"
    day: str | None
    seed: int

    @property
    def name(self) -> str:
        where = f"day{self.day}" if self.split == "day" else "random"
        return f"{self.method}_{where}_seed{self.seed}"


def plan_runs(days: list[str], methods: list[str], seeds: list[int]) -> list[Run]:
    runs = []
    for method in methods:
        for day in days:
            runs.extend(Run(method, "day", day, seed) for seed in seeds)
        runs.extend(Run(method, "random", None, seed) for seed in seeds)
    return runs


def train_command(run: Run, args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable, str(PROJECT_ROOT / "train_cnn.py"),
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--output-dir", str(run_dir),
        "--preprocess", run.method,
        "--split-strategy", run.split,
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--seed", str(run.seed),
    ]
    if run.day:
        command += ["--validation-day", run.day]
    return command


def settings_mismatch(run: Run, meta: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Settings in a finished run's metadata that differ from what this comparison asks for."""
    expected = {
        "preprocess": run.method,
        "split_strategy": run.split,
        "validation_day": run.day,
        "seed": run.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    return [f"{key}={meta.get(key)!r} (want {value!r})" for key, value in expected.items() if meta.get(key) != value]


def load_metadata(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def execute(runs: list[Run], args: argparse.Namespace, out_dir: Path) -> None:
    todo = []
    for run in runs:
        meta = load_metadata(out_dir / run.name)
        if meta is None:
            todo.append(run)
            continue
        mismatch = settings_mismatch(run, meta, args)
        if mismatch:
            raise SystemExit(
                f"{out_dir / run.name} was trained with different settings: {', '.join(mismatch)}. "
                "Use another --output-dir, or delete that folder to retrain it."
            )
    print(f"{len(runs) - len(todo)} of {len(runs)} runs already finished; training {len(todo)}.", flush=True)
    if not todo:
        return

    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def work(run: Run) -> tuple[Run, int, float]:
        run_dir = out_dir / run.name
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                train_command(run, args, run_dir), cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env
            )
        return run, result.returncode, time.time() - start

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(work, run) for run in todo]
        for future in as_completed(futures):
            run, code, seconds = future.result()
            done += 1
            status = "ok" if code == 0 else f"FAILED (exit {code}, see {run.name}/train.log)"
            if code != 0:
                failed.append(run.name)
            print(f"[{done}/{len(todo)}] {run.name}: {status} in {seconds:.0f}s", flush=True)
    if failed:
        print(f"{len(failed)} runs failed: {', '.join(failed)}. Run the script again to retry them.", flush=True)


# ---------------------------------------------------------------- summary


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def pct(value: float) -> str:
    return "–" if value != value else f"{value * 100:.1f}%"


def pct_sd(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    if mean != mean:
        return "–"
    return f"{mean * 100:.1f}% ± {sd * 100:.1f}" if len(values) > 1 else f"{mean * 100:.1f}%"


def points(value: float) -> str:
    return "–" if value != value else f"{value * 100:+.1f}"


def collect(runs: list[Run], args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """One record per finished run. Runs that are missing or trained with other settings are reported."""
    records = []
    problems = []
    for run in runs:
        meta = load_metadata(out_dir / run.name)
        if meta is None:
            problems.append(f"{run.name}: not trained yet")
            continue
        mismatch = settings_mismatch(run, meta, args)
        if mismatch:
            problems.append(f"{run.name}: different settings ({', '.join(mismatch)}), left out")
            continue
        history = meta["history"]
        matrix = meta["confusion_matrix"]["matrix"]
        labels = meta["labels"]
        records.append({
            "run": run,
            "labels": labels,
            "n": meta["validation_samples"],
            "final": history[-1]["validation_accuracy"],
            "last10": statistics.mean(epoch["validation_accuracy"] for epoch in history[-LAST_EPOCHS:]),
            "correct": {label: matrix[i][i] for i, label in enumerate(labels)},
            "total": {label: sum(matrix[i]) for i, label in enumerate(labels)},
        })
    return records, problems


def pooled(records: list[dict[str, Any]], key: str) -> float:
    """Accuracy over all validation recordings of these runs (weighted by validation size)."""
    total = sum(record["n"] for record in records)
    return sum(record[key] * record["n"] for record in records) / total if total else float("nan")


def write_results_csv(records: list[dict[str, Any]], labels: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["run", "method", "split", "validation_day", "seed", "validation_samples",
             "final_accuracy", f"last{LAST_EPOCHS}_accuracy"] + [f"accuracy_{label}" for label in labels]
        )
        for record in records:
            run = record["run"]
            per_person = [
                f"{record['correct'][label] / record['total'][label]:.4f}" if record["total"].get(label) else ""
                for label in labels
            ]
            writer.writerow(
                [run.name, run.method, run.split, run.day or "", run.seed, record["n"],
                 f"{record['final']:.4f}", f"{record['last10']:.4f}"] + per_person
            )


def summarize(runs: list[Run], args: argparse.Namespace, out_dir: Path) -> None:
    records, problems = collect(runs, args, out_dir)
    if not records:
        print("No finished runs to summarize.")
        return
    labels = records[0]["labels"]
    methods = [method for method in METHODS if any(r["run"].method == method for r in records)]
    days = sorted({r["run"].day for r in records if r["run"].day})
    seeds = sorted({r["run"].seed for r in records})

    def select(method: str, split: str, day: str | None = None, seed: int | None = None) -> list[dict[str, Any]]:
        return [
            r for r in records
            if r["run"].method == method and r["run"].split == split
            and (day is None or r["run"].day == day) and (seed is None or r["run"].seed == seed)
        ]

    def per_seed_by_seed(method: str, split: str, key: str) -> dict[int, float]:
        """For each seed: accuracy pooled over every validation recording of that split."""
        values = {}
        for seed in seeds:
            chosen = select(method, split, seed=seed)
            if split == "day" and len(chosen) != len(days):
                continue  # this seed has not finished every day yet
            if chosen:
                values[seed] = pooled(chosen, key)
        return values

    def per_seed(method: str, split: str, key: str) -> list[float]:
        return list(per_seed_by_seed(method, split, key).values())

    write_results_csv(records, labels, out_dir / "results.csv")

    lines = [
        "# Old vs new preprocessing",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `compare_methods.py`. "
        f"Settings for every run: {args.epochs} epochs, learning rate {args.learning_rate:g}, "
        f"batch size {args.batch_size}, seeds {', '.join(map(str, seeds))}, people: {', '.join(labels)}.",
        "",
        "Accuracy is the validation accuracy after the last epoch, as mean ± standard deviation over the seeds. "
        f"Last-{LAST_EPOCHS} is the mean over the final {LAST_EPOCHS} epochs, which is less sensitive to where "
        "training happened to stop. Per-run numbers are in `results.csv`.",
        "",
    ]
    if problems:
        lines += ["**Incomplete:** " + "; ".join(problems), ""]

    # 1. Held-out day, one row per day.
    header = "| Held-out day | Recordings | People tested | " + " | ".join(METHOD_NAMES[m] for m in methods)
    header += " | New − Old (points) |" if len(methods) == 2 else " |"
    lines += [
        "## 1. Tested on a day the model never saw",
        "",
        "Each row trains on every other day and validates on that day only.",
        "",
        header,
        "|" + "---|" * (header.count("|") - 1),
    ]
    for day in days:
        day_records = [r for m in methods for r in select(m, "day", day)]
        if not day_records:
            continue
        example = day_records[0]
        tested = [label for label in labels if example["total"].get(label)]
        cells = [pct_sd([r["final"] for r in select(m, "day", day)]) for m in methods]
        row = f"| {day[:4]}-{day[4:6]}-{day[6:]} | {example['n']} | {', '.join(tested)} | " + " | ".join(cells)
        if len(methods) == 2:
            diff = mean_sd([r["final"] for r in select("onset", "day", day)])[0] - mean_sd(
                [r["final"] for r in select("resize", "day", day)])[0]
            row += f" | {points(diff)}"
        lines.append(row + " |")

    # 2. Overall, pooled over all held-out days.
    lines += [
        "",
        "## 2. All days together",
        "",
        "Every recording is validated exactly once, by a model that never saw its day. "
        "Accuracy is pooled over all of them, per seed, then averaged over the seeds.",
        "",
        "| Method | Accuracy (last epoch) | Accuracy (last-10) |",
        "|---|---|---|",
    ]
    for method in methods:
        lines.append(
            f"| {METHOD_NAMES[method]} | {pct_sd(per_seed(method, 'day', 'final'))} | "
            f"{pct_sd(per_seed(method, 'day', 'last10'))} |"
        )

    # 3. Random split vs held-out day.
    lines += [
        "",
        "## 3. Same-session test vs new-day test",
        "",
        "The random split validates on recordings from sessions the model trained on. The drop to the "
        "held-out-day accuracy shows how much a method relies on session cues such as the sensor "
        "baseline. Compare the drops, not the random-split scores across methods.",
        "",
        "| Method | Random split (last-10) | Held-out day (last-10) | Drop (points) |",
        "|---|---|---|---|",
    ]
    drops: dict[str, tuple[float, float]] = {}
    for method in methods:
        random_by_seed = per_seed_by_seed(method, "random", "last10")
        day_by_seed = per_seed_by_seed(method, "day", "last10")
        random_values = list(random_by_seed.values())
        day_values = list(day_by_seed.values())
        paired = [random_by_seed[seed] - day_by_seed[seed] for seed in random_by_seed if seed in day_by_seed]
        if paired:
            drops[method] = mean_sd(paired)
            drop_text = f"{drops[method][0] * 100:.1f} ± {drops[method][1] * 100:.1f}"
        else:
            drop_text = "–"
        lines.append(f"| {METHOD_NAMES[method]} | {pct_sd(random_values)} | {pct_sd(day_values)} | {drop_text} |")

    # 4. Per person, pooled over held-out days.
    lines += [
        "",
        "## 4. Per person, all held-out days together",
        "",
        "Share of each person's recordings recognised by a model that never saw that day, averaged over the seeds.",
        "",
        "| Person | Recordings | " + " | ".join(METHOD_NAMES[m] for m in methods)
        + (" | New − Old (points) |" if len(methods) == 2 else " |"),
        "|" + "---|" * (len(methods) + 2 + (1 if len(methods) == 2 else 0)),
    ]
    for label in labels:
        means = {}
        recordings = 0
        for method in methods:
            seed_accuracies = []
            for seed in seeds:
                chosen = select(method, "day", seed=seed)
                total = sum(r["total"].get(label, 0) for r in chosen)
                if total and len(chosen) == len(days):
                    seed_accuracies.append(sum(r["correct"].get(label, 0) for r in chosen) / total)
                    recordings = total
            means[method] = seed_accuracies
        row = f"| {label} | {recordings} | " + " | ".join(pct_sd(means[m]) for m in methods)
        if len(methods) == 2:
            row += f" | {points(mean_sd(means['onset'])[0] - mean_sd(means['resize'])[0])}"
        lines.append(row + " |")

    reference = out_dir / "reference_old_splits.md"
    if reference.exists():
        lines += ["", f"Earlier single runs of the old method on the old random and time splits: see `{reference.name}`."]

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_comparison(methods, days, select, per_seed, drops, out_dir / "comparison.png")
    print(f"Wrote {out_dir / 'results.csv'}, {out_dir / 'summary.md'} and {out_dir / 'comparison.png'}")
    if problems:
        print(f"Incomplete: {len(problems)} runs missing or left out (listed in summary.md).")


def plot_comparison(methods, days, select, per_seed, drops, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 1.2]}, facecolor=SURFACE
    )
    categories = [f"{day[4:6]}-{day[6:]}" for day in days] + ["All days"]
    offsets = {method: (index - (len(methods) - 1) / 2) * 0.22 for index, method in enumerate(methods)}
    lowest = 1.0
    for method in methods:
        means, sds = [], []
        for day in days:
            mean, sd = mean_sd([r["final"] for r in select(method, "day", day)])
            means.append(mean)
            sds.append(sd)
        mean, sd = mean_sd(per_seed(method, "day", "final"))
        means.append(mean)
        sds.append(sd)
        xs = [index + offsets[method] for index in range(len(categories))]
        left.errorbar(
            xs, [m * 100 for m in means], yerr=[s * 100 for s in sds], fmt="o", markersize=8,
            color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method], elinewidth=2, capsize=0,
            markeredgecolor=SURFACE, markeredgewidth=2, label=METHOD_NAMES[method], zorder=3,
        )
        left.annotate(f"{means[-1] * 100:.1f}%", (xs[-1], means[-1] * 100), textcoords="offset points",
                      xytext=(10, -4), color=INK, fontsize=9)
        lowest = min([lowest] + [m - s for m, s in zip(means, sds) if m == m])
    left.set_xticks(range(len(categories)), categories)
    left.set_ylim(max(0, (lowest * 100) // 5 * 5 - 5), 100)
    left.set_ylabel("Accuracy on the held-out day (%)", color=INK_SECONDARY)
    left.set_title("Tested on a day the model never saw", color=INK, loc="left", fontsize=11)
    left.legend(frameon=False, loc="lower left", labelcolor=INK)

    names = [METHOD_NAMES[m] for m in methods if m in drops]
    values = [drops[m][0] * 100 for m in methods if m in drops]
    errors = [drops[m][1] * 100 for m in methods if m in drops]
    colors = [METHOD_COLORS[m] for m in methods if m in drops]
    if values:
        bars = right.bar(names, values, yerr=errors, color=colors, width=0.5, ecolor=INK_SECONDARY, capsize=0,
                         edgecolor=SURFACE, linewidth=2)
        for bar, value, error in zip(bars, values, errors):
            # Above the error bar, so the whisker does not run through the number.
            right.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, max(value, 0) + error),
                           textcoords="offset points", xytext=(0, 4), ha="center", color=INK, fontsize=9)
    right.axhline(0, color=INK_SECONDARY, linewidth=1)
    right.set_ylabel("Random split minus held-out day (points)", color=INK_SECONDARY)
    right.set_title("Drop on a new day (last-10)", color=INK, loc="left", fontsize=11)

    for ax in (left, right):
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


# ---------------------------------------------------------------- main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default="experiments/method_comparison")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--days", nargs="+", help="Days to hold out (YYYYMMDD). Default: every day in the dataset.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--parallel", type=int, default=3, help="Runs trained at the same time.")
    parser.add_argument("--dry-run", action="store_true", help="List the runs and exit without training.")
    parser.add_argument("--summarize-only", action="store_true", help="Skip training; rebuild the summary.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir).resolve()
    days = args.days or list_recording_days(args.dataset_root, DEFAULT_LABELS)
    runs = plan_runs(days, args.methods, args.seeds)

    if args.dry_run:
        print(f"{len(runs)} runs -> {out_dir}")
        for run in runs:
            state = "done" if (out_dir / run.name / "metadata.json").exists() else "to train"
            print(f"  {run.name:28s} {state:9s} {' '.join(train_command(run, args, out_dir / run.name)[1:])}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        execute(runs, args, out_dir)
    summarize(runs, args, out_dir)


if __name__ == "__main__":
    main()
