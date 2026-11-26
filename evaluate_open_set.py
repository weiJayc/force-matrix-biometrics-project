"""Can the CNN reject a stranger? Leave-one-person-out open-set evaluation.

The CNN is a closed-set classifier: it always answers with one of the people it was
trained on. This script measures how well a score computed from the trained model tells
the enrolled people apart from someone it has never seen.

Each fold leaves one person out as the stranger and holds out one recording day:

- Training and enrollment: the other people's recordings from every other day. The model
  trains on them, and the same recordings form each person's enrollment templates.
- Known test: the other people's recordings from the held-out day (a session the model
  never saw, as in compare_methods.py).
- Stranger test: every recording of the left-out person (the model never saw them at all).

Scores (higher = more like that enrolled person):

- softmax:  the class probability.
- centroid: cosine similarity between the model's feature vector and each person's
            enrollment centroid.
- knn:      mean cosine similarity to the person's 5 nearest enrollment feature vectors.

The feature vector is the model's embed(): for the classifier (--loss cross_entropy) the 32
values before the last layer, for --loss arcface the learned embedding. Feature vectors are
centered on the mean enrollment vector, then scaled to length 1.

Train each kind of model into its own --output-dir, for example:
    python evaluate_open_set.py                                    # classifier
    python evaluate_open_set.py --loss arcface --cross-day-weight 1 --learning-rate 0.001 \
        --output-dir experiments/open_set_arcface_crossday

Training runs that already finished are skipped, so an interrupted run can be resumed.
Results go to the output folder: one sub-folder per fold (model, train.log, scores.npz),
plus results.csv, summary.md and open_set.png.

Usage:
    python evaluate_open_set.py --dry-run          # list the folds, train nothing
    python evaluate_open_set.py                    # train missing folds, score them, summarize
    python evaluate_open_set.py --summarize-only   # rebuild the summary from saved scores
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from force_matrix_biometrics.cnn import (
    DEFAULT_ARCFACE_CONFIG,
    DEFAULT_LABELS,
    DEFAULT_MAX_VALUE,
    LOSS_CHOICES,
    ArcFaceConfig,
    OnsetConfig,
    _require_torch,
    build_dataset_bundle,
    build_model_from_metadata,
    list_recording_days,
    recording_day,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SCORERS = ("softmax", "centroid", "knn")
SCORER_NAMES = {
    "softmax": "Softmax probability",
    "centroid": "Centroid distance",
    "knn": "5-NN distance",
}
SHORT_NAMES = {"softmax": "Softmax", "centroid": "Centroid", "knn": "5-NN"}
KNN_K = 5
FAR_TARGETS = (0.01, 0.05)
FAR_GRID = np.linspace(0.0, 1.0, 201)

# Chart colors: categorical slots 1-3 of the validated default palette, light mode.
SCORER_COLORS = {"softmax": "#257de0", "centroid": "#f47230", "knn": "#00b894"}
SURFACE = "#ffffff"
INK = "#191919"
INK_SECONDARY = "#666666"
GRID = "#e1e0d9"


@dataclass(frozen=True)
class Fold:
    method: str
    stranger: str
    day: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.method}_stranger-{self.stranger}_day{self.day}_seed{self.seed}"

    @property
    def known(self) -> list[str]:
        return [label for label in DEFAULT_LABELS if label != self.stranger]


def plan_folds(strangers: list[str], days: list[str], method: str, seeds: list[int]) -> list[Fold]:
    return [Fold(method, stranger, day, seed) for seed in seeds for stranger in strangers for day in days]


# ---------------------------------------------------------------- training


def arcface_config(args: argparse.Namespace) -> ArcFaceConfig:
    return ArcFaceConfig(
        embedding_dim=args.embedding_dim,
        scale=args.arcface_scale,
        margin=args.arcface_margin,
        cross_day_weight=args.cross_day_weight,
        temperature=args.contrastive_temperature,
        samples_per_person=args.samples_per_person,
    )


def train_command(fold: Fold, args: argparse.Namespace, fold_dir: Path) -> list[str]:
    loss_args = ["--loss", args.loss]
    if args.loss == "arcface":
        config = arcface_config(args)
        loss_args += [
            "--embedding-dim", str(config.embedding_dim),
            "--arcface-scale", str(config.scale),
            "--arcface-margin", str(config.margin),
            "--cross-day-weight", str(config.cross_day_weight),
            "--contrastive-temperature", str(config.temperature),
            "--samples-per-person", str(config.samples_per_person),
        ]
    return [
        sys.executable, str(PROJECT_ROOT / "train_cnn.py"),
        *loss_args,
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--output-dir", str(fold_dir),
        "--preprocess", fold.method,
        "--split-strategy", "day",
        "--validation-day", fold.day,
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--seed", str(fold.seed),
        "--labels", *fold.known,
    ]


def settings_mismatch(fold: Fold, meta: dict[str, Any], args: argparse.Namespace) -> list[str]:
    expected = {
        "preprocess": fold.method,
        "split_strategy": "day",
        "validation_day": fold.day,
        "seed": fold.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "labels": fold.known,
        # Models trained before --loss existed are classifiers.
        "loss": args.loss,
        "arcface": asdict(arcface_config(args)) if args.loss == "arcface" else None,
    }
    actual = dict(meta, loss=meta.get("loss") or "cross_entropy", arcface=meta.get("arcface"))
    return [f"{key}={actual.get(key)!r} (want {value!r})" for key, value in expected.items() if actual.get(key) != value]


def load_metadata(fold_dir: Path) -> dict[str, Any] | None:
    path = fold_dir / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def train_missing(folds: list[Fold], args: argparse.Namespace, out_dir: Path) -> None:
    todo = []
    for fold in folds:
        meta = load_metadata(out_dir / fold.name)
        if meta is None:
            todo.append(fold)
            continue
        mismatch = settings_mismatch(fold, meta, args)
        if mismatch:
            raise SystemExit(
                f"{out_dir / fold.name} was trained with different settings: {', '.join(mismatch)}. "
                "Use another --output-dir, or delete that folder to retrain it."
            )
    print(f"{len(folds) - len(todo)} of {len(folds)} folds already trained; training {len(todo)}.", flush=True)
    if not todo:
        return

    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def work(fold: Fold) -> tuple[Fold, int, float]:
        fold_dir = out_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with (fold_dir / "train.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                train_command(fold, args, fold_dir), cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env
            )
        return fold, result.returncode, time.time() - start

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(work, fold) for fold in todo]
        for future in as_completed(futures):
            fold, code, seconds = future.result()
            done += 1
            if code != 0:
                failed.append(fold.name)
            status = "ok" if code == 0 else f"FAILED (exit {code}, see {fold.name}/train.log)"
            print(f"[{done}/{len(todo)}] {fold.name}: {status} in {seconds:.0f}s", flush=True)
    if failed:
        print(f"{len(failed)} folds failed: {', '.join(failed)}. Run the script again to retry them.", flush=True)


# ---------------------------------------------------------------- scoring


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def extract(torch: Any, model: Any, samples: list) -> tuple[np.ndarray, np.ndarray]:
    """The model's feature vector (embed) and its class probabilities, for each sample."""
    tensor = torch.tensor([sample.frames for sample in samples], dtype=torch.float32).unsqueeze(1)
    tensor = tensor / DEFAULT_MAX_VALUE
    with torch.no_grad():
        features = model.embed(tensor)
        probabilities = torch.softmax(model(tensor), dim=1)
    return features.numpy(), probabilities.numpy()


def distance_scores(
    enroll: np.ndarray, enroll_y: np.ndarray, test: np.ndarray, class_count: int
) -> dict[str, np.ndarray]:
    """Cosine similarity of each test vector to each enrolled person: centroid and k-NN."""
    center = enroll.mean(axis=0)
    enroll_unit = unit_rows(enroll - center)
    test_unit = unit_rows(test - center)
    centroids = unit_rows(np.stack([enroll_unit[enroll_y == c].mean(axis=0) for c in range(class_count)]))
    similarity = test_unit @ enroll_unit.T
    knn = np.stack(
        [np.sort(similarity[:, enroll_y == c], axis=1)[:, -KNN_K:].mean(axis=1) for c in range(class_count)],
        axis=1,
    )
    return {"centroid": test_unit @ centroids.T, "knn": knn}


def score_fold(fold: Fold, args: argparse.Namespace, fold_dir: Path) -> None:
    """Write scores.npz: every scorer's score matrix for the known test and stranger recordings."""
    torch, _, _, _ = _require_torch()
    meta = load_metadata(fold_dir)
    labels = tuple(meta["labels"])
    onset_config = OnsetConfig(**meta["onset_config"]) if meta["preprocess"] == "onset" else None
    common = dict(
        dataset_root=args.dataset_root,
        target_frames=meta["target_frames"],
        preprocess=meta["preprocess"],
        onset_config=onset_config,
        # The model's own tail correction (fitted on its training split), so the stranger
        # is preprocessed exactly as a new recording would be.
        tail_correction=meta.get("tail_correction"),
    )
    known = build_dataset_bundle(
        labels=labels, split_strategy="day", validation_day=fold.day, seed=meta["seed"],
        skip_practice=meta.get("skip_practice", 0), skip_practice_from=meta.get("skip_practice_from", "start"),
        **common,
    )
    if known.train_counts != meta["train_counts"] or known.validation_counts != meta["validation_counts"]:
        raise SystemExit(f"{fold.name}: rebuilt split counts differ from metadata; has the dataset changed?")
    stranger = build_dataset_bundle(
        labels=(fold.stranger,), split_strategy="random", validation_ratio=0.0, seed=meta["seed"], **common
    )

    model = build_model_from_metadata(meta)
    model.load_state_dict(torch.load(fold_dir / "pressure_cnn.pt", map_location="cpu"))
    model.eval()

    enroll_features, _ = extract(torch, model, known.train_samples)
    known_features, known_probabilities = extract(torch, model, known.validation_samples)
    stranger_features, stranger_probabilities = extract(torch, model, stranger.train_samples)
    enroll_y = np.array([known.label_to_index[s.label] for s in known.train_samples])
    known_y = np.array([known.label_to_index[s.label] for s in known.validation_samples])

    known_scores = distance_scores(enroll_features, enroll_y, known_features, len(labels))
    stranger_scores = distance_scores(enroll_features, enroll_y, stranger_features, len(labels))
    known_scores["softmax"] = known_probabilities
    stranger_scores["softmax"] = stranger_probabilities

    # The softmax argmax must reproduce the accuracy recorded at the end of training.
    accuracy = float((known_probabilities.argmax(axis=1) == known_y).mean())
    recorded = meta["history"][-1]["validation_accuracy"]
    if abs(accuracy - recorded) > 1e-3:
        print(f"WARNING {fold.name}: re-scored accuracy {accuracy:.4f} differs from recorded {recorded:.4f}")

    arrays: dict[str, np.ndarray] = {
        "labels": np.array(labels),
        "known_y": known_y,
        "stranger_day": np.array([recording_day(s.path) for s in stranger.train_samples]),
        # File names start with the recording time, so they give each person's press order.
        "known_file": np.array([s.path.name for s in known.validation_samples]),
        "stranger_file": np.array([s.path.name for s in stranger.train_samples]),
    }
    for scorer in SCORERS:
        arrays[f"{scorer}_known"] = known_scores[scorer]
        arrays[f"{scorer}_stranger"] = stranger_scores[scorer]
    np.savez(fold_dir / "scores.npz", **arrays)


def score_missing(folds: list[Fold], args: argparse.Namespace, out_dir: Path) -> None:
    todo = []
    for fold in folds:
        fold_dir = out_dir / fold.name
        model_path = fold_dir / "pressure_cnn.pt"
        scores_path = fold_dir / "scores.npz"
        if not model_path.exists() or load_metadata(fold_dir) is None:
            continue
        if not scores_path.exists() or scores_path.stat().st_mtime < model_path.stat().st_mtime:
            todo.append(fold)
            continue
        with np.load(scores_path) as data:
            if "stranger_file" not in data.files:  # written before file names were saved
                todo.append(fold)
    print(f"Scoring {len(todo)} folds.", flush=True)
    for index, fold in enumerate(todo, 1):
        start = time.time()
        score_fold(fold, args, out_dir / fold.name)
        print(f"[{index}/{len(todo)}] scored {fold.name} in {time.time() - start:.0f}s", flush=True)


# ---------------------------------------------------------------- metrics


def accept_rate(scores: np.ndarray, threshold: float) -> float:
    return float((scores >= threshold).mean()) if len(scores) else float("nan")


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Equal error rate: accept when score >= threshold; the point where the share of
    genuine scores rejected equals the share of impostor scores accepted."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    genuine_sorted = np.sort(genuine)
    impostor_sorted = np.sort(impostor)
    frr = np.searchsorted(genuine_sorted, thresholds, side="left") / len(genuine)
    far = 1.0 - np.searchsorted(impostor_sorted, thresholds, side="left") / len(impostor)
    best = int(np.argmin(np.abs(far - frr)))
    return float((far[best] + frr[best]) / 2)


def auroc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Chance that a random genuine score is higher than a random impostor score."""
    greater = (genuine[:, None] > impostor[None, :]).mean()
    equal = (genuine[:, None] == impostor[None, :]).mean()
    return float(greater + 0.5 * equal)


def dir_at_far(top_known: np.ndarray, correct: np.ndarray, top_stranger: np.ndarray, far: float) -> float:
    """Detection and identification rate: the share of known recordings that are accepted AND
    named correctly, with the threshold set so that at most `far` of the strangers get in."""
    allowed = int(np.floor(far * len(top_stranger) + 1e-9))
    if allowed >= len(top_stranger):
        return float(correct.mean())
    threshold = np.sort(top_stranger)[::-1][allowed]
    return float(((top_known > threshold) & correct).mean())


def fold_metrics(fold: Fold, data: Any) -> dict[str, Any]:
    known_y = data["known_y"]
    stranger_day = data["stranger_day"]
    same_day = stranger_day == fold.day
    record: dict[str, Any] = {"fold": fold, "known_n": len(known_y), "stranger_n": len(stranger_day)}
    for scorer in SCORERS:
        known = data[f"{scorer}_known"]
        stranger = data[f"{scorer}_stranger"]
        top_known = known.max(axis=1)
        top_stranger = stranger.max(axis=1)
        correct = known.argmax(axis=1) == known_y
        record[scorer] = {
            "accuracy": float(correct.mean()),
            "auroc": auroc(top_known, top_stranger),
            "eer": eer(top_known, top_stranger),
            **{f"dir@{far:g}": dir_at_far(top_known, correct, top_stranger, far) for far in FAR_TARGETS},
            # Verification: each recording claims one identity. Genuine = a known recording
            # claiming its own name; impostor = a stranger recording claiming each enrolled name.
            "verify_eer": eer(known[np.arange(len(known_y)), known_y], stranger.ravel()),
            "dir_curve": [dir_at_far(top_known, correct, top_stranger, far) for far in FAR_GRID],
            # Strangers recorded on the held-out day vs on the days the model trained on.
            "eer_stranger_same_day": eer(top_known, top_stranger[same_day]) if same_day.any() else float("nan"),
            "eer_stranger_train_days": eer(top_known, top_stranger[~same_day]) if (~same_day).any() else float("nan"),
        }
    return record


# ---------------------------------------------------------------- summary


def mean_sd(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if v == v]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def pct_sd(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    if mean != mean:
        return "–"
    return f"{mean * 100:.1f}% ± {sd * 100:.1f}"


def pct(values: list[float]) -> str:
    mean, _ = mean_sd(values)
    return "–" if mean != mean else f"{mean * 100:.1f}%"


def write_results_csv(records: list[dict[str, Any]], path: Path) -> None:
    keys = ["accuracy", "auroc", "eer"] + [f"dir@{far:g}" for far in FAR_TARGETS] + [
        "verify_eer", "eer_stranger_same_day", "eer_stranger_train_days"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["fold", "method", "stranger", "held_out_day", "seed", "known_test", "stranger_test"]
                        + [f"{scorer}_{key}" for scorer in SCORERS for key in keys])
        for record in records:
            fold = record["fold"]
            writer.writerow(
                [fold.name, fold.method, fold.stranger, fold.day, fold.seed, record["known_n"], record["stranger_n"]]
                + [f"{record[scorer][key]:.4f}" for scorer in SCORERS for key in keys]
            )


def summarize(folds: list[Fold], args: argparse.Namespace, out_dir: Path) -> None:
    records = []
    missing = []
    for fold in folds:
        path = out_dir / fold.name / "scores.npz"
        if not path.exists():
            missing.append(fold.name)
            continue
        with np.load(path) as data:
            records.append(fold_metrics(fold, data))
    if not records:
        print("No scored folds to summarize.")
        return

    strangers = [s for s in DEFAULT_LABELS if any(r["fold"].stranger == s for r in records)]
    days = sorted({r["fold"].day for r in records})
    seeds = sorted({r["fold"].seed for r in records})

    def values(scorer: str, key: str, chosen: list[dict[str, Any]] | None = None) -> list[float]:
        return [r[scorer][key] for r in (records if chosen is None else chosen)]

    write_results_csv(records, out_dir / "results.csv")

    far_columns = " | ".join(f"Accepted & named right at {far * 100:g}% strangers in" for far in FAR_TARGETS)
    if args.loss == "arcface":
        config = arcface_config(args)
        model_text = (
            f"ArcFace embedding ({config.embedding_dim} values, scale {config.scale:g}, margin {config.margin:g}) "
            f"plus the cross-day contrastive loss at weight {config.cross_day_weight:g} "
            f"(temperature {config.temperature:g}); batches of {config.samples_per_person} presses per person. "
            "Feature vector = the embedding"
            if config.cross_day_weight > 0 else
            f"ArcFace embedding ({config.embedding_dim} values, scale {config.scale:g}, margin {config.margin:g}), "
            f"no cross-day loss; batches of {config.samples_per_person} presses per person. "
            "Feature vector = the embedding"
        )
        batch_text = ""
    else:
        model_text = "cross-entropy classifier. Feature vector = the 32 values before the last layer"
        batch_text = f", batch size {args.batch_size}"
    lines = [
        "# Can the CNN reject a stranger?",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `evaluate_open_set.py`. Model: {model_text}. "
        f"Preprocessing: {args.method}. Training: {args.epochs} epochs, learning rate {args.learning_rate:g}"
        f"{batch_text}, seeds {', '.join(map(str, seeds))}. "
        f"{len(records)} folds ({len(strangers)} strangers × {len(days)} held-out days × {len(seeds)} seeds).",
        "",
        "Each fold leaves one person out as the stranger and holds out one day. The model trains on the "
        "other people's other days, and those same recordings are the enrollment templates. Known test = "
        "the other people's recordings from the held-out day; stranger test = every recording of the "
        "left-out person. A recording is accepted when its best score clears a threshold.",
        "",
        "Numbers are mean ± standard deviation over folds. Thresholds are picked on each fold's own test "
        "scores, so they show how separable the scores are, not what a fixed threshold would achieve on "
        "new data.",
        "",
    ]
    if missing:
        lines += [f"**Incomplete:** {len(missing)} folds not scored yet.", ""]

    lines += [
        "## 1. Overall",
        "",
        "- **Closed-set accuracy:** best-scoring person is right, no rejection (what training reports).",
        "- **AUROC:** chance that a known recording's best score beats a stranger's (0.5 = no better than a coin).",
        "- **EER (known vs stranger):** error rate where rejected known recordings = accepted strangers.",
        "- **Accepted & named right at X% strangers in:** share of known recordings let in under the right "
        "name when the threshold lets X% of the stranger's recordings in.",
        "- **Verification EER:** the person claims a name first (\"I am Jay\"); EER of own-name vs "
        "stranger-claiming-that-name scores.",
        "",
        f"| Score | Closed-set accuracy | AUROC | EER (known vs stranger) | {far_columns} | Verification EER |",
        "|---|---|---|---|" + "---|" * len(FAR_TARGETS) + "---|",
    ]
    for scorer in SCORERS:
        auroc_mean, auroc_sd = mean_sd(values(scorer, "auroc"))
        lines.append(
            f"| {SCORER_NAMES[scorer]} | {pct_sd(values(scorer, 'accuracy'))} | {auroc_mean:.3f} ± {auroc_sd:.3f} | "
            f"{pct_sd(values(scorer, 'eer'))} | "
            + " | ".join(pct_sd(values(scorer, f"dir@{far:g}")) for far in FAR_TARGETS)
            + f" | {pct_sd(values(scorer, 'verify_eer'))} |"
        )

    lines += [
        "",
        "## 2. EER (known vs stranger) by who the stranger is",
        "",
        "Mean over the held-out days. A high EER means that person is hard to tell from the enrolled people.",
        "",
        "| Stranger | Recordings | " + " | ".join(SHORT_NAMES[s] for s in SCORERS) + " |",
        "|---|---|" + "---|" * len(SCORERS),
    ]
    for stranger in strangers:
        chosen = [r for r in records if r["fold"].stranger == stranger]
        lines.append(
            f"| {stranger} | {chosen[0]['stranger_n']} | "
            + " | ".join(pct_sd(values(s, "eer", chosen)) for s in SCORERS) + " |"
        )

    lines += [
        "",
        "## 3. EER (known vs stranger) by held-out day",
        "",
        "Mean over the strangers.",
        "",
        "| Held-out day | " + " | ".join(SHORT_NAMES[s] for s in SCORERS) + " |",
        "|---|" + "---|" * len(SCORERS),
    ]
    for day in days:
        chosen = [r for r in records if r["fold"].day == day]
        lines.append(
            f"| {day[:4]}-{day[4:6]}-{day[6:]} | " + " | ".join(pct_sd(values(s, "eer", chosen)) for s in SCORERS) + " |"
        )

    paired = [r for r in records if r["softmax"]["eer_stranger_same_day"] == r["softmax"]["eer_stranger_same_day"]
              and r["softmax"]["eer_stranger_train_days"] == r["softmax"]["eer_stranger_train_days"]]
    lines += [
        "",
        "## 4. Does the recording session matter?",
        "",
        "The same EER, computed once with only the stranger's recordings from the held-out day (a session "
        "the model never saw, like the known test) and once with only their recordings from the training "
        "days (sessions the enrollment recordings come from). If the model matched session conditions "
        "rather than the person, training-day strangers would sit closer to the enrollment templates, be "
        "harder to reject, and show a higher EER. "
        f"Only folds where the stranger has recordings on both kinds of day ({len(paired)} folds).",
        "",
        "| Score | Stranger from the held-out day | Stranger from training days |",
        "|---|---|---|",
    ]
    for scorer in SCORERS:
        lines.append(
            f"| {SCORER_NAMES[scorer]} | {pct_sd(values(scorer, 'eer_stranger_same_day', paired))} | "
            f"{pct_sd(values(scorer, 'eer_stranger_train_days', paired))} |"
        )

    lines += ["", "Per-fold numbers: `results.csv`. Chart: `open_set.png`."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_open_set(records, strangers, out_dir / "open_set.png")
    print(f"Wrote {out_dir / 'results.csv'}, {out_dir / 'summary.md'} and {out_dir / 'open_set.png'}")
    if missing:
        print(f"Incomplete: {len(missing)} folds not scored yet.")


def plot_open_set(records: list[dict[str, Any]], strangers: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1.4]}, facecolor=SURFACE
    )
    far_percent = FAR_GRID * 100
    shown = far_percent <= 30
    label_y = []
    for scorer in SCORERS:
        curve = np.mean([r[scorer]["dir_curve"] for r in records], axis=0) * 100
        left.plot(far_percent[shown], curve[shown], color=SCORER_COLORS[scorer], linewidth=2,
                  label=SCORER_NAMES[scorer])
        label_y.append([curve[shown][-1], scorer])
    # Direct labels at the right end, nudged apart so they do not overlap.
    label_y.sort()
    for index in range(1, len(label_y)):
        label_y[index][0] = max(label_y[index][0], label_y[index - 1][0] + 4)
    for y, scorer in label_y:
        left.annotate(SHORT_NAMES[scorer], (30, y), textcoords="offset points", xytext=(6, -3),
                      color=INK, fontsize=9, annotation_clip=False)
    left.set_xlim(0, 30)
    left.set_ylim(0, 100)
    left.set_xlabel("Strangers let in (%)", color=INK_SECONDARY)
    left.set_ylabel("Known recordings let in under the right name (%)", color=INK_SECONDARY)
    left.set_title("Trade-off as the threshold moves (mean over folds)", color=INK, loc="left", fontsize=11)
    left.legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=8)

    offsets = {scorer: (index - (len(SCORERS) - 1) / 2) * 0.22 for index, scorer in enumerate(SCORERS)}
    for scorer in SCORERS:
        means, sds = [], []
        for stranger in strangers:
            mean, sd = mean_sd([r[scorer]["eer"] for r in records if r["fold"].stranger == stranger])
            means.append(mean * 100)
            sds.append(sd * 100)
        xs = [index + offsets[scorer] for index in range(len(strangers))]
        right.errorbar(xs, means, yerr=sds, fmt="o", markersize=8, color=SCORER_COLORS[scorer],
                       ecolor=SCORER_COLORS[scorer], elinewidth=2, capsize=0, markeredgecolor=SURFACE,
                       markeredgewidth=2, label=SHORT_NAMES[scorer], zorder=3)
    right.axhline(50, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    right.annotate("coin flip", (len(strangers) - 0.5, 50), textcoords="offset points", xytext=(0, 4),
                   ha="right", color=INK_SECONDARY, fontsize=8)
    right.set_xticks(range(len(strangers)), strangers)
    highest = max(r[s]["eer"] for r in records for s in SCORERS) * 100
    right.set_ylim(0, max(60, (highest // 10 + 2) * 10))
    right.set_xlabel("Person left out as the stranger", color=INK_SECONDARY)
    right.set_ylabel("EER, known vs stranger (%)", color=INK_SECONDARY)
    right.set_title("Equal error rate by stranger (lower is better)", color=INK, loc="left", fontsize=11)
    right.legend(frameon=False, loc="upper left", labelcolor=INK, ncols=3, fontsize=8)

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
    parser.add_argument("--output-dir", default="experiments/open_set")
    parser.add_argument("--method", choices=("resize", "onset"), default="onset")
    parser.add_argument("--strangers", nargs="+", choices=DEFAULT_LABELS, default=list(DEFAULT_LABELS),
                        help="People to leave out, one at a time. Default: everyone.")
    parser.add_argument("--days", nargs="+", help="Days to hold out (YYYYMMDD). Default: every day in the dataset.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="cross_entropy",
                        help="Model to train: the classifier or the ArcFace embedding (see train_cnn.py).")
    metric = parser.add_argument_group("metric learning (only used with --loss arcface)")
    metric.add_argument("--embedding-dim", type=int, default=DEFAULT_ARCFACE_CONFIG.embedding_dim)
    metric.add_argument("--arcface-scale", type=float, default=DEFAULT_ARCFACE_CONFIG.scale)
    metric.add_argument("--arcface-margin", type=float, default=DEFAULT_ARCFACE_CONFIG.margin)
    metric.add_argument("--cross-day-weight", type=float, default=DEFAULT_ARCFACE_CONFIG.cross_day_weight)
    metric.add_argument("--contrastive-temperature", type=float, default=DEFAULT_ARCFACE_CONFIG.temperature)
    metric.add_argument("--samples-per-person", type=int, default=DEFAULT_ARCFACE_CONFIG.samples_per_person)
    parser.add_argument("--parallel", type=int, default=3, help="Folds trained at the same time.")
    parser.add_argument("--dry-run", action="store_true", help="List the folds and exit without training.")
    parser.add_argument("--summarize-only", action="store_true", help="Skip training and scoring.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir).resolve()
    days = args.days or list_recording_days(args.dataset_root, DEFAULT_LABELS)
    folds = plan_folds(args.strangers, days, args.method, args.seeds)

    if args.dry_run:
        print(f"{len(folds)} folds -> {out_dir}")
        for fold in folds:
            state = "done" if (out_dir / fold.name / "metadata.json").exists() else "to train"
            print(f"  {fold.name:42s} {state}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        train_missing(folds, args, out_dir)
        score_missing(folds, args, out_dir)
    summarize(folds, args, out_dir)


if __name__ == "__main__":
    main()
