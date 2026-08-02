from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER

from .common import BenchmarkRecord, format_records_table, load_benchmark_dataset, summarize_records, DEFAULT_USERS
from .experiment_distance import run_distance_metric_experiment
from .experiment_template import run_template_strategy_experiment
from .experiment_threshold import run_threshold_strategy_experiment


def run_authentication_benchmark(
    users: Sequence[str] = DEFAULT_USERS,
    registration_count: int = 10,
    k_value: float = 2.0,
    feature_names: Sequence[str] = ENGINEERED_FEATURE_ORDER,
) -> list[BenchmarkRecord]:
    X, raw_y = load_benchmark_dataset()

    threshold_records = run_threshold_strategy_experiment(
        X,
        raw_y,
        users=users,
        registration_count=registration_count,
        k_value=k_value,
        feature_names=feature_names,
    )
    template_records = run_template_strategy_experiment(
        X,
        raw_y,
        users=users,
        registration_count=registration_count,
        k_value=k_value,
        feature_names=feature_names,
    )
    distance_records = run_distance_metric_experiment(
        X,
        raw_y,
        users=users,
        registration_count=registration_count,
        k_value=k_value,
        feature_names=feature_names,
    )

    return summarize_records([*threshold_records, *template_records, *distance_records])


def main() -> None:
    summary = run_authentication_benchmark()
    print("\n========== Authentication Benchmark ==========")
    print(format_records_table(summary))


if __name__ == "__main__":
    main()
