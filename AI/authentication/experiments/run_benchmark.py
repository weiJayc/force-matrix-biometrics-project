from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Sequence

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.feature_extractor import ENGINEERED_FEATURE_ORDER

from authentication.experiments.benchmark import run_authentication_benchmark
from authentication.experiments.common import DEFAULT_USERS, format_records_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the authentication benchmark suite.")
    parser.add_argument(
        "--users",
        nargs="*",
        default=list(DEFAULT_USERS),
        help="User IDs to include in the benchmark. Defaults to the full baseline set.",
    )
    parser.add_argument(
        "--registration-count",
        type=int,
        default=10,
        help="Number of samples to use for registration per user.",
    )
    parser.add_argument(
        "--k-value",
        type=float,
        default=2.0,
        help="k value used by the Mean + k × Std threshold strategy.",
    )
    return parser.parse_args()


def run(users: Sequence[str], registration_count: int, k_value: float) -> None:
    summary = run_authentication_benchmark(
        users=users,
        registration_count=registration_count,
        k_value=k_value,
        feature_names=ENGINEERED_FEATURE_ORDER,
    )
    print("\n========== Authentication Benchmark ==========")
    print(format_records_table(summary))


def main() -> None:
    args = parse_args()
    run(args.users, args.registration_count, args.k_value)


if __name__ == "__main__":
    main()
