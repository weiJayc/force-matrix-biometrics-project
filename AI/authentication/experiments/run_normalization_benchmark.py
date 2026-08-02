from __future__ import annotations

from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.experiments.normalization_benchmark.benchmark import main


if __name__ == "__main__":
    main()
