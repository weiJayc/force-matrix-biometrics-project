from __future__ import annotations

from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from authentication.validation_suite_zscore_cosine_v2 import main


if __name__ == "__main__":
    main()