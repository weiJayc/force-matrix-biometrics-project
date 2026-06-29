from datetime import datetime

from force_matrix_biometrics.capture import capture_for_duration
from force_matrix_biometrics.profiles import TIMED_CAPTURE_PROFILE


def main() -> None:
    filename = datetime.now().strftime("pressData_%Y%m%d_%H%M%S.txt")
    capture_for_duration(
        profile=TIMED_CAPTURE_PROFILE,
        duration_seconds=5,
        output_path=filename,
    )


if __name__ == "__main__":
    main()