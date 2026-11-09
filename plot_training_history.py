"""Plot CNN training history (loss/accuracy curves + confusion matrix) from metadata.json."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from force_matrix_biometrics.training_plots import load_metadata, save_training_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="models/cnn",
        help="Directory containing metadata.json (default: models/cnn)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively instead of only saving them",
    )
    args = parser.parse_args()

    save_training_plots(load_metadata(args.model_dir), args.model_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
