from __future__ import annotations

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from .config import SerialProfile
from .serial_io import open_serial, packet_to_grid, read_one_packet


def run_heatmap(profile: SerialProfile, title: str = "Serial Heatmap") -> None:
    ser = open_serial(profile)

    plt.ion()
    fig, ax = plt.subplots()
    image = None
    value_texts: list[list[object]] = []

    try:
        while True:
            packet = read_one_packet(ser, profile)
            if packet is None:
                continue

            grid = packet_to_grid(packet, profile)

            if image is None:
                image = ax.imshow(grid, cmap="Blues", aspect="auto", vmin=profile.vmin, vmax=profile.vmax)
                fig.colorbar(image, ax=ax)
                for row in range(profile.rows or 0):
                    row_texts = []
                    for col in range(profile.cols or 0):
                        text = ax.text(
                            col,
                            row,
                            f"{int(grid[row, col])}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=10,
                            fontweight="bold",
                        )
                        text.set_path_effects([
                            path_effects.Stroke(linewidth=2.5, foreground="black"),
                            path_effects.Normal(),
                        ])
                        row_texts.append(text)
                    value_texts.append(row_texts)
            else:
                image.set_data(grid)
                for row in range(profile.rows or 0):
                    for col in range(profile.cols or 0):
                        value_texts[row][col].set_text(f"{int(grid[row, col])}")

            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            fig.canvas.draw_idle()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        plt.ioff()
        plt.show()
