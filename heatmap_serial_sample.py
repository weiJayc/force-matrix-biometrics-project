import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

PORT = "COM3"
BAUDRATE = 115200
TIMEOUT = 1
HEADER = b"\xAA\x01"
PAYLOAD_SIZE = 32
PACKET_SIZE = len(HEADER) + PAYLOAD_SIZE + 1
ROWS = 4
COLS = 4
VMIN = 0
VMAX = 16384


def read_one_packet(ser):
    """Read one complete packet and return the raw bytes, including header."""
    ser.reset_input_buffer()
    while True:
        first = ser.read(1)
        if not first:
            return None

        if first != HEADER[:1]:
            continue

        second = ser.read(1)
        if not second:
            return None

        if second != HEADER[1:]:
            continue

        payload = ser.read(PAYLOAD_SIZE)
        if len(payload) != PAYLOAD_SIZE:
            return None

        checksum = ser.read(1)
        if len(checksum) != 1:
            return None
        print(HEADER + payload + checksum)

        return HEADER + payload + checksum


def packet_to_grid(packet):
    payload = packet[len(HEADER):-1]
    values = np.frombuffer(payload, dtype="<u2")

    if values.size != ROWS * COLS:
        raise ValueError(f"Expected {ROWS * COLS} values, got {values.size}")

    return values.reshape(ROWS, COLS)


def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
    ser.reset_input_buffer()

    plt.ion()
    fig, ax = plt.subplots()
    image = None
    colorbar = None
    value_texts = []

    try:
        while True:
            packet = read_one_packet(ser)
            if packet is None:
                continue

            grid = packet_to_grid(packet)

            if image is None:
                image = ax.imshow(grid, cmap="Blues", aspect="auto", vmin=VMIN, vmax=VMAX)
                colorbar = fig.colorbar(image, ax=ax)
                for row in range(ROWS):
                    row_texts = []
                    for col in range(COLS):
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

                for row in range(ROWS):
                    for col in range(COLS):
                        value_texts[row][col].set_text(f"{int(grid[row, col])}")

            ax.set_title("Serial Heatmap")
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


if __name__ == "__main__":
    main()
