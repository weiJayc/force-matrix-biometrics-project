import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = "COM3"
BAUDRATE = 115200
TIMEOUT = 1
HEADER = b"\xAA\x01"
PAYLOAD_SIZE = 32
PACKET_SIZE = len(HEADER) + PAYLOAD_SIZE + 1
ROWS = 4
COLS = 4


def read_one_packet(ser):
    """Read one complete packet and return the raw bytes, including header."""
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

    try:
        while True:
            packet = read_one_packet(ser)
            if packet is None:
                continue

            grid = packet_to_grid(packet)
            vmin = float(grid.min())
            vmax = float(grid.max())
            if vmin == vmax:
                vmax = vmin + 1.0

            if image is None:
                image = ax.imshow(grid, cmap="inferno", aspect="auto", vmin=vmin, vmax=vmax)
                colorbar = fig.colorbar(image, ax=ax)
            else:
                image.set_data(grid)
                image.set_clim(vmin, vmax)
                if colorbar is not None:
                    colorbar.update_normal(image)

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
