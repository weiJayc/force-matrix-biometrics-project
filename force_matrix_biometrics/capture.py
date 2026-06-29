from __future__ import annotations

import time
from pathlib import Path

from .config import SerialProfile
from .serial_io import open_serial, packet_hex, read_one_packet


def capture_packets(
    profile: SerialProfile,
    output_path: str | Path,
    packet_limit: int,
    delay_seconds: float = 0.0,
    include_index: bool = True,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ser = open_serial(profile)
    packet_received = 0

    try:
        with output_file.open("a", encoding="utf-8") as file:
            while packet_received < packet_limit:
                packet = read_one_packet(ser, profile)
                if packet is None:
                    continue

                packet_text = packet_hex(packet)
                print(packet_text)
                if include_index:
                    file.write(f"{packet_received}: {packet_text}\n")
                else:
                    file.write(packet_text + "\n")

                packet_received += 1
                if delay_seconds:
                    time.sleep(delay_seconds)
    finally:
        ser.close()


def capture_for_duration(
    profile: SerialProfile,
    duration_seconds: float,
    output_path: str | Path,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ser = open_serial(profile)
    data_packets: list[bytes] = []
    packet_count = 0
    time_count = 0
    start = time.time()

    try:
        while time_count < duration_seconds:
            data = read_one_packet(ser, profile)
            if data is None:
                continue

            data_packets.append(data)
            packet_count += 1

            if time.time() - start >= 1:
                time_count += 1
                print(f"{packet_count} packets/s")
                packet_count = 0
                start = time.time()
    finally:
        ser.close()

    with output_file.open("w", encoding="utf-8") as file:
        for data in data_packets:
            file.write(data.hex(" ") + "\n")

    print(f"Saved to {output_file}")
