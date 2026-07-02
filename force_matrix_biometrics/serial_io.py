from __future__ import annotations

import numpy as np
import serial

from .config import SerialProfile


def open_serial(profile: SerialProfile) -> serial.Serial:
    return serial.Serial(profile.port, profile.baudrate, timeout=profile.timeout)


def read_one_packet(ser: serial.Serial, profile: SerialProfile) -> bytes | None:
    """Read one complete packet and return the raw bytes, including header."""
    while True:
        first = ser.read(1)
        if not first:
            return None

        if first != profile.header[:1]:
            continue

        second = ser.read(1)
        if not second:
            return None

        if second != profile.header[1:]:
            continue

        remaining = ser.read(profile.packet_size - len(profile.header))
        if len(remaining) != profile.packet_size - len(profile.header):
            return None

        return profile.header + remaining


def packet_to_grid(packet: bytes, profile: SerialProfile) -> np.ndarray:
    if profile.rows is None or profile.cols is None:
        raise ValueError("Grid shape is not configured for this profile")

    payload = packet[len(profile.header) : -1]
    values = np.frombuffer(payload, dtype=profile.value_dtype)

    expected_size = profile.rows * profile.cols
    if values.size != expected_size:
        raise ValueError(f"Expected {expected_size} values, got {values.size}")

    return values.reshape(profile.rows, profile.cols)


def packet_hex(packet: bytes) -> str:
    return packet.hex(" ")
