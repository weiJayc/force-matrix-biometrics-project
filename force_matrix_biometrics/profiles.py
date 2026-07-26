from __future__ import annotations

from .config import SerialProfile

HEADER = b"\xAA\x01"

COUNT_CAPTURE_PROFILE = SerialProfile(
    port="COM4",
    baudrate=115200,
    timeout=0,
    header=HEADER,
    packet_size=35,
)

TIMED_CAPTURE_PROFILE = SerialProfile(
    port="COM4",
    baudrate=115200,
    timeout=0,
    header=HEADER,
    packet_size=35,
)

HEATMAP_PROFILE = SerialProfile(
    port="COM4",
    baudrate=115200,
    timeout=1,
    header=HEADER,
    packet_size=35,
    rows=4,
    cols=4,
    value_dtype="<u2",
    vmin=0,
    vmax=16384,
)
