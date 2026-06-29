from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SerialProfile:
    port: str
    baudrate: int
    timeout: float
    header: bytes
    packet_size: int
    rows: int | None = None
    cols: int | None = None
    value_dtype: str = "<u1"
    vmin: int = 0
    vmax: int = 255

    @property
    def payload_size(self) -> int:
        return self.packet_size - len(self.header) - 1
