# GitHub Copilot Instructions

## 1. Project Overview & Core Tech Stack

**Matrix-Force Sensor Biometrics**: Captures pressure-distribution data from a matrix-force sensor over serial, visualizes it live as a heatmap, records labeled datasets to CSV, and trains classical ML classifiers (KNN, Random Forest) to identify users by their pressure pattern.

- **Language**: Python 3.x
- **Core Libraries**: `numpy`, `matplotlib`, `pyserial` (core); `pandas`, `scikit-learn` for the `AI/` training scripts
- **Hardware**: Matrix-force sensor array, 4×4 grid (16 values), serial communication at 115200 baud, COM4
- **Data Format**: 35-byte packets (2-byte header `\xAA\x01` + 32-byte payload of 16-bit little-endian pressure values, range 0-16384)

## 2. Code Style & Architectural Guidelines

### Directory & Module Organization
- **`force_matrix_biometrics/`**: Core library modules (stable, reusable)
  - `config.py`: `SerialProfile` dataclass (immutable, frozen)
  - `profiles.py`: Predefined serial configurations (`COUNT_CAPTURE_PROFILE`, `TIMED_CAPTURE_PROFILE`, `HEATMAP_PROFILE`) — add new configurations here, never mutate existing ones
  - `serial_io.py`: Low-level packet reading, header sync, and grid conversion
  - `capture.py`: Packet-oriented logging and capture orchestration
  - `recording.py`: Frame-oriented capture -> normalize -> CSV pipeline used by the desktop app
  - `visualization.py`: Matplotlib heatmap rendering loop
  - `app.py`: Tkinter GUI (`PressureMatrixApp`), backing `desktop_app.py`
  - `__init__.py`: Package docstring only
- **`desktop_app.py`**: Primary entrypoint — Tkinter GUI for labeled dataset recording, live preview, and playback. This is the actively developed interface for data collection; prefer extending `app.py`/`recording.py` over adding new root scripts.
- **`AI/`**: Offline model training scripts (no `__init__.py`, uses relative imports and `../dataset` paths — must be run with `AI/` as the working directory)

### Naming Conventions
- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`

### Key Patterns
- **Profiles**: Serial configurations are immutable dataclasses; store in `profiles.py` and import into scripts
- **Grid Reshape**: Raw packet bytes → numpy array via `packet_to_grid()` → heatmap / CSV
- **Resource Cleanup**: Always close serial port in a `finally` block (see `capture.py`)

## 3. Strict Prohibitions & Special Considerations

### Hardware Safety
- **Serial Communication**: Assume hardware may not be connected; provide mock/fallback paths or warn users

### Testing Strategy
- No pytest/unittest framework configured
- When modifying `serial_io.py` or `capture.py`, validate manually against a real device or an existing recorded dataset CSV under `dataset/`

### Preferred Edits
- Enhance library modules (`force_matrix_biometrics/`) over duplicating logic in root scripts
- Add new profiles to `profiles.py` for new sensor configurations; do not modify existing ones in place
- If adding visualization features, extend `visualization.py` or the desktop app's embedded heatmap and call from there
