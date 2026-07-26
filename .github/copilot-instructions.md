# GitHub Copilot Instructions

## 1. Project Overview & Core Tech Stack

**Matrix-Force Sensor Biometrics**: Real-time pressure distribution capture and heatmap visualization for biometric analysis.

- **Language**: Python 3.x
- **Core Libraries**: `numpy`, `matplotlib`, `pyserial`
- **Hardware**: Matrix-force sensor array (5×7 grid, default; 4×4 configurable), serial communication at 115200 baud
- **Data Format**: 35-byte packets (2-byte header `0xAB 0xAA` + 33-byte payload of 8-bit pressure values)

## 2. Code Style & Architectural Guidelines

### Directory & Module Organization
- **`force_matrix_biometrics/`**: Core library modules (stable, reusable)
  - `config.py`: `SerialProfile` dataclass (immutable, frozen)
  - `serial_io.py`: Low-level packet reading and grid conversion
  - `capture.py`: Packet logging and capture orchestration
  - `visualization.py`: Matplotlib heatmap rendering
  - `profiles.py`: Predefined serial configurations
  - `__init__.py`: Package docstring only
- **Root scripts**: Entrypoints (`catch.py`, `heatmap_serial_sample.py`, `timeCatch.py`) import from package

### Naming Conventions
- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`

### Key Patterns
- **Profiles**: Serial configurations are immutable dataclasses; store in `profiles.py` and import into scripts
- **Grid Reshape**: Raw packet bytes → numpy array via `packet_to_grid()` → matplotlib
- **Resource Cleanup**: Always close serial port in `finally` block (see `capture.py`)

## 3. Strict Prohibitions & Special Considerations

### Hardware Safety
- **Serial Communication**: Assume hardware may not be connected; provide mock/fallback paths or warn users
- **Sample Data**: Use existing `press_img.txt`, `pressData_*.txt` files for testing without hardware

### Testing Strategy
- No pytest/unittest framework; use manual validation with sample data
- When modifying `serial_io.py` or `capture.py`, test with a sample packet file if hardware unavailable

### Preferred Edits
- Enhance library modules (`force_matrix_biometrics/`) over duplicating logic in root scripts
- Add new profiles to `profiles.py` for new sensor configurations; do not modify existing ones in place
- If adding visualization features, extend `visualization.py` and call from scripts