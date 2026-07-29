# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A biometrics system that captures pressure-distribution data from a matrix-force sensor over serial, visualizes it live as a heatmap, records labeled datasets to CSV, and trains classical ML classifiers (KNN, Random Forest) to identify users by their pressure pattern.

**Note:** `README.md` describes an older 5×7, `0xAB 0xAA`-header, COM3 configuration. The active configuration (see `force_matrix_biometrics/profiles.py`) is a **4×4 grid (16 values), header `\xAA\x01`, COM4, 16-bit values (`<u2`, range 0-16384)**. Trust `profiles.py` and the AI scripts (which hardcode `value_i` for `i in range(16)` and frame shape `(50, 16)`) over the README when they conflict.

## Environment

- Windows-only project (PowerShell scripts, `tkinter` desktop app, COM ports).
- Two virtualenvs exist in the repo root: `.venv` and `venv`. `build_exe.ps1` looks for `.venv` first, falling back to `venv`.
- Install deps: `pip install -r requirements.txt` (core: `numpy`, `matplotlib`, `pyserial`; also pulls in `pandas`/`scikit-learn` indirectly for the `AI/` scripts — verify they're installed before running training scripts).

## Common commands

```powershell
# Real-time heatmap viewer (legacy entrypoint, uses HEATMAP_PROFILE)
python heatmap_serial_sample.py

# Count-based packet capture to press_img.txt (legacy entrypoint)
python catch.py

# Timed packet capture (legacy entrypoint)
python timeCatch.py

# Desktop GUI: labeled dataset recording + live preview + playback
python desktop_app.py

# Build a standalone Windows executable (wraps desktop_app.py via PyInstaller)
.\build_exe.ps1
# Output: dist\PressureMatrixCollector.exe (override name: .\build_exe.ps1 -Name Foo)
```

AI training scripts run from inside `AI/` and expect the dataset one level up:

```powershell
cd AI
python train_knn.py              # KNN train/test split + accuracy, confusion matrix, classification report
python train_random_forest.py    # Same pipeline with RandomForestClassifier
python cross_validation_knn.py   # StratifiedKFold CV sweep over k in [1,3,5,7,9,11]
```

There is no test suite, linter, or CI configured. There is no packaging/build step for the `AI/` or `force_matrix_biometrics/` code beyond the PyInstaller desktop build.

## Architecture

### `force_matrix_biometrics/` — core library

All serial/capture/visualization logic lives here; root scripts are thin entrypoints that import from it. Do not duplicate logic into root scripts — extend the library instead.

- **`config.py`** — `SerialProfile`, a frozen dataclass describing a serial link + packet layout (port, baudrate, timeout, header bytes, packet size, optional grid `rows`/`cols`, `value_dtype`, `vmin`/`vmax` for display scaling).
- **`profiles.py`** — the concrete `SerialProfile` instances used by each entrypoint (`COUNT_CAPTURE_PROFILE`, `TIMED_CAPTURE_PROFILE`, `HEATMAP_PROFILE`). Add new configurations here rather than constructing `SerialProfile` inline; don't mutate the existing ones (they're frozen).
- **`serial_io.py`** — lowest layer: `open_serial`, `read_one_packet` (byte-by-byte header sync + fixed-size payload read), `packet_to_grid` (payload → numpy array reshaped to `rows × cols`), `packet_hex`.
- **`capture.py`** — packet-oriented logging: `capture_packets` (fixed count, optional delay, appends hex lines to a text file) and `capture_for_duration` (time-boxed capture with a packets/sec printout, writes hex lines on completion). Always closes the serial port in a `finally` block — follow this pattern for any new capture function.
- **`recording.py`** — frame-oriented (not raw-packet) capture pipeline used by the desktop app: `capture_frames_by_count` → `normalize_frames` (pads/truncates every recording to the same frame count with zero-frames) → `save_recording_csv` (one row per frame, columns `label, frame_index, value_0..N`) → `capture_and_save_recording` orchestrates all three and returns a `RecordingResult`. Also provides `load_recording_csv`/`discover_recording_csv_files` for reading datasets back (handles both the current `value_*` CSV format and an older `hex_packet` column format for backward compatibility with earlier recordings). `sanitize_label` restricts labels to `[A-Za-z0-9_-]`.
- **`visualization.py`** — `run_heatmap`: a blocking matplotlib `ion()` loop that reads packets and redraws a heatmap with per-cell value labels. Used by the legacy `heatmap_serial_sample.py` entrypoint only (the desktop app has its own embedded heatmap).
- **`app.py`** — `PressureMatrixApp`, a Tkinter GUI (entrypoint: `main()`, invoked by `desktop_app.py`). Combines: a background thread continuously polling the serial port for a live heatmap preview (`_live_preview_worker`, guarded by `_live_stop_event`/locks so it can be paused during an actual capture), a capture workflow that calls `recording.capture_and_save_recording` on a worker thread and streams frame/progress callbacks back to the Tk main thread via `self.after(0, ...)`, a dataset browser (lists CSVs under a chosen dataset root) and a CSV recording playback feature (`_start_preview_playback`, drives the same heatmap widget from a stored recording instead of the live serial feed). Live preview and capture/playback are mutually exclusive — capture and playback both call `_stop_live_preview` before starting and resume it afterward.

### `AI/` — offline model training (not part of the installable package; no `__init__.py`, uses relative imports and relative `../dataset` paths, so scripts must be run with `AI/` as the CWD)

- **`data_loader.py`** — `load_dataset(data_dir)` walks a dataset directory for `*.csv`, loads each with `load_csv` (expects exactly `value_0..value_15` columns from a 4×4 sensor and the `label` column), and skips (with a printed warning) any file whose frame shape isn't `(50, 16)` — i.e. every recording is expected to be exactly 50 frames from `capture_and_save_recording`/the desktop app's "Fixed frames" field.
- **`preprocess.py`** — `preprocess_dataset`: label-encodes the raw label strings, does a stratified train/test split, then min-max normalizes each of the 16 sensor channels independently using **train-set statistics only** (applied to both train and test) to avoid leakage.
- **`train_knn.py`** / **`train_random_forest.py`** — near-identical pipelines: load → preprocess → flatten each `(50, 16)` sample to a single feature vector → fit classifier → report accuracy/confusion matrix/classification report. When editing one, check whether the same change applies to the other (they've historically been kept in sync).
- **`cross_validation_knn.py`** — standalone k-sweep over `KNeighborsClassifier` using `StratifiedKFold(5)`, doesn't use `preprocess.py`'s train/test split (evaluates on the whole dataset via CV instead).
- **`prepare_dataset.py`** — ad hoc inspection script (prints unique values per sensor channel for one hardcoded CSV path); not part of the training pipeline.

### Data format

- Each recording CSV: header `label, frame_index, value_0, ..., value_15`; one row per frame; exactly 50 frames per file (shorter/longer captures are normalized by zero-padding/truncation in `recording.normalize_frames` before saving).
- Recordings are organized as `dataset/<label>/<timestamp>.csv` (see `recording.build_recording_path`), where `<label>` is the person/class name entered in the desktop app.
- `dataset/`, `test_dataset/`, and `dist/dataset/` all hold recordings in this format; `AI/` scripts point at `../dataset` by default.

### Legacy vs. current entrypoints

`catch.py`, `heatmap_serial_sample.py`, and `timeCatch.py` are older single-purpose scripts kept for backward compatibility; `desktop_app.py` (backed by `force_matrix_biometrics/app.py`) is the actively developed, primary interface for data collection going forward. Prefer extending `app.py`/`recording.py` for new capture features rather than the legacy scripts.
