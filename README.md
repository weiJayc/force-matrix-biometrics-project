# Matrix-Force Sensor Biometrics Project

A biometrics system that captures pressure-distribution data from a matrix-force sensor over serial, visualizes it live as a heatmap, records labeled datasets to CSV, and trains classical ML classifiers (KNN, Random Forest) to identify users by their pressure pattern.

## Overview

This project interfaces with a matrix-force sensor array to capture biometric data. The sensor is configured as a 4×4 matrix (16 data points) that detects pressure distribution across a surface, used to identify individual users by the pressure pattern they produce.

## Features

- **Serial Communication**: Reads data from the sensor via a serial connection
- **Real-time Visualization**: Live heatmap preview using matplotlib, embedded in the desktop app
- **Labeled Dataset Recording**: Captures fixed-length, per-user recordings and saves them as CSV
- **Dataset Playback**: Replays a saved recording through the same heatmap widget
- **Classifier Training**: KNN and Random Forest pipelines for user identification (`AI/`)

## Hardware Requirements

- Matrix-Force Sensor Array (4×4 grid configuration, 16 pressure points)
- Serial connection (COM4 at 115200 baud rate)
- PC/Computer with USB-to-serial converter (if needed)

## Software Requirements

- Python 3.x
- Core dependencies listed in `requirements.txt`:
   - `numpy` - Numerical operations and data reshaping
   - `matplotlib` - Heatmap visualization
   - `pyserial` - Serial communication with the sensor
- `pandas` and `scikit-learn` are also required for the `AI/` training scripts (pulled in indirectly — verify they're installed before running training scripts)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Serial Connection:**
   - Ensure the sensor is connected to COM4
   - Verify baud rate is 115200
   - Change the port in `force_matrix_biometrics/profiles.py` if using a different COM port

## Project Structure

```
.
├── desktop_app.py                 # Primary entrypoint: Tkinter GUI for dataset recording + live preview + playback
├── build_exe.ps1                  # Builds a standalone Windows executable (PyInstaller) from desktop_app.py
├── force_matrix_biometrics/       # Core library: serial I/O, capture, recording, visualization, GUI
│   ├── config.py                  # SerialProfile dataclass (frozen)
│   ├── profiles.py                # Concrete serial profiles (COUNT_CAPTURE_PROFILE, TIMED_CAPTURE_PROFILE, HEATMAP_PROFILE)
│   ├── serial_io.py                # Packet reading + header sync + grid decoding
│   ├── capture.py                  # Packet-oriented capture/logging
│   ├── recording.py                # Frame-oriented capture -> normalize -> CSV pipeline
│   ├── visualization.py             # Matplotlib heatmap loop
│   └── app.py                      # Tkinter GUI (PressureMatrixApp)
├── AI/                              # Offline model training (run with AI/ as the working directory)
│   ├── data_loader.py               # Loads dataset/*.csv into (50, 16) arrays
│   ├── preprocess.py                # Label encoding, train/test split, per-channel normalization
│   ├── train_knn.py
│   ├── train_random_forest.py
│   └── cross_validation_knn.py
├── dataset/                         # Recorded training datasets: dataset/<label>/<timestamp>.csv
├── test_dataset/                    # Additional recorded datasets used for experimentation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Usage

### Desktop app (primary entrypoint)

```bash
python desktop_app.py
```

This is the actively developed interface for data collection: it shows a live heatmap preview, records a fixed number of frames per label, saves recordings as CSV, and can browse/play back existing recordings.

### Build a standalone EXE

```powershell
.\build_exe.ps1
```

The built executable will appear in `dist\PressureMatrixCollector.exe` by default (override with `.\build_exe.ps1 -Name Foo`). It wraps `desktop_app.py` via PyInstaller.

### Train classifiers

```powershell
cd AI
python train_knn.py              # KNN train/test split + accuracy, confusion matrix, classification report
python train_random_forest.py    # Same pipeline with RandomForestClassifier
python cross_validation_knn.py   # StratifiedKFold CV sweep over k in [1,3,5,7,9,11]
```

## Data Format

### Packet Structure
- **Header**: 2 bytes (`\xAA\x01`)
- **Payload**: 32 bytes (16 values × 2 bytes each)
- **Total Packet Size**: 35 bytes
- **Value Type**: 16-bit unsigned little-endian (`<u2`), range 0–16384

### Sensor Grid
```
[0][1][2][3]
[4][5][6][7]
[8][9][10][11]
[12][13][14][15]
```

Each value represents pressure intensity at that sensor point.

### Recording CSV Format
Each recording CSV has the header `label, frame_index, value_0, ..., value_15`, with one row per frame and exactly 50 frames per file (shorter/longer captures are zero-padded/truncated before saving).

## Configuration

### Serial Port Settings
Edit the profile definitions in `force_matrix_biometrics/profiles.py` if you need to change the serial port, baud rate, timeout, or packet layout. Profiles are frozen dataclasses — add a new profile rather than mutating an existing one.

## Troubleshooting

### No data received
- Verify sensor is connected to correct COM port (COM4 by default)
- Check baud rate matches sensor configuration (115200)
- Ensure sensor is powered on
- Verify USB-to-serial driver is installed (if using converter)

### Serial connection errors
- List available COM ports: `python -m serial.tools.list_ports`
- Update the port in `force_matrix_biometrics/profiles.py` to the correct port
- Ensure no other application is using the COM port

### Heatmap not displaying
- Ensure `matplotlib` is installed correctly
- Try running in an IDE with display support
- Check for any serial read errors in console output

## Future Enhancements

- Add biometric feature extraction beyond raw pressure frames
- Expand classifier coverage beyond KNN/Random Forest
- Support multiple sensors or sensor arrays
- Implement touch/pressure event detection

## License

[Add your license here]

## Author

[Your name/team]

## Contact

[Your contact information]
