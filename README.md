# Matrix-Force Sensor Biometrics Project

A biometrics system that captures and analyzes pressure distribution patterns using a matrix-force sensor. This project reads sensor data via serial communication and visualizes it as heatmaps for biometric analysis.

## Project Status (temporary)

> Temporary section, last checked 2026-10-01 on branch `feature/cnn`. Remove it once the cleanup below is done. Where it disagrees with the rest of this README, trust this section.

### Active

- `desktop_app.py` + `force_matrix_biometrics/` (`app.py`, `recording.py`, `serial_io.py`, `config.py`, `profiles.py`): the data-collection desktop app. It is the only capture entrypoint in use.
- `force_matrix_biometrics/cnn.py`, `train_cnn.py`, `check_cnn_predictions.py`, `plot_training_history.py`: CNN training and evaluation, developed on this branch. See [CNN Training and Evaluation](#cnn-training-and-evaluation) for how to run them.
- `dataset/`: current training data, recorded 2026-09-30. It has 9 labels (`666`, `Amber`, `Andy`, `Apple`, `Jay`, `Joanna`, `Reyna`, `Tiffany`, `background`), with 100 recordings per label and 50 frames per recording.
- `models/`: output of `train_cnn.py`. All earlier models were deleted on 2026-10-01 so training can start over on the new `dataset/`.
- `build_exe.ps1`: still the way to package the desktop app.

### Active on another branch

- `AI/`: the copy on this branch is the July 2026 KNN / random-forest feasibility check. Development continues on `feature/authentication` (`AI/authentication/`, contact gate, sequence-statistics benchmarks), so don't build on the copy here.
  - `AI/prepare_dataset.py` is the earlier all-in-one version of `data_loader.py` + `preprocess.py`.
  - `AI/model.py` is a scratch script that reads a file no longer in `dataset/`. It has already been deleted on `feature/authentication`.

### No longer used

- `catch.py`, `timeCatch.py`, `heatmap_serial_sample.py`: early capture scripts, replaced by `desktop_app.py`. `force_matrix_biometrics/capture.py`, `force_matrix_biometrics/visualization.py` and the `COUNT_CAPTURE_PROFILE` / `TIMED_CAPTURE_PROFILE` profiles exist only for these scripts.
- `press_img.txt`, `pressData_20260625_*.txt`: raw packet logs from June 2026, written by those scripts.
- `test_dataset/`: the July 2026 recordings, moved here when `dataset/` was re-recorded. No pipeline reads it by default. `jay_test`, `lin_test` and `wu_test` use the old `hex_packet` CSV format, and `background/` is an exact copy of `background_test1/`.

### Outdated

- `build/`, `dist/`: PyInstaller output from 2026-06-30 (`PressureMatrixCollectorTest.exe`). It predates the July app changes (fixed frame count, live preview), so rebuild it before use. `dist/dataset/class_a/` holds two stray test recordings.
- `TODO.txt`: the June 2026 to-do list for the data collector. Every item except mmWave capture is now done in the desktop app.
- Most of this README below this section (Overview, Hardware Requirements, Project Structure, Usage §1–2, Data Format, Configuration): it describes the old 5×7 grid with 8-bit values, `0xAB 0xAA` header and COM3, plus the legacy scripts. The current sensor is a 4×4 grid of 16-bit values with header `AA 01` on COM4 (see `force_matrix_biometrics/profiles.py`).
- `.github/copilot-instructions.md`: has the same outdated sensor description, and its module list is missing `app.py`, `recording.py` and `cnn.py`.

The legacy scripts, their logs and `dist/dataset/` are already deleted on the unmerged `file-clean-up` branch, which also rewrites this README for the current sensor.

## Overview

This project interfaces with a pressure/force sensor array to capture biometric data. The sensor is configured as a 5×7 matrix (35 data points) that detects pressure distribution across a surface, which can be used for fingerprint recognition, hand geometry analysis, or other pressure-based biometric applications.

## Features

- **Serial Communication**: Reads data from a matrix-force sensor via RS-232 serial connection
- **Real-time Visualization**: Displays sensor data as heatmaps using matplotlib
- **Data Logging**: Saves raw sensor packets to files for analysis and calibration
- **Packet Processing**: Robust packet parsing with header verification (0xAB 0xAA protocol)

## Hardware Requirements

- Matrix-Force Sensor Array (5×7 grid configuration)
- Serial connection (COM3 at 115200 baud rate)
- PC/Computer with USB-to-serial converter (if needed)

## Software Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`:
   - `numpy` - Numerical operations and data reshaping
   - `matplotlib` - Real-time heatmap visualization
   - `pyserial` - Serial communication with the sensor

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Serial Connection:**
   - Ensure the sensor is connected to COM3
   - Verify baud rate is 115200
   - (Modify `PORT` variable in the scripts if using a different COM port)

## Project Structure

```
.
├── catch.py                     # Legacy entrypoint for count-based capture
├── heatmap_serial_sample.py      # Legacy entrypoint for the heatmap viewer
├── timeCatch.py                  # Legacy entrypoint for timed capture
├── force_matrix_biometrics/      # Shared package for serial + visualization logic
├── press_img.txt                 # Log file containing captured sensor packets
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

### Build a standalone EXE

Run the PowerShell build script to generate a single-file Windows executable:

```powershell
.uild_exe.ps1
```

The built executable will appear in `dist\PressureMatrixCollector.exe` by default. The script uses the Tkinter desktop launcher in `desktop_app.py`.

### 1. **Real-time Heatmap Visualization**

Run `heatmap_serial_sample.py` to continuously capture and display sensor data as a heatmap:

```bash
python heatmap_serial_sample.py
```

This script will:
- Connect to the sensor on COM3 at 115200 baud
- Read sensor packets in real-time
- Display each 5×7 grid as a color heatmap
- Update continuously as new data arrives

### 2. **Capture and Log Sensor Data**

Run `catch.py` to record sensor packets to `press_img.txt`:

```bash
python catch.py
```

This script will:
- Read 17 sensor packets (configurable)
- Save each packet as hexadecimal data to `press_img.txt`
- Wait 3 seconds between captures (configurable)

## CNN Training and Evaluation

Three scripts train and check the CNN. Run all of them from the project root:

- `train_cnn.py` trains the model.
- `plot_training_history.py` plots the training curves.
- `check_cnn_predictions.py` re-checks a trained model's predictions.

Install PyTorch first:

```powershell
pip install -r requirements-cnn.txt
```

`plot_training_history.py` doesn't need PyTorch, only `matplotlib` and `numpy`.

A typical run:

```powershell
python train_cnn.py --summary                                          # check that every label has recordings
python train_cnn.py --output-dir models/cnn_time                       # train and save the model
python plot_training_history.py models/cnn_time                        # save the loss/accuracy and confusion-matrix plots
python check_cnn_predictions.py --model-dir models/cnn_time --verbose  # re-check predictions one recording at a time
```

### `train_cnn.py`

Trains the CNN on the dataset and saves two files to the output folder:

- `pressure_cnn.pt`: the model weights.
- `metadata.json`: the labels, split settings, seed, per-epoch loss and accuracy, and the final validation confusion matrix.

It runs on a CUDA GPU if one is available, then on Apple MPS, and otherwise on the CPU.

| Argument | Default | Description |
|---|---|---|
| `--dataset-root` | `dataset` | Folder with one sub-folder of CSV recordings per label. |
| `--output-dir` | `models/cnn` | Where the model and metadata are saved. Files already in the folder are overwritten, so use a new folder for each experiment. |
| `--target-frames` | `50` | Each recording is resampled to this many frames by linear interpolation. |
| `--validation-ratio` | `0.2` | Share of each label's recordings held out for validation. Must be at least 0 and less than 1. |
| `--split-strategy` | `time` | `time`: for each label, the most recent recordings (by the timestamp in the filename) go to validation, so the model is tested on later sessions. `random`: each label's recordings are shuffled with `--seed` before splitting. |
| `--epochs` | `40` | Number of training epochs. |
| `--batch-size` | `16` | Recordings per training batch. |
| `--learning-rate` | `0.001` | Adam learning rate. |
| `--seed` | `42` | Seed for the `random` split. It does not seed weight initialization or batch order, so two runs with the same settings still give slightly different results. |
| `--summary` | off | Print the number of recordings per label and exit without training. |

The classes come from the `DEFAULT_LABELS` tuple in `force_matrix_biometrics/cnn.py`, not from whichever folders exist. Every label in the tuple needs a folder under `--dataset-root` with at least one CSV, otherwise the script stops with an error. To add or remove a person, edit that tuple.

```powershell
python train_cnn.py --output-dir models/cnn_random --split-strategy random --seed 7
python train_cnn.py --output-dir models/cnn_e100 --epochs 100 --learning-rate 0.0005
```

### `plot_training_history.py`

Reads `metadata.json` from a model folder and saves two plots into the same folder:

- `training_curves.png`: loss and accuracy per epoch, train vs. validation.
- `confusion_matrix.png`: validation results after the last epoch.

| Argument | Default | Description |
|---|---|---|
| `model_dir` (positional) | `models/cnn` | Model folder containing `metadata.json`. |
| `--show` | off | Also open the plots in a window after saving them. |

```powershell
python plot_training_history.py models/cnn_random --show
```

### `check_cnn_predictions.py`

Loads a trained model and rebuilds the exact train/validation split it was trained on, using the settings stored in `metadata.json`. It predicts every recording one at a time on the CPU with the model in eval mode. For each split it prints accuracy, loss and a confusion matrix, then compares them with the last epoch recorded during training. Use it to confirm that the numbers in `metadata.json` hold up.

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `models/cnn_train10` | Model folder containing `pressure_cnn.pt` and `metadata.json`. The default folder no longer exists, so always pass this argument. |
| `--dataset-root` | `dataset` | Dataset to rebuild the split from. Use the same data the model was trained on. |
| `--seed` | read from `metadata.json` (`42` if missing) | Override the split seed. Only matters for models trained with `--split-strategy random`. |
| `--verbose` | off | Print each recording's actual label, predicted label and class probabilities, marked `OK` or `BAD`. |

If the dataset has changed since training, the rebuilt split won't match, and the script prints a `WARNING` with both sets of counts. Expect the train accuracy to differ slightly from the recorded value, because during training it is measured with dropout on and the weights still changing.

```powershell
python check_cnn_predictions.py --model-dir models/cnn_random --verbose
```

## Data Format

### Packet Structure
- **Header**: 2 bytes (`0xAB 0xAA`)
- **Payload**: 33 bytes (35 byte total packet size)
- **Data Layout**: 5 rows × 7 columns of 8-bit pressure values
- **Total Packet Size**: 35 bytes

### Sensor Grid
```
[0][1][2][3][4][5][6]
[7][8]...
...
[28][29][30][31][32][33][34]
```

Each value represents pressure intensity (0-255) at that sensor point.

## Configuration

### Serial Port Settings
Edit the profile definitions in `force_matrix_biometrics/profiles.py` if you need to change the serial port, baud rate, timeout, or packet layout.

### Sensor Grid Dimensions
```python
ROWS = 5                # Number of rows in sensor matrix
COLS = 7                # Number of columns in sensor matrix
```

### Data Capture
In `catch.py`, adjust the `packet_limit` and `delay_seconds` values passed to `capture_packets`.

## Reorganized Layout

The project is now organized around a shared package instead of duplicated script logic:

- `force_matrix_biometrics/serial_io.py` handles packet scanning and decoding
- `force_matrix_biometrics/capture.py` handles packet logging and timed capture
- `force_matrix_biometrics/visualization.py` handles the heatmap display
- `force_matrix_biometrics/profiles.py` stores the active serial layouts for each script

The original root scripts remain as entrypoints so existing commands keep working.

For the desktop app and packaging workflow, use:

```powershell
python desktop_app.py
```

## Troubleshooting

### No data received
- Verify sensor is connected to correct COM port
- Check baud rate matches sensor configuration (115200)
- Ensure sensor is powered on
- Verify USB-to-serial driver is installed (if using converter)

### Serial connection errors
- List available COM ports: `python -m serial.tools.list_ports`
- Update `PORT` variable to the correct port
- Ensure no other application is using the COM port

### Heatmap not displaying
- Ensure `matplotlib` is installed correctly
- Try running in an IDE with display support
- Check for any serial read errors in console output

## Future Enhancements

- Add biometric feature extraction (fingerprint minutiae, pressure patterns)
- Implement machine learning classification for biometric matching
- Add GUI for real-time monitoring and configuration
- Support multiple sensors or sensor arrays
- Add data preprocessing and normalization
- Implement touch/pressure event detection

## License

[Add your license here]

## Author

[Your name/team]

## Contact

[Your contact information]
