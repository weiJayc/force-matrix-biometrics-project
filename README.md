# Matrix-Force Sensor Biometrics Project

A biometrics system that captures and analyzes pressure distribution patterns using a matrix-force sensor. This project reads sensor data via serial communication and visualizes it as heatmaps for biometric analysis.

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
