# Matrix-Force Sensor Biometrics Project

Identifies people by the way they press on a 4×4 pressure sensor. A desktop app records labeled presses from the sensor, and a CNN learns to tell the people apart. The `AI/` folder holds simpler KNN and random-forest baselines, plus the authentication experiments.

## Features

- **Live heatmap**: shows the 16 sensor values in real time.
- **Labeled recording**: records a fixed number of frames per press and saves them as a CSV under `dataset/<label>/`.
- **Playback**: replays any saved recording on the same heatmap.
- **CNN classifier**: trains and evaluates a 3D CNN on the recordings.
- **Baselines**: KNN and random-forest classifiers on the same data.
- **Standalone app**: packages the desktop app as a Windows `.exe`.

## Hardware

- A 4×4 matrix-force sensor (16 pressure points).
- A serial connection at 115200 baud, through a USB-to-serial adapter if needed. The default port is COM4, and you can pick another one in the app.

## Installation

Developed on Windows with Python 3.14.

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt       # desktop app and AI/ baselines: numpy, matplotlib, pyserial, pandas, scikit-learn
pip install -r requirements-cnn.txt   # adds PyTorch for the CNN
```

## Project Structure

```
.
├── desktop_app.py               # starts the data-collection app
├── curve_viewer.py              # starts the recording curve viewer
├── build_exe.ps1                # packages the app into a Windows .exe
├── train_cnn.py                 # trains the CNN
├── check_cnn_predictions.py     # re-evaluates a trained CNN
├── plot_training_history.py     # plots a trained CNN's training curves
├── compare_methods.py           # trains and compares the old and new preprocessing
├── evaluate_open_set.py         # tests whether the CNN can reject an unseen person
├── evaluate_multi_press.py      # same test, deciding from several presses at once
├── compare_open_set.py          # compares models on the open-set test
├── force_matrix_biometrics/
│   ├── app.py                   # Tkinter app: live heatmap, recording, playback
│   ├── curve_viewer.py          # Tkinter app: plots the pressure curves of saved recordings
│   ├── recording.py             # records frames, saves and loads recording CSVs
│   ├── serial_io.py             # reads and decodes sensor packets
│   ├── config.py                # SerialProfile: serial and packet settings
│   ├── profiles.py              # the sensor's settings (HEATMAP_PROFILE)
│   ├── cnn.py                   # CNN data loading, preprocessing, model and training
│   ├── metric_learning.py       # ArcFace embedding and cross-day contrastive loss
│   └── training_plots.py        # draws training curves and the confusion matrix
├── AI/                          # KNN and random-forest baselines; authentication/ experiments
├── dataset/                     # recordings: <label>/<timestamp>.csv
├── models/                      # trained CNN models, one folder per experiment
├── experiments/                 # output of compare_methods.py and evaluate_open_set.py
├── requirements.txt             # desktop app dependencies
└── requirements-cnn.txt         # adds PyTorch for the CNN
```

## Collecting Data

Start the app from the project root:

```powershell
python desktop_app.py
```

1. Choose the sensor's port under **COM Port** and click **Connect**. Click **Refresh COM** if the port isn't listed. The heatmap starts updating once the port is open.
2. Enter the person's name under **Class label**, for example `Amber`. Characters other than letters, digits, `_` and `-` are replaced with `_`.
3. Set **Fixed frames** to `50`. The app defaults to 64, but every recording in `dataset/` has 50 frames, and the `AI/` scripts only accept 50.
4. Leave **Dataset root** at the `dataset` folder in the project, or click **Browse** to save somewhere else.
5. Click **Start Capture** and press the sensor. Once all frames are in, the app saves the recording to `<dataset root>/<label>/<timestamp>.csv`.

To replay a recording, select it under **Saved recordings** and click **Preview selected**. Click **Refresh** to rescan the folder.

To train the CNN on a new person, the label must exactly match a name in `DEFAULT_LABELS` (see [CNN Training and Evaluation](#cnn-training-and-evaluation)).

### View recording curves

`curve_viewer.py` plots the pressure curves of saved recordings, for checking or comparing data without the sensor:

```powershell
python curve_viewer.py                                   # browse dataset/
python curve_viewer.py --dataset-root D:\path\to\dataset  # browse another folder
```

- The left panel lists recordings grouped by folder. The filter box matches folder or file names, for example `Jay` or `20261029`.
- Selecting a recording plots it immediately. Ctrl+click or Shift+click selects several to compare. **Pin selected** keeps recordings on the plot while you browse others.
- Options: overlay the curves or draw one plot each; plot the total of all 16 cells or each cell; subtract the baseline (mean of the first 5 frames); color by label; share the Y axis.
- Values are shown as raw / 256.

### Build a standalone EXE

```powershell
.\build_exe.ps1                    # creates dist\PressureMatrixCollector.exe
.\build_exe.ps1 -Name MyCollector  # creates dist\MyCollector.exe
```

The script needs a virtual environment named `.venv` or `venv` in the project root with `requirements.txt` installed, and installs PyInstaller into it if it's missing. `build/` and `dist/` are ignored by git. The exe saves recordings to a `dataset` folder inside the folder it is started from.

## CNN Training and Evaluation

These scripts train and check the CNN. Run all of them from the project root:

- `train_cnn.py` trains the model.
- `plot_training_history.py` plots the training curves.
- `check_cnn_predictions.py` re-checks a trained model's predictions.
- `compare_methods.py` trains both preprocessing methods under the same conditions and compares them.
- `evaluate_open_set.py` tests whether scores from the model can reject a person it never saw. `evaluate_multi_press.py` repeats that test deciding from several presses, and `compare_open_set.py` compares models on it.

`plot_training_history.py` only needs `matplotlib` and `numpy`. The others also need PyTorch from `requirements-cnn.txt`.

A typical run:

```powershell
python train_cnn.py --summary                                          # check that every label has recordings
python train_cnn.py --output-dir models/cnn_time                       # train, save the model and its plots
python train_cnn.py --preprocess onset --output-dir models/cnn_onset   # train on baseline-subtracted press windows
python check_cnn_predictions.py --model-dir models/cnn_time --verbose  # re-check predictions one recording at a time
```

`models/` already holds the team's trained models. Each folder is named after its settings. For example, `cnn_random_epoch400_lr3e-4` used the random split, 400 epochs and a learning rate of 3e-4. Folders starting with `cnn2000_` were trained on the 2,150 recordings made up to 2026-10-30, and the other `cnn_` folders on the 900 recordings from 2026-09-30. None of them saw the 2026-11-25 session. To re-check an existing model with `check_cnn_predictions.py`, point `--dataset-root` at a copy of `dataset/` that holds only the recordings it was trained on.

### `train_cnn.py`

Trains the CNN on the dataset and saves these files to the output folder:

- `pressure_cnn.pt`: the model weights.
- `metadata.json`: the labels, split settings, seed, epochs, learning rate, batch size, per-epoch loss and accuracy, and the final validation confusion matrix. For `--preprocess onset` it also holds the preprocessing settings, the fitted tail correction and counts of the special cases (see [Onset preprocessing](#onset-preprocessing)).
- `training_curves.png` and `confusion_matrix.png`: the same plots that `plot_training_history.py` makes, drawn automatically when training finishes.

It runs on a CUDA GPU if one is available, then on Apple MPS, and otherwise on the CPU.

| Argument | Default | Description |
|---|---|---|
| `--dataset-root` | `dataset` | Folder with one sub-folder of CSV recordings per label. |
| `--output-dir` | `models/cnn` | Where the model and metadata are saved. Files already in the folder are overwritten, so use a new folder for each experiment. |
| `--preprocess` | `resize` | How recordings become model input. `resize`: the original pipeline, which resamples the whole recording. `onset`: subtract the baseline and cut a fixed window at the press onset; see [Onset preprocessing](#onset-preprocessing). |
| `--target-frames` | `50` | `resize` only. Each recording is resampled to this many frames by linear interpolation. |
| `--validation-ratio` | `0.2` | Share of each label's recordings held out for validation. Must be at least 0 and less than 1. |
| `--split-strategy` | `time` | `time`: for each label, the most recent recordings (by the timestamp in the filename) go to validation. `random`: each label's recordings are shuffled with `--seed` before splitting. `day`: every recording from `--validation-day` goes to validation and every other day to training, so the model is tested on a session it never saw. With `random` and `time`, validation shares recording sessions with training, which lets a model score well on session cues such as the sensor baseline; `day` does not. |
| `--validation-day` | none | Required with `--split-strategy day`: the day to hold out, as `20261029` or `2026-10-29`, matched against the date at the start of each file name. `--validation-ratio` is ignored. |
| `--labels` | every name in `DEFAULT_LABELS` | People to train on. Leave someone out to keep them unseen, as `evaluate_open_set.py` does for its strangers. |
| `--loss` | `cross_entropy` | `cross_entropy`: the classifier. `arcface`: a metric-learning embedding for rejecting strangers; see [Metric learning](#metric-learning---loss-arcface). |
| `--epochs` | `40` | Number of training epochs. |
| `--batch-size` | `16` | Recordings per training batch. |
| `--learning-rate` | `0.001` | Adam learning rate. |
| `--seed` | `42` | Fixes the `random` split, the baseline jitter, the starting weights, dropout and the batch order, so a run can be repeated. GPU kernels can still make results differ very slightly. Models saved before this change only used it for the split. |
| `--summary` | off | Print the number of recordings per label and exit without training. |

The classes come from the `DEFAULT_LABELS` tuple in `force_matrix_biometrics/cnn.py`, not from whichever folders exist. Every label in the tuple needs a folder under `--dataset-root` with at least one CSV, otherwise the script stops with an error. To add or remove a person, edit that tuple. `background` is not in it, so neither pipeline reads `dataset/background`: whether someone is pressing is decided by the onset trigger, not by the CNN. Models trained before this change have 9 classes including `background`; `check_cnn_predictions.py` still rebuilds them with the labels saved in their `metadata.json`.

```powershell
python train_cnn.py --output-dir models/cnn_random --split-strategy random --seed 7
python train_cnn.py --output-dir models/cnn_e100 --epochs 100 --learning-rate 0.0005
```

#### Onset preprocessing

Every recording starts at a static sensor baseline that follows the recording session rather than the person, and the `resize` models partly use it to tell people apart. `--preprocess onset` removes the baseline and gives the model only the press, the way a live system that triggers when someone grips the handle would see it. Thresholds are in raw / 256 units, because the values in the CSV are multiples of 256.

1. **Find the onset.** For each frame `i`, the baseline is the mean 16-cell total of frames `i-8` to `i-4`. The 3 frames just before `i` are skipped so that a slow rise does not pull the baseline up. Frame `i` is the onset when both it and the next frame exceed the baseline by more than `--onset-threshold`. Recordings where this never happens are skipped.
2. **Subtract the baseline.** Each cell's average over those same 5 frames is subtracted from every frame. Negative values are kept.
3. **Cut the window.** The model gets `--window-frames` frames starting `--pre-onset-frames` before the onset. Frames past the end of the recording are zero.
4. **Jitter the baseline (training only).** Each time a training recording is drawn, its baseline is scaled by a random factor between `1 - j` and `1 + j`, where `j` is `--baseline-jitter`, so the model does not depend on the baseline being exact. Validation uses the unscaled baseline.

**Early onsets.** An onset at frame 8, the earliest frame the rule can test, means the press may have started inside frames 0–4. The window then starts at frame 0. Frames 0–4 are still the baseline unless their total is more than 30 above a fallback estimate: the mean of the last 5 frames plus a per-cell correction. The correction is needed because the signal dips below the baseline after release and has not recovered when the recording ends. It is the per-cell median gap between the real baseline and the last 5 frames, fitted on the training split only and saved in `metadata.json`.

| Argument | Default | Description |
|---|---|---|
| `--window-frames` | `28` | Frames given to the model. |
| `--pre-onset-frames` | `4` | How many frames before the onset the window starts. |
| `--onset-threshold` | `60` | Rise of the 16-cell total above the baseline that counts as a press, in raw / 256 units. |
| `--baseline-jitter` | `0.1` | Training only: range of the random baseline scale. `0` turns it off. |

The other settings (5 baseline frames, 3 skipped frames, 5 tail frames, the tolerance of 30) are in `OnsetConfig` in `force_matrix_biometrics/cnn.py`. With the defaults on the current dataset, the trigger fires on all 2,400 person recordings. 287 of them are early onsets, and 35 of those use the fallback baseline. A 28-frame window holds the whole press in about 99% of recordings.

```powershell
python train_cnn.py --preprocess onset --output-dir models/cnn_onset_w20 --window-frames 20
```

#### Metric learning (`--loss arcface`)

The classifier always names one of the people it was trained on. `--loss arcface` instead trains an embedding in which a person's presses sit close together and away from everyone else, so a distance threshold can reject someone the model never saw (see [`evaluate_open_set.py`](#evaluate_open_setpy)). The code is in `force_matrix_biometrics/metric_learning.py`.

- **Model.** Same convolutional layers as the classifier, then a unit-length embedding (`--embedding-dim` values). Each training person has a learned direction.
- **ArcFace loss.** A press is scored by the cosine to each person's direction, times `--arcface-scale`. During training, the true person's angle is widened by `--arcface-margin` radians, so a press only scores well when it sits clearly closer to its own direction than to anyone else's.
- **Cross-day contrastive loss**, added with weight `--cross-day-weight`. For each press, presses by the same person on a *different* day are pulled closer, and presses by other people are pushed away. Presses by the same person on the same day are left out, so matching session conditions earns nothing. `0` trains with ArcFace alone.
- **Batches.** Each batch draws `--samples-per-person` presses from every training person, spread over that person's recording days, so every batch has cross-day partners. `--batch-size` is not used for training.

`forward()` still returns one score per person, so the model works with `check_cnn_predictions.py`. `metadata.json` records `loss` and the `arcface` settings.

| Argument | Default | Description |
|---|---|---|
| `--embedding-dim` | `64` | Length of the embedding. |
| `--arcface-scale` | `16` | Logit scale. |
| `--arcface-margin` | `0.3` | Angular margin in radians. |
| `--cross-day-weight` | `1` | Weight of the cross-day contrastive loss. `0` turns it off. |
| `--contrastive-temperature` | `0.1` | Temperature of the contrastive loss. |
| `--samples-per-person` | `8` | Presses per person in each batch. |

```powershell
python train_cnn.py --preprocess onset --loss arcface --learning-rate 0.001 --epochs 250 --output-dir models/arcface_onset
```

### `plot_training_history.py`

Reads `metadata.json` from a model folder and saves two plots into the same folder. `train_cnn.py` already does this at the end of every run, so use this script to redraw the plots for an existing model:

- `training_curves.png`: loss and accuracy per epoch, train vs. validation.
- `confusion_matrix.png`: validation results after the last epoch.

| Argument | Default | Description |
|---|---|---|
| `model_dir` (positional) | `models/cnn` | Model folder containing `metadata.json`. |
| `--show` | off | Also open the plots in a window after saving them. |

```powershell
python plot_training_history.py models/cnn_random_epoch400_lr3e-4 --show
```

### `check_cnn_predictions.py`

Loads a trained model and rebuilds the exact train/validation split it was trained on, using the split and preprocessing settings stored in `metadata.json`. For `onset` models it reuses the saved tail correction instead of fitting a new one. Models trained before `--preprocess` existed have no preprocessing settings and are rebuilt with `resize`. It predicts every recording one at a time on the CPU with the model in eval mode. For each split it prints accuracy, loss and a confusion matrix, then compares them with the last epoch recorded during training. Use it to confirm that the numbers in `metadata.json` hold up.

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | required | Model folder containing `pressure_cnn.pt` and `metadata.json`. |
| `--dataset-root` | `dataset` | Dataset to rebuild the split from. Use the same data the model was trained on. |
| `--seed` | read from `metadata.json` (`42` if missing) | Override the split seed. Only matters for models trained with `--split-strategy random`. |
| `--verbose` | off | Print each recording's actual label, predicted label and class probabilities, marked `OK` or `BAD`. |

If the dataset has changed since training, the rebuilt split won't match, and the script prints a `WARNING` with both sets of counts. Expect the train accuracy to differ slightly from the recorded value, because during training it is measured with dropout on and the weights still changing.

```powershell
python check_cnn_predictions.py --model-dir models/cnn_onset --verbose
```

### `compare_methods.py`

Answers one question: does the onset preprocessing recognize people better than resize on a day the model has never seen? Both methods use the same people, settings and seeds, so the preprocessing is the only difference.

- **Held-out day (main comparison).** Each recording day is held out once (`--split-strategy day`), so every recording is validated by a model that never saw its day. This is the closest to a door handle meeting a new day.
- **Random split (supporting check).** The same runs on a random split, where validation shares sessions with training. The drop from the random split to the held-out day shows how much a method relies on session cues such as the sensor baseline. Compare each method's drop, not random-split scores across methods.

Each combination runs once per seed, because a single run can move by several points. With the defaults that is 2 methods × (4 days + random split) × 3 seeds = 30 runs, about 25 minutes on a laptop GPU with 3 at a time.

| Argument | Default | Description |
|---|---|---|
| `--output-dir` | `experiments/method_comparison` | One folder per run, plus the summary files. |
| `--days` | every day in the dataset | Days to hold out, as `YYYYMMDD`. |
| `--seeds` | `1 2 3` | One run per seed for every combination. |
| `--epochs` / `--learning-rate` / `--batch-size` | `250` / `0.0002` / `16` | Used for every run. |
| `--methods` | `resize onset` | Which preprocessing methods to train. |
| `--parallel` | `3` | Runs trained at the same time. |
| `--dry-run` | off | List the runs and their `train_cnn.py` commands without training. |
| `--summarize-only` | off | Skip training and rebuild the summary from finished runs. |

Finished runs are skipped, so running the script again resumes after an interruption or retries failed runs. A finished run trained with other settings stops the script instead of being mixed in; use another `--output-dir`.

The output folder gets:

- `<method>_day<YYYYMMDD>_seed<n>/` and `<method>_random_seed<n>/`: the usual `train_cnn.py` output for each run, plus `train.log`.
- `results.csv`: one row per run with its accuracy after the last epoch, its mean over the last 10 epochs, and each person's accuracy.
- `summary.md`: four tables. Per held-out day, all days pooled, the random-split-to-held-out-day drop, and per person.
- `comparison.png`: held-out-day accuracy per day for both methods, and each method's drop.

```powershell
python compare_methods.py --dry-run          # list the 30 runs, train nothing
python compare_methods.py                    # train missing runs, then write the summary
python compare_methods.py --summarize-only   # rewrite the summary from finished runs
```

### `evaluate_open_set.py`

Answers: can the CNN reject someone it was never trained on? The CNN itself always names one of the people it knows, so this script tests scores computed from a trained model, accepting a recording only when its best score clears a threshold.

Each fold leaves one person out as the stranger (`--labels` with the other seven) and holds out one day (`--split-strategy day`):

- **Enrollment:** the other people's recordings from the training days. The model trains on them, and they also form each person's templates.
- **Known test:** the other people's recordings from the held-out day.
- **Stranger test:** every recording of the left-out person.

Three scores are compared, all higher-is-more-alike:

- `softmax`: the class probability.
- `centroid`: cosine similarity of the model's feature vector to each person's mean enrollment vector.
- `knn`: mean cosine similarity to that person's 5 nearest enrollment vectors.

The feature vector is the 32 values before the last layer for the classifier, and the embedding for `--loss arcface`. Vectors are centered on the mean enrollment vector, then scaled to length 1.

With the defaults that is 8 strangers × 4 days × 1 seed = 32 trainings, about 25 minutes on a laptop GPU with 3 at a time. Arguments are the same as `compare_methods.py`, except `--method` (one preprocessing, default `onset`), `--strangers` (default everyone) and `--seeds` (default `1`). `--loss` and the metric-learning arguments of `train_cnn.py` choose the model; give each model its own `--output-dir`. Finished trainings and scores are skipped, so running it again resumes. A finished fold trained with other settings stops the script.

The output folder (default `experiments/open_set`) gets:

- One folder per fold: the usual `train_cnn.py` output, `train.log`, and `scores.npz` with every score.
- `results.csv`: one row per fold.
- `summary.md`, with four tables:
  - closed-set accuracy, AUROC, equal error rate between known recordings and strangers, the share of known recordings accepted under the right name when 1% or 5% of strangers get in, and the verification EER;
  - the EER by stranger;
  - the EER by held-out day;
  - the EER for strangers recorded on the held-out day versus on training days, which shows whether the model matches sessions rather than people.
- `open_set.png`: two charts, the accept-rate trade-off and the EER by stranger.

Thresholds are chosen on each fold's own test scores, so the summary shows how separable the scores are, not the error rate a fixed threshold would give on new data.

```powershell
python evaluate_open_set.py --dry-run                      # list the 32 folds
python evaluate_open_set.py                                # classifier: train, score, summarize
python evaluate_open_set.py --strangers Andy --days 20261029 --epochs 50   # one quick fold
# metric learning, ArcFace alone and with the cross-day loss
python evaluate_open_set.py --loss arcface --cross-day-weight 0 --learning-rate 0.001 --output-dir experiments/open_set_arcface
python evaluate_open_set.py --loss arcface --cross-day-weight 1 --learning-rate 0.001 --output-dir experiments/open_set_arcface_crossday
```

#### `evaluate_multi_press.py`

Reuses the saved `scores.npz` files, so nothing is retrained. It checks whether deciding from several presses rejects strangers better than deciding from one.

An attempt is n presses in a row by the same person on the same day, in recording order. Their per-person scores are averaged, and the attempt is accepted or rejected like a single press. Every run of n consecutive presses counts as an attempt, so attempts overlap.

It writes `multi_press.md` and `multi_press.png` into the results folder, with the same metrics as `summary.md` for each n. Score files written before file names were saved are re-scored by running `evaluate_open_set.py` again.

```powershell
python evaluate_multi_press.py                    # n = 1 2 3 5
python evaluate_multi_press.py --presses 1 3 5 10
```

#### `compare_open_set.py`

Puts several `evaluate_open_set.py` result folders side by side.

- It uses only the folds every model was trained on, so differences are paired: the same stranger, held-out day and seed.
- It reports each model's metrics for 1 and 5 presses.
- It reports each model's difference from the first model, with the number of folds where it did better.
- It reports accuracy and EER per held-out day.

It writes `experiments/open_set_comparison.md` and `.png`.

```powershell
python compare_open_set.py                    # classifier vs ArcFace vs ArcFace + cross-day
python compare_open_set.py --runs "Classifier=experiments/open_set" "ArcFace=experiments/open_set_arcface"
```

## AI Baselines

The `AI/` scripts train KNN and random-forest classifiers on the same recordings, as a baseline for the CNN. Run them from inside `AI/`, because `cross_validation_knn.py` reads `../dataset` relative to the current folder:

```powershell
cd AI
python train_knn.py             # KNN: accuracy, confusion matrix and classification report
python train_random_forest.py   # the same with a random forest
python cross_validation_knn.py  # 5-fold cross-validation for k = 1, 3, 5, 7, 9, 11
```

- Every recording must have exactly 50 frames. Other files are skipped with a message.
- The label comes from the CSV's `label` column, not from the folder name.
- `prepare_dataset.py` is an early exploration script, not part of the pipeline.

Authentication experiments (accepting a registered user and rejecting everyone else) live in `AI/authentication/`; see `AI/authentication/readme.md`.

## Data Format

### Sensor packet

Each packet is 35 bytes:

| Bytes | Content |
|---|---|
| 0–1 | Header `AA 01` |
| 2–33 | 16 pressure values, each an unsigned 16-bit little-endian integer |
| 34 | Trailing byte, ignored |

The heatmap's color scale runs from 0 to 16384.

### Sensor grid

The 16 values are numbered row by row, and saved as `value_0` to `value_15` in the CSV:

```
[ 0][ 1][ 2][ 3]
[ 4][ 5][ 6][ 7]
[ 8][ 9][10][11]
[12][13][14][15]
```

### Recording CSV

Each file is one recording, with one row per frame and the header `label,frame_index,value_0,...,value_15`. Files are saved as `dataset/<label>/<YYYYmmdd_HHMMSS_ffffff>.csv`.

The current `dataset/` has 9 labels and 2,500 recordings of 50 frames each, recorded in four sessions. Each person has 300 recordings:

| Label | 2026-09-30 | 2026-10-29 | 2026-10-30 | 2026-11-25 | Total |
|---|---|---|---|---|---|
| `666`, `Amber`, `Andy`, `Reyna`, `Tiffany` | 100 | 100 | 100 | – | 300 each |
| `Apple` | 100 | 100 | – | 100 | 300 |
| `Jay` | 100 | 50 | 100 | 50 | 300 |
| `Joanna` | 100 | – | – | 200 | 300 |
| `background` | 100 | – | – | – | 100 |

`background` holds recordings with nobody touching the sensor. The CNN does not use them; the `AI/` baselines and authentication experiments do.

## Configuration

The serial settings live in `HEATMAP_PROFILE` in `force_matrix_biometrics/profiles.py`: port, baud rate, read timeout, packet header and size, grid shape, value type, and the heatmap's color range. The app starts with this profile and only changes the port at runtime. For a sensor with a different packet layout, add a new `SerialProfile` to `profiles.py` and point `app.py` at it.

## Troubleshooting

### The heatmap doesn't update

- Read the status line in the app:
  - "Live preview waiting for COMx: …" means the port can't be opened. Check that the sensor is plugged in and powered, and that no other program, such as a serial monitor, has the port open.
  - "Live preview data error: …" means packets arrive but don't match the layout set in `profiles.py`.
- Make sure the right port is selected: click **Refresh COM**, pick the port, then click **Connect**. To list ports from the command line, run `python -m serial.tools.list_ports`.
- If you use a USB-to-serial adapter, check that its driver is installed.

### Start Capture never finishes

The app waits until it has received every frame, with no timeout. If the sensor stops sending, the capture never ends. Close the app, fix the connection, and check that the live heatmap is updating before you start again.

## Future Enhancements

- Add biometric feature extraction (fingerprint minutiae, pressure patterns)
- Support multiple sensors or sensor arrays
- Implement touch/pressure event detection

## License

[Add your license here]

## Author

[Your name/team]

## Contact

[Your contact information]
