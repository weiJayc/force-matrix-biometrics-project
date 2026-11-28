# Temporal Split Random Forest EER Scenario - Unknown Jay

## 1. Experiment Objective

This scenario trains one Random Forest on earlier recordings from Known Users, validates on later recordings, and uses the excluded user's timestamp-valid recordings only as Unknown Test data.

## 2. Dataset Information

- Dataset CSV total: 2500
- Valid 50x16 CSV: 2500
- Enrolled users: 666, Amber, Andy, Apple, Joanna, Reyna, Tiffany
- Unknown user: Jay
- Excluded: background
- Exact duplicate file count: 0
- Exact duplicate pressure-matrix count: 0
- Files with unparseable recording time: 0

## 3. Dataset Split

- Split rule: per enrolled user, earliest 80% training / latest remainder genuine validation; tied timestamps are kept together
- Sorting key: filename's YYYYMMDD_HHMMSS timestamp; trailing identifier is ignored
- Training samples: 1680
- Genuine test samples: 420
- Unknown test samples: 300
- Excluded background samples: 100
- Split integrity: each active sample is assigned once; exact file and matrix duplicates are rejected

| User | Total | Training | Validation | Training earliest | Training latest | Validation earliest | Validation latest |
|---|---:|---:|---:|---|---|---|---|
| 666 | 300 | 240 | 60 | 2026-09-30 15:10:31 | 2026-10-30 16:14:32 | 2026-10-30 16:14:35 | 2026-10-30 16:43:16 |
| Amber | 300 | 240 | 60 | 2026-09-30 15:14:00 | 2026-10-30 16:18:37 | 2026-10-30 16:18:40 | 2026-10-30 16:34:22 |
| Andy | 300 | 240 | 60 | 2026-09-30 15:21:11 | 2026-10-30 16:16:40 | 2026-10-30 16:16:44 | 2026-10-30 16:31:36 |
| Apple | 300 | 240 | 60 | 2026-09-30 15:25:56 | 2026-11-25 15:53:32 | 2026-11-25 16:43:57 | 2026-11-25 16:47:49 |
| Joanna | 300 | 240 | 60 | 2026-09-30 15:37:56 | 2026-11-25 16:43:22 | 2026-11-25 16:48:28 | 2026-11-25 16:52:18 |
| Reyna | 300 | 240 | 60 | 2026-09-30 16:26:40 | 2026-10-30 16:12:11 | 2026-10-30 16:12:14 | 2026-10-30 16:37:29 |
| Tiffany | 300 | 240 | 60 | 2026-09-30 17:02:50 | 2026-10-30 18:42:47 | 2026-10-30 18:49:23 | 2026-10-30 18:58:54 |
| Jay | 300 | 0 | 0 |  |  |  |  |

## 4. Model Configuration

- Feature pipeline: `extract_dataset_statistics()` from the main Sequence Statistics benchmark
- Feature dimension: 67
- RandomForestClassifier parameters:
  - n_estimators: 100
  - random_state: 42
  - class_weight: None
  - n_jobs: -1
- Fit count: 1
- Training sample count: 1680
- Maximum class probability is used as confidence.

## 5. Evaluation Metrics

- GAR: genuine correct identity accepted / total genuine test samples
- FAR: accepted Jay samples / 300 Unknown Test samples
- FRR: 1 - GAR
- Genuine Rejection Rate: genuine samples with confidence below threshold
- Genuine Misidentification Rate: accepted genuine samples with wrong identity

## 6. Experimental Results

- Raw RF Accuracy: 0.79761905
- EER: 0.4088050314465409
- EER Threshold: 0.56033962
- FAR at EER: 0.40880503
- FRR at EER: 0.40880503
- GAR at EER: 0.59119497
- Genuine Rejection Rate at EER: 0.37946990
- Genuine Misidentification Rate at EER: 0.02933513
- EER calculation method: linear_interpolation
- Unknown FAR matrix: `Jay\plots\unknown_far_confusion_matrix_Jay.png`

## 7. Limitations

- This scenario uses one Unknown identity: Jay.
- Unknown sample count is not the number of distinct unknown people.
- All timestamp-valid Unknown recordings are test-only, as in the original Joanna temporal experiment.
- Threshold analysis applies thresholds to the fixed validation and Unknown predictions; it is not independent threshold calibration.
- The result applies to this fixed dataset and should not be generalized to all strangers.
- No background samples are used.

## 8. Final Conclusion

The reported values use one per-user chronological split, one fitted Random Forest, and one prediction pass for each fixed validation and Unknown set.
