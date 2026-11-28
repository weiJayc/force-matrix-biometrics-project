# Random Split Random Forest EER Scenario - Unknown 666

## 1. Experiment Objective

This scenario trains one Random Forest on the Known Users and evaluates the excluded user's recordings only as Unknown Test data.

## 2. Dataset Information

- Dataset CSV total: 2500
- Valid 50x16 CSV: 2500
- Enrolled users: Amber, Andy, Apple, Jay, Joanna, Reyna, Tiffany
- Unknown user: 666
- Excluded: background
- Exact duplicate file count: 0
- Exact duplicate pressure-matrix count: 0

## 3. Dataset Split

- Split rule: each enrolled user, 80% training / 20% genuine test
- Fixed random seed: 42
- Training samples: 1680
- Genuine test samples: 420
- Unknown test samples: 300
- Excluded background samples: 100

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
- FAR: accepted 666 samples / 300 Unknown Test samples
- FRR: 1 - GAR
- Genuine Rejection Rate: genuine samples with confidence below threshold
- Genuine Misidentification Rate: accepted genuine samples with wrong identity

## 6. Experimental Results

- Raw RF Accuracy: 0.93333333
- EER: 0.22499999999999948
- EER Threshold: 0.61050000
- FAR at EER: 0.22500000
- FRR at EER: 0.22500000
- GAR at EER: 0.77500000
- Genuine Rejection Rate at EER: 0.20476190
- Genuine Misidentification Rate at EER: 0.02023810
- EER calculation method: linear_interpolation
- Unknown FAR matrix: `666\plots\unknown_far_confusion_matrix_666.png`

## 7. Limitations

- This scenario uses one Unknown identity: 666.
- Unknown sample count is not the number of distinct unknown people.
- Threshold analysis uses these fixed predictions to locate EER; it is not independent threshold calibration.
- The result applies to this fixed dataset and should not be generalized to all strangers.
- No background samples are used.

## 8. Final Conclusion

The reported values use one per-user random split, one fitted Random Forest, and one prediction pass for each fixed test set.
