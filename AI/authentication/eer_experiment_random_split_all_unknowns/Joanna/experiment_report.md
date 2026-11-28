# Random Split Random Forest EER Scenario - Unknown Joanna

## 1. Experiment Objective

This scenario trains one Random Forest on the Known Users and evaluates the excluded user's recordings only as Unknown Test data.

## 2. Dataset Information

- Dataset CSV total: 2500
- Valid 50x16 CSV: 2500
- Enrolled users: 666, Amber, Andy, Apple, Jay, Reyna, Tiffany
- Unknown user: Joanna
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
- FAR: accepted Joanna samples / 300 Unknown Test samples
- FRR: 1 - GAR
- Genuine Rejection Rate: genuine samples with confidence below threshold
- Genuine Misidentification Rate: accepted genuine samples with wrong identity

## 6. Experimental Results

- Raw RF Accuracy: 0.93571429
- EER: 0.2847222222222222
- EER Threshold: 0.64064583
- FAR at EER: 0.28472222
- FRR at EER: 0.28472222
- GAR at EER: 0.71527778
- Genuine Rejection Rate at EER: 0.27281746
- Genuine Misidentification Rate at EER: 0.01190476
- EER calculation method: linear_interpolation
- Unknown FAR matrix: `Joanna\plots\unknown_far_confusion_matrix_Joanna.png`

## 7. Limitations

- This scenario uses one Unknown identity: Joanna.
- Unknown sample count is not the number of distinct unknown people.
- Threshold analysis uses these fixed predictions to locate EER; it is not independent threshold calibration.
- The result applies to this fixed dataset and should not be generalized to all strangers.
- No background samples are used.

## 8. Final Conclusion

The reported values use one per-user random split, one fitted Random Forest, and one prediction pass for each fixed test set.
