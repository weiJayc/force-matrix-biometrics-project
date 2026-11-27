# Fixed Dataset Random Forest GAR / FAR / FRR / EER Experiment

## 1. Experiment Objective

This experiment evaluates one fixed Random Forest model using fixed enrolled-user and unknown-user test datasets.

## 2. Dataset Information

- Dataset CSV total: 2500
- Valid 50x16 CSV: 2500
- Enrolled users: Amber, Jay, 666, Andy, Apple, Reyna, Tiffany
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
- FAR: accepted Joanna samples / 100 Joanna test samples
- FRR: 1 - GAR
- Genuine Rejection Rate: genuine samples with confidence below threshold
- Genuine Misidentification Rate: accepted genuine samples with wrong identity

## 6. Experimental Results

- Raw RF Accuracy: 0.92380952
- EER: 0.25964912280701746
- EER Threshold: 0.64052632
- FAR at EER: 0.25964912
- FRR at EER: 0.25964912
- GAR at EER: 0.74035088
- Genuine Rejection Rate at EER: 0.25137845
- Genuine Misidentification Rate at EER: 0.00827068
- EER calculation method: linear_interpolation

## 7. Limitations

- Unknown consists of one user: Joanna.
- Joanna's 100 recordings are 100 samples, not 100 distinct individuals.
- The single fixed result does not estimate variability across random splits.
- Threshold analysis uses fixed test predictions and is not independent calibration.
- The result applies to this fixed dataset and should not be generalized to all strangers.
- No background samples are used.

## 8. Final Conclusion

The reported values are based only on the fixed split, one fitted Random Forest, and one prediction pass for each fixed test set.
