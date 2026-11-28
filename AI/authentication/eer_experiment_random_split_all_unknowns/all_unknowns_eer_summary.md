# Random Split All-Unknown EER Summary

- Average FAR/FRR EER curve: `all_unknowns_average_eer_curve.png`
- Average curve data: `all_unknowns_average_eer_curve.csv`
- EER from mean curves: 0.24398975 at threshold 0.62028175
- Mean-curve EER is computed from the pointwise average of the eight scenario FAR/FRR curves; it is not the arithmetic mean of scenario EERs.
- Combined open-set confusion matrix: `all_unknowns_open_set_confusion_matrix.png`
- Combined matrix data: `all_unknowns_open_set_confusion_matrix.csv`
- Genuine rows pool 7 scenarios (420 samples per user); Stranger rows contain 300 samples per user. Each scenario uses its own EER threshold.

- Unknown scenarios: 8
- Mean EER: 0.23983994
- EER standard deviation across Unknown identities: 0.04654296
- Lowest EER: Reyna (0.14126984126984143)
- Highest EER: Andy (0.29267399267399263)
- The standard deviation describes the distinct Unknown-user scenarios, not repeated random splits.

| Unknown | Known Users | Train | Genuine | Unknown Test | Accuracy | FAR | FRR | EER | EER Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 666 | Amber|Andy|Apple|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.933333 | 0.225000 | 0.225000 | 0.225000 | 0.610500 |
| Amber | 666|Andy|Apple|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.942857 | 0.221930 | 0.221930 | 0.221930 | 0.620242 |
| Andy | 666|Amber|Apple|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.945238 | 0.292674 | 0.292674 | 0.292674 | 0.650275 |
| Apple | 666|Amber|Andy|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.947619 | 0.220455 | 0.220455 | 0.220455 | 0.630432 |
| Jay | 666|Amber|Andy|Apple|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.942857 | 0.281944 | 0.281944 | 0.281944 | 0.650854 |
| Joanna | 666|Amber|Andy|Apple|Jay|Reyna|Tiffany | 1680 | 420 | 300 | 0.935714 | 0.284722 | 0.284722 | 0.284722 | 0.640646 |
| Reyna | 666|Amber|Andy|Apple|Jay|Joanna|Tiffany | 1680 | 420 | 300 | 0.919048 | 0.141270 | 0.141270 | 0.141270 | 0.469762 |
| Tiffany | 666|Amber|Andy|Apple|Jay|Joanna|Reyna | 1680 | 420 | 300 | 0.933333 | 0.250725 | 0.250725 | 0.250725 | 0.620261 |
