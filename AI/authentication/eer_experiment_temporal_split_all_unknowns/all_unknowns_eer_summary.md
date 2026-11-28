# Temporal Split All-Unknown EER Summary

- Average FAR/FRR EER curve: `all_unknowns_average_eer_curve.png`
- Average curve data: `all_unknowns_average_eer_curve.csv`
- EER from mean curves: 0.37356862 at threshold 0.54065559
- Mean-curve EER is computed from the pointwise average of the eight scenario FAR/FRR curves; it is not the arithmetic mean of scenario EERs.
- Combined open-set confusion matrix: `all_unknowns_open_set_confusion_matrix.png`
- Combined matrix data: `all_unknowns_open_set_confusion_matrix.csv`
- Genuine rows pool 7 scenarios (420 samples per user); Stranger rows contain 300 samples per user. Each scenario uses its own EER threshold.

- Unknown scenarios: 8
- Mean EER: 0.37582034
- EER standard deviation across Unknown identities: 0.05818810
- Lowest EER: Tiffany (0.2805555555555555)
- Highest EER: Andy (0.466666666666667)
- The standard deviation describes the distinct Unknown-user scenarios, not repeated temporal splits.

| Unknown | Known Users | Train | Genuine | Unknown Test | Accuracy | FAR | FRR | EER | EER Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 666 | Amber|Andy|Apple|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.807143 | 0.367490 | 0.367490 | 0.367490 | 0.530469 |
| Amber | 666|Andy|Apple|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.823810 | 0.319091 | 0.319091 | 0.319091 | 0.550627 |
| Andy | 666|Amber|Apple|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.792857 | 0.466667 | 0.466667 | 0.466667 | 0.590500 |
| Apple | 666|Amber|Andy|Jay|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.826190 | 0.337879 | 0.337879 | 0.337879 | 0.569303 |
| Jay | 666|Amber|Andy|Apple|Joanna|Reyna|Tiffany | 1680 | 420 | 300 | 0.797619 | 0.408805 | 0.408805 | 0.408805 | 0.560340 |
| Joanna | 666|Amber|Andy|Apple|Jay|Reyna|Tiffany | 1680 | 420 | 300 | 0.795238 | 0.437255 | 0.437255 | 0.437255 | 0.560565 |
| Reyna | 666|Amber|Andy|Apple|Jay|Joanna|Tiffany | 1680 | 420 | 300 | 0.776190 | 0.388821 | 0.388821 | 0.388821 | 0.469305 |
| Tiffany | 666|Amber|Andy|Apple|Jay|Joanna|Reyna | 1680 | 420 | 300 | 0.892857 | 0.280556 | 0.280556 | 0.280556 | 0.530139 |
