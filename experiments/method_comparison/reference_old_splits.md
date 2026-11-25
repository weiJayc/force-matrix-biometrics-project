# Old method on the old splits (reference only)

Single runs of the old method (`--preprocess resize`) on the 2,400 recordings of the 8 people,
trained on 2026-11-26 with the learning rates and epoch counts of the earlier `cnn_` and
`cnn2000_` models (batch size 16, seed 42). The seed did not yet fix the starting weights, so
these runs cannot be reproduced exactly, and the model files were not kept.

They are not part of the comparison in `summary.md`: the random and time splits validate on
sessions the model also trained on, so they do not show how a model does on a new day.

| Split | Epochs | Learning rate | Last epoch | Mean of last 10 epochs | Amber | Jay | 666 | Andy | Apple | Joanna | Reyna | Tiffany |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 250 | 1e-3 | 98.3% | 98.1% | 95 | 98 | 100 | 100 | 100 | 97 | 100 | 97 |
| random | 250 | 2e-4 | 97.1% | 97.1% | 97 | 97 | 100 | 92 | 98 | 95 | 100 | 98 |
| random | 250 | 3e-4 | 97.7% | 97.3% | 95 | 98 | 98 | 97 | 100 | 95 | 100 | 98 |
| random | 400 | 3e-4 | 96.9% | 97.4% | 93 | 95 | 100 | 95 | 100 | 95 | 100 | 97 |
| random | 400 | 5e-4 | 98.1% | 97.6% | 93 | 98 | 100 | 100 | 100 | 97 | 100 | 97 |
| random | 400 | 7e-4 | 98.1% | 98.1% | 95 | 100 | 100 | 100 | 100 | 97 | 100 | 93 |
| time | 250 | 1e-3 | 92.5% | 92.6% | 98 | 92 | 100 | 97 | 60 | 98 | 100 | 95 |
| time | 250 | 1e-4 | 92.1% | 93.1% | 95 | 95 | 97 | 97 | 85 | 90 | 100 | 78 |
| time | 250 | 2e-4 | 92.9% | 94.6% | 98 | 97 | 100 | 97 | 78 | 97 | 100 | 77 |
| time | 250 | 3e-4 | 95.2% | 91.0% | 98 | 98 | 100 | 93 | 98 | 88 | 100 | 85 |
| time | 400 | 3e-4 | 94.6% | 94.9% | 95 | 97 | 100 | 98 | 88 | 95 | 100 | 83 |

Per-person columns are the last-epoch accuracy (%) on that person's validation recordings.

What they show: no learning rate or epoch count is clearly better. Random-split runs all land
at 97–98% and time-split runs at 92–95%, and a single run moves by up to 4 points depending
on which epoch it stops at. That is why the comparison uses one setting (250 epochs, learning
rate 2e-4) for both methods and repeats it with several seeds.
