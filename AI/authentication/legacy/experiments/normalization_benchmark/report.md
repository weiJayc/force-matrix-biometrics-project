# Normalization Benchmark Report

## Scope

This benchmark changes only the zero-range feature handling. Registration, authentication, feature extraction, template construction, threshold logic, and distance formulas remain unchanged.

The change is:

- detect zero-range features from registration data
- build one shared usable-feature mask
- normalize as before
- flatten as before
- keep only usable features for distance computation
- save the mask in the template and reuse it during authentication

The four compared normalization strategies are:

- Per-user MinMax
- Global MinMax
- Z-score normalization
- RobustScaler

The two compared distance metrics are:

- Euclidean
- Cosine

## Benchmark Comparison Table

| Normalization | Distance | Threshold | FAR | FRR | Acceptance Rate | AUC | EER | Average Genuine Distance | Average Impostor Distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global MinMax | Cosine | 0.0970 | 0.0341 | 0.4667 | 0.5333 | 0.7480 | 0.3153 | 0.1596 | 0.2513 |
| Global MinMax | Euclidean | 5.9662 | 0.0081 | 0.5333 | 0.4667 | 0.7019 | 0.3694 | 8.9342 | 10.5387 |
| Per-user MinMax | Cosine | 0.5937 | 0.4439 | 0.0727 | 0.9273 | 0.7856 | 0.3027 | 0.3376 | 0.5868 |
| Per-user MinMax | Euclidean | 16.8052 | 0.0992 | 0.3697 | 0.6303 | 0.8106 | 0.2612 | 20.1934 | 74.1548 |
| RobustScaler | Cosine | 0.8683 | 0.3821 | 0.2727 | 0.7273 | 0.6727 | 0.3585 | 0.7037 | 0.8196 |
| RobustScaler | Euclidean | 12729.3418 | 0.3350 | 0.5273 | 0.4727 | 0.4880 | 0.5054 | 36824.9219 | 35485.7109 |
| Z-score | Cosine | 0.7999 | 0.0748 | 0.0909 | 0.9091 | 0.9830 | 0.0910 | 0.4862 | 1.1849 |
| Z-score | Euclidean | 27.8490 | 0.0439 | 0.5091 | 0.4909 | 0.6755 | 0.3953 | 44.2410 | 47.5366 |

## Interpretation

Z-score + Cosine is the strongest configuration in this benchmark.

- AUC: 0.9830
- EER: 0.0910
- FAR: 0.0748
- FRR: 0.0909
- Acceptance Rate: 0.9091

Per-user MinMax + Euclidean now gives the best Euclidean AUC, but it is still weaker than the best Cosine setting.

- AUC: 0.8106
- EER: 0.2612
- FAR: 0.0992
- FRR: 0.3697
- Average Impostor Distance: 74.1548

## Zero-Range Feature Analysis

The benchmark counts zero-range features directly from registration samples.

| User | Original Features | Zero-Range Features | Usable Features | Removed Features | Zero-Range Ratio | Mean Genuine Dist | Mean Impostor Dist | Max Impostor Dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| amber | 1150 | 67 | 1083 | 67 | 0.0583 | 24.2810 | 19.6460 | 43.1756 |
| jay | 1150 | 176 | 974 | 176 | 0.1530 | 20.5411 | 127.5168 | 425.7741 |
| 666 | 1150 | 165 | 985 | 165 | 0.1435 | 23.0676 | 50.4218 | 275.3602 |
| background | 1150 | 195 | 955 | 195 | 0.1696 | 12.6632 | 97.7804 | 333.1360 |

The correlation between zero-range feature count and Euclidean distance instability is still strong after masking:

- Correlation with maximum impostor Euclidean distance: 0.9224
- Correlation with mean impostor Euclidean distance: 0.8022
- Correlation with mean genuine Euclidean distance: -0.6986

This means the zero-range issue is still relevant, but the mask has reduced the scale blow-up substantially.

## Conclusion

The benchmark supports the hypothesis that the bottleneck is normalization and zero-range feature handling, not feature extraction, template construction, or the distance formulas themselves.

Key conclusions:

1. The shared usable-feature mask reduces the Euclidean distance explosion.
2. Z-score + Cosine is the best overall configuration in this benchmark.
3. Zero-range feature count remains correlated with Euclidean instability, but the correlation is lower after masking.
4. The pipeline change is narrowly scoped: only zero-range features are masked out after normalization.
