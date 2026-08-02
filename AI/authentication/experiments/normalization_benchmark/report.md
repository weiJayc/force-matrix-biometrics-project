# Normalization Benchmark Report

## Scope

This benchmark changes only the normalization strategy. Registration, authentication, feature extraction, template construction, and distance calculation remain unchanged.

The four compared strategies are:

- Per-user MinMax
- Global MinMax
- Z-score normalization
- RobustScaler

The two compared distance metrics are:

- Euclidean
- Cosine

ROC plots:

- [Euclidean](normalization_roc_euclidean.png)
- [Cosine](normalization_roc_cosine.png)

## Benchmark Comparison Table

| Normalization | Distance | Threshold | FAR | FRR | Acceptance Rate | AUC | EER | Average Genuine Distance | Average Impostor Distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global MinMax | Cosine | 0.0957 | 0.0309 | 0.5030 | 0.4970 | 0.6921 | 0.3694 | 0.2723 | 0.3436 |
| Global MinMax | Euclidean | 5.9663 | 0.0081 | 0.5818 | 0.4182 | 0.6641 | 0.3953 | 2036.7104 | 1643.1997 |
| Per-user MinMax | Cosine | 0.9971 | 0.4829 | 0.1030 | 0.8970 | 0.7156 | 0.3451 | 0.6805 | 0.8288 |
| Per-user MinMax | Euclidean | 4470.8589 | 0.4813 | 0.1030 | 0.8970 | 0.7173 | 0.3467 | 3507.3640 | 34471.7148 |
| RobustScaler | Cosine | 0.8683 | 0.3057 | 0.3030 | 0.6970 | 0.7163 | 0.3044 | 0.7122 | 0.8535 |
| RobustScaler | Euclidean | 16573.4648 | 0.1772 | 0.5212 | 0.4788 | 0.6061 | 0.4729 | 37569.4062 | 52421.5078 |
| Z-score | Cosine | 0.9997 | 0.2585 | 0.0000 | 1.0000 | 0.9594 | 0.1451 | 0.5394 | 1.1591 |
| Z-score | Euclidean | 27.8491 | 0.0423 | 0.5576 | 0.4424 | 0.6451 | 0.4188 | 2062.5090 | 1676.7804 |

## Interpretation

### Best performing setting by AUC

Cosine + Z-score is the strongest configuration in this benchmark.

- AUC: 0.9594
- EER: 0.1451
- FAR: 0.2585
- FRR: 0.0000
- Acceptance Rate: 1.0000

### Best Euclidean setting

Within the Euclidean comparisons, Per-user MinMax gives the highest AUC.

- AUC: 0.7173
- EER: 0.3467
- FAR: 0.4813
- FRR: 0.1030

However, its distance scale is extremely unstable, which is a warning sign rather than a practical advantage.

### Best tradeoff overall

If the goal is to reduce verification error without changing the model, Z-score is the most promising normalization for Cosine distance.

## Zero-Range Feature Analysis

The per-user registration split shows many zero-range dimensions in the flattened feature space.

| User | Zero-Range Features | Zero-Range Ratio | Mean Genuine Dist | Mean Impostor Dist | Max Impostor Dist |
|---|---:|---:|---:|---:|---:|
| amber | 67 | 0.0583 | 5254.6260 | 3787.0737 | 83148.7891 |
| jay | 176 | 0.1530 | 2013.4983 | 39228.6250 | 221889.3594 |
| 666 | 165 | 0.1435 | 4838.7827 | 27104.6387 | 221821.5781 |
| background | 195 | 0.1696 | 1790.7371 | 67185.7031 | 267392.6875 |

### Direct relationship with Euclidean explosion

The correlation between zero-range feature count and Euclidean impostor explosion is very strong in this benchmark:

- Correlation with maximum impostor Euclidean distance: 0.9946
- Correlation with mean impostor Euclidean distance: 0.8908
- Correlation with mean genuine Euclidean distance: -0.7695

This is strong evidence that the Euclidean scale blow-up is directly linked to the per-user MinMax normalization pathology on low-variance registration subsets.

### Why this happens

Per-user MinMax uses only the user's registration samples to compute min/max. When one or more flattened features have range $0$, the normalization code replaces that range with $1$.

That keeps the calculation numerically valid, but it also leaves those dimensions uncompressed. If later genuine or impostor samples vary strongly in those dimensions, the Euclidean norm becomes dominated by those raw-like coordinates.

The result is a large distance scale, especially for users with many zero-range features.

## Conclusion

The benchmark supports the hypothesis that the current bottleneck is normalization rather than feature extraction, template construction, or distance formula.

Key conclusions:

1. Per-user MinMax is not the most stable normalization strategy.
2. Z-score + Cosine gives the best overall verification performance in this benchmark.
3. Zero-range features are strongly correlated with Euclidean distance explosion.
4. The pipeline itself remains unchanged; only the normalization layer was varied.
