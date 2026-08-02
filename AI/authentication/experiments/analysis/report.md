# Authentication Benchmark Analysis Report

## Scope

This report analyzes the current benchmark outputs without changing any algorithmic code. The benchmark keeps the same dataset, feature pipeline, registration flow, authentication flow, and per-user template/threshold storage. Only the benchmark layer varies template, threshold, and distance strategy.

ROC plots are available here:

- [Euclidean ROC](roc_euclidean.png)
- [Cosine ROC](roc_cosine.png)

## 1. Threshold Sweep Objective

The current threshold sweep does not optimize EER, accuracy, or acceptance rate.
It explicitly minimizes:

$$
\text{score}(t) = \text{FAR}(t) + \text{FRR}(t)
$$

Where:

$$
\text{FAR}(t) = \frac{\#\{d_{imp} \le t\}}{N_{imp}}, \quad
\text{FRR}(t) = \frac{\#\{d_{gen} > t\}}{N_{gen}}
$$

The candidate thresholds are the unique values from the combined genuine and impostor distance set. The sweep selects the first threshold that achieves the minimum $\text{FAR} + \text{FRR}$.

## 2. ROC / AUC / EER

Distances were aggregated across the benchmark users using the current benchmark pipeline.

| Metric | AUC | EER | EER Threshold | Best Threshold by FAR+FRR | Best FAR | Best FRR | Acceptance Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Euclidean | 0.7768 | 0.3027 | 6.8316 | 10.7974 | 0.4130 | 0.0485 | 0.9515 |
| Cosine | 0.8221 | 0.3082 | 0.1656 | 0.2482 | 0.4114 | 0.1091 | 0.8909 |

Interpretation:

- Cosine has the better AUC in the current benchmark.
- Euclidean gives a lower FRR at the threshold that minimizes FAR+FRR, but only because that optimal point occurs at a high FAR.
- The EERs are close, so the two metrics are not dramatically separated at the equal-error operating point.

## 3. FRR Under the Same FAR

The table below compares FRR at the nearest available threshold for the same FAR target.

| Target FAR | Euclidean FAR | Euclidean FRR | Euclidean Threshold | Cosine FAR | Cosine FRR | Cosine Threshold |
|---|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.0098 | 0.9273 | 4.1164 | 0.0098 | 0.6667 | 0.0919 |
| 0.05 | 0.0504 | 0.9091 | 4.3279 | 0.0504 | 0.5333 | 0.1047 |
| 0.10 | 0.1008 | 0.6788 | 5.2203 | 0.1008 | 0.4909 | 0.1108 |
| 0.20 | 0.2000 | 0.4848 | 5.9483 | 0.2000 | 0.3879 | 0.1357 |
| 0.30 | 0.2992 | 0.3091 | 6.8144 | 0.2992 | 0.3152 | 0.1648 |
| 0.40 | 0.4000 | 0.1152 | 9.0904 | 0.4000 | 0.1455 | 0.2272 |

Current conclusion:

- Cosine is clearly better at low FAR operating points.
- Euclidean only becomes competitive when FAR is already very high.

## 4. Euclidean Distance Scale Change

The current benchmark shows Euclidean distances that can reach roughly 132 to 6000+ in some user splits, while the earlier baseline example was around 3 to 7.

This is not caused by a different distance formula. The reason is data-dependent normalization instability in the per-user registration subset.

What is the same as the baseline:

- Feature extraction logic.
- Flattening logic.
- Min-max normalization logic.
- Euclidean distance computation.
- Template construction based on the registration samples.

What changes the scale:

- Some users have registration subsets where one or more feature dimensions have zero range.
- In that case, the normalization code replaces the zero range with 1.
- Later validation or impostor samples can still take large raw values on those dimensions.
- After normalization, those dimensions remain on a raw-like scale and can dominate the Euclidean norm.

Observed examples from the current benchmark:

- Amber: no zero-range dimensions, Euclidean distances stay in a small range.
- Jay: zero-range dimensions exist, and impostor distances blow up to more than 194,000 in the aggregated benchmark.
- 666 and background also show zero-range dimensions and very large impostor distances.

This means the pipeline is consistent with the baseline code path, but the benchmark distribution is not scale-stable for all users.

## 5. Evidence From the Baseline Path

The verified baseline path still behaves as expected on a user-level registration/authentication check:

- Registration distances for Amber were in the range 3.60 to 19.18.
- The threshold was 7.2074.
- A sample authentication distance was 3.9568.

That confirms the baseline code path is intact.

The large benchmark-scale values come from specific benchmark users and their registration statistics, not from a change in the underlying distance implementation.

## 6. Summary

1. Threshold sweep currently optimizes $\text{FAR} + \text{FRR}$.
2. Cosine has better AUC and better low-FAR FRR in the current benchmark.
3. The Euclidean scale inflation is caused by per-user normalization on low-variance registration subsets.
4. No algorithmic code was modified for this analysis.
