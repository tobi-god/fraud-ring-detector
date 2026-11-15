# Track A: all windows in one table

| window start | rows | laundering rows | matched to pattern | test positives | M1 PR-AUC | M2 PR-AUC | M3 PR-AUC | M2-M1 (95%) | M3-M2 (95%) |
|---|---|---|---|---|---|---|---|---|---|
| rows 0.85 | 320099 | 267 (0.08%) | 60% | 47 | 0.034 (0.013 to 0.114) | 0.070 (0.030 to 0.167) | 0.001 (0.001 to 0.001) | 0.039 (-0.053 to 0.132) | -0.079 (-0.166 to -0.029) |
| time 0.3 | 311621 | 395 (0.13%) | 64% | 42 | 0.016 (0.008 to 0.029) | 0.133 (0.061 to 0.294) | 0.052 (0.021 to 0.140) | 0.138 (0.051 to 0.267) | -0.088 (-0.178 to -0.003) |
| time 0.5 | 306128 | 504 (0.16%) | 60% | 132 | 0.045 (0.032 to 0.059) | 0.151 (0.095 to 0.220) | 0.036 (0.019 to 0.065) | 0.107 (0.058 to 0.171) | -0.114 (-0.177 to -0.064) |

## Recall at EQUAL alert volume (per 10,000 transactions), test period

| window | volume | M1 | M2 | M3 | UNCLASSIFIED recall at 20 (M1 / M2 / M3) |
|---|---|---|---|---|---|
| rows 0.85 | 20 | 0.106 | 0.191 | 0.000 | 0.056 / 0.111 / 0.000 |
| rows 0.85 | 50 | 0.213 | 0.277 | 0.000 |  |
| time 0.3 | 20 | 0.071 | 0.357 | 0.167 | 0.000 / 0.222 / 0.037 |
| time 0.3 | 50 | 0.143 | 0.452 | 0.238 |  |
| time 0.5 | 20 | 0.061 | 0.242 | 0.076 | 0.000 / 0.091 / 0.018 |
| time 0.5 | 50 | 0.167 | 0.311 | 0.182 |  |

## Listed pattern instances in the test period at 20 alerts per 10,000 (instances / any flagged / majority flagged)

| window | M1 | M2 | M3 |
|---|---|---|---|
| rows 0.85 | (2, 0, 0) | (2, 0, 0) | (2, 0, 0) |
| time 0.3 | (1, 1, 1) | (1, 1, 1) | (1, 0, 0) |
| time 0.5 | (16, 4, 3) | (16, 9, 9) | (16, 2, 2) |

**Excluded: window time 0.7** has only 374 rows with 58.8% laundering (activity thins out late in the file), so its scores are meaningless.

## Across windows

- M1: mean PR-AUC 0.032 (range 0.016 to 0.045) over 3 window(s)
- M2: mean PR-AUC 0.118 (range 0.070 to 0.151) over 3 window(s)
- M3: mean PR-AUC 0.029 (range 0.001 to 0.052) over 3 window(s)
- Windows where the M3 minus M2 interval is entirely below zero: 3 of 3
