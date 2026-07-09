# Technical Feature Test - 2026-05-24

Protocol:
- Weekly target setup from `analysis/test_feature_pruning_weekly.py`
- GRU sequence model
- lookback = 8
- epochs = 40
- learning rate = 0.001
- seed = 42
- expanding time splits = 5
- `--fair-mode` enabled so baseline and extended setup use the same valid rows

## Features Added

Original baseline:
- `f_std`
- `f_mean`
- `f_return`
- `f_skew`

Previously tested extras:
- `alpha_excess`
- `weekly_range_lag1`
- `rel_pos_4w`

New technical features:
- `sma_ratio_4`
- `ema_ratio_4`
- `bollinger_width_4`
- `bollinger_percent_b_4`
- `rsi_14`
- `macd_norm`
- `hurst_20`
- `entropy_20`

`vw_return` was still unavailable because the dataset has no volume columns.

## Run 1: All Features Including Entropy

Common valid rows after `dropna`: 1239 / 7422.

`entropy_20` caused heavy missingness:
- `entropy_20`: 6121 / 7422 NaN (82.5%)
- `hurst_20`: 1221 / 7422 NaN (16.5%)
- `macd_norm`: 650 / 7422 NaN (8.8%)

| setup | folds | r2_avg_mean | mae_avg_mean |
|---|---:|---:|---:|
| baseline_4 | 5 | -0.110931 | 0.026178 |
| extended_features | 5 | -0.730500 | 0.035249 |
| pruned_extended_to_8 | 5 | -0.278472 | 0.030173 |
| optuna_best_subset | 5 | -0.083920 | 0.027799 |

Best Optuna subset:
- `alpha_excess`
- `weekly_range_lag1`
- `sma_ratio_4`
- `bollinger_width_4`
- `bollinger_percent_b_4`
- `hurst_20`
- `entropy_20`

Conclusion: all features including entropy did not improve MAE. The best subset improved R2 slightly versus the reduced-row baseline, but MAE was worse.

## Run 2: All Technical Features Excluding Entropy

Common valid rows after `dropna`: 5846 / 7422.

| setup | folds | r2_avg_mean | mae_avg_mean |
|---|---:|---:|---:|
| baseline_4 | 5 | -0.008645 | 0.023310 |
| extended_features | 5 | -0.191124 | 0.025983 |
| pruned_extended_to_8 | 5 | -0.091123 | 0.025047 |
| optuna_best_subset | 5 | 0.007098 | 0.023144 |

Best Optuna subset:
- `f_skew`
- `weekly_range_lag1`
- `rel_pos_4w`
- `bollinger_width_4`

Conclusion: excluding entropy makes the test much fairer. The Optuna subset gives a very small MAE improvement over baseline (`0.023144` vs `0.023310`), but the full extended feature set is worse.

## Overall Conclusion

Do not add all technical features blindly to the main pipeline. The full feature set makes performance worse.

The only promising result from this run is a small subset:
- `f_skew`
- `weekly_range_lag1`
- `rel_pos_4w`
- `bollinger_width_4`

This subset should be treated as a weak candidate and retested across multiple seeds before replacing the current four-feature pipeline.
