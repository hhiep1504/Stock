# Feature Benchmark Plan - 2026-05-24

## Recommended protocol

Use an expanding-window walk-forward benchmark for the official comparison.
This trains only on past weekly sequences and tests on the next future block,
which is safer for stock-market experiments than k-fold validation that can
train on data after the test fold.

Default protocol in the new configs:

- Weekly aggregation
- Window size: 8
- Hybrid graph
- Pearson similarity
- top_k: 4
- MSE loss for fair model comparison
- Seeds: 42, 52, 62
- Runs per fold: 3
- GNN/baseline epochs: 80
- Min train sequences: 150
- Test step: 15

## Feature sets to benchmark

| Feature set | Dim | Complete periods | Sequences | Purpose |
| --- | ---: | ---: | ---: | --- |
| baseline4 | 4 | 288 | 280 | Current official baseline |
| compact_technical | 4 | 285 | 277 | Exact compact subset suggested by feature pruning |
| baseline_plus_compact | 7 | 285 | 277 | Safer candidate: baseline plus compact technical signals |
| technical_no_entropy | 14 | 255 | 247 | Full technical ablation without entropy |

Do not use `entropy_20` in the main graph benchmark. It creates too much
missingness for a complete 26-node weekly panel and should be reported as an
unsuccessful feature-engineering ablation instead.

## RTX 3090 command

Run the feature-window sweep first:

```bash
bash scripts/run_feature_window_sweep_3090.sh
```

Or on Windows PowerShell:

```powershell
.\scripts\run_feature_window_sweep_3090.ps1
```

The sweep tests these default parameter grids:

- SMA/EMA ratio windows: 2, 3, 4, 5, 6, 8, 12 weeks
- Bollinger windows: 4, 5, 6, 8, 12 weeks
- Bollinger multipliers: 1.5, 2.0
- RSI windows: 4, 5, 8, 14, 20 weeks
- MACD fast/slow pairs: 3/6, 4/8, 5/10, 6/12, 8/17, 12/26 weeks
- Hurst windows: 12, 20, 26 weeks
- Entropy windows: 12, 20, 26 weeks
- Relative-position windows: 2, 3, 4, 5, 6, 8, 12 weeks

Then run the official benchmark on selected feature sets:

```bash
bash scripts/run_feature_benchmarks_3090.sh
```

Quick smoke command:

```bash
bash scripts/run_feature_benchmarks_3090.sh --quick
```
