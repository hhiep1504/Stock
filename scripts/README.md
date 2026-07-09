# Experiment Runners

This directory provides reproducible launch helpers for the paper workflow.

## Principles

- `env.sh` locks the runtime environment and exports shared variables.
- `run_arg.py` materialises a runtime JSON config for each run under `logs/runtime_configs/`.
- Shell wrappers call `run_arg.py` with a fixed experimental preset.

This separation keeps graph-construction protocol, model protocol, and run metadata traceable.

## Usage

### Final tuned neural baselines

Run all three:

```bash
bash scripts/run_final_baselines.sh
```

Run one:

```bash
bash scripts/run_final_baselines.sh --family lstm
```

Dry command preview:

```bash
python scripts/run_arg.py final-baseline --family gru --print-only
```

### Hybrid merged graph tuning

GCN-LSTM:

```bash
bash scripts/run_hybrid_merged.sh --family gcn_lstm --trials 100 --max-epochs-per-trial 12 --prune-after-epochs 8
```

GAT-LSTM:

```bash
bash scripts/run_hybrid_merged.sh --family gat_lstm --trials 100 --max-epochs-per-trial 12 --prune-after-epochs 8
```

### Dual-path GAT tuning

```bash
bash scripts/run_dual_gat.sh --trials 100 --max-epochs-per-trial 12 --prune-after-epochs 8
```

### Feature-set walk-forward benchmark

Full run for RTX 3090:

```bash
bash scripts/run_feature_benchmarks_3090.sh
```

Smoke run:

```bash
bash scripts/run_feature_benchmarks_3090.sh --quick
```

### Common-period feature benchmark

Use this for the fair feature-set comparison after screening:

```bash
bash scripts/run_common_feature_benchmarks_3090.sh --dry-run
bash scripts/run_common_feature_benchmarks_3090.sh --quick
bash scripts/run_common_feature_benchmarks_3090.sh
```

The default common-period comparison runs only:

- `baseline4`
- `baseline_plus_sma12`
- `screened_with_hurst`

### Feature-window sweep

Use this before the official feature-set benchmark when selecting technical
indicator windows:

```bash
bash scripts/run_feature_window_sweep_3090.sh
```

On Windows PowerShell:

```powershell
.\scripts\run_feature_window_sweep_3090.ps1
```

Quick smoke run:

```bash
bash scripts/run_feature_window_sweep_3090.sh --quick
```

```powershell
.\scripts\run_feature_window_sweep_3090.ps1 -Quick
```

## Notes

- The final baseline presets currently point to the best Optuna runs already produced in `logs/`.
- If a new tuning run becomes the official winner, update the corresponding path in `scripts/run_arg.py`.
- The graph presets are intentionally fixed:
  - `window_size = 8`
  - `similarity_metric = pearson`
  - `top_k = 4`
  - `corr_threshold = 0.7`
