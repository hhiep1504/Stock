# Reproducibility

## Environment

The project was developed with Python 3.10+ and PyTorch/PyTorch Geometric. A
CUDA GPU is recommended for the full neural benchmark, but most configuration
and data-preparation checks can run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Checks

Use dry-run mode to validate paths, data loading, feature construction, and
sequence creation without launching a full training run:

```bash
python main.py --config configs/default_experiment.json --dry-run
```

Run the lightweight unit checks:

```bash
python -m unittest discover -s tests
```

## Main Workflows

Standard benchmark:

```bash
python main.py --mode benchmark --config configs/default_benchmark.json
```

Cross-validation benchmark:

```bash
python main.py --mode benchmark-cv --config configs/cross_validation_benchmark.json
```

Optuna tuning:

```bash
python main.py --mode tune --config configs/default_experiment.json --trials 40
```

## Generated Artifacts

Experiment outputs are written to `logs/` and `outputs/`. These directories are
ignored by git because they can become large and are run-specific. Curated
figures for the README live in `docs/figures/`.
