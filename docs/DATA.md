# Data Notes

The repository keeps small, reproducible input files needed for smoke tests and
the default experiment:

- `dataset/stock_market_19_24.csv`
- `dataset/min_max_return.csv`

Several larger HOSE universe files are present locally but are intentionally
ignored by git because they are raw or derived data artifacts:

- `dataset/*_ohlcv_raw_long.csv`
- `dataset/*_ohlcv_selected_long.csv`
- `dataset/cafef_*.zip`
- `dataset/source_audit/`

These files can be regenerated with the data scripts in `scripts/`, depending
on data-source availability:

```bash
python scripts/build_hose_universe_dataset.py
python scripts/derive_hose_strict_dataset.py
python scripts/compare_vietnam_data_sources.py
```

Configuration files under `configs/` reference dataset paths relative to the
project root. If you use a different dataset location, update the `data` section
of the relevant JSON config.
