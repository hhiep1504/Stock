"""Derive a cleaner HOSE-Liquid-100-Strict dataset from the HOSE crawl.

This script reuses the full VNDIRECT HOSE crawl and selection metrics produced
by build_hose_universe_dataset.py. It applies stricter coverage and liquidity
filters, writes model-ready wide prices and targets, and creates recommended
configs for paper-main experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_hose_universe_dataset import (
    write_daily_returns,
    write_recommended_config,
    write_weekly_min_max,
    write_wide_prices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-long", type=Path, default=Path("dataset/hose_liquid_150_ohlcv_raw_long.csv"))
    parser.add_argument("--base-report", type=Path, default=Path("dataset/hose_liquid_150_selection_report.csv"))
    parser.add_argument("--base-sector-map", type=Path, default=Path("dataset/hose_liquid_150_sector_map.csv"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-prefix", default="hose_liquid_100_strict")
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--min-trading-days", type=int, default=1800)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.85)
    parser.add_argument("--max-stale-days", type=int, default=7)
    parser.add_argument("--min-median-value", type=float, default=0.0)
    parser.add_argument("--wide-price-field", default="adClose")
    return parser.parse_args()


def select_strict_universe(report: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = ["code", "trading_days", "coverage_ratio", "last_gap_days", "median_value_recent"]
    missing = [column for column in required if column not in report.columns]
    if missing:
        raise ValueError(f"Selection report is missing columns: {missing}")

    out = report.copy()
    out["strict_eligible"] = (
        (pd.to_numeric(out["trading_days"], errors="coerce").fillna(0) >= args.min_trading_days)
        & (pd.to_numeric(out["coverage_ratio"], errors="coerce").fillna(0.0) >= args.min_coverage_ratio)
        & (pd.to_numeric(out["last_gap_days"], errors="coerce").fillna(9999) <= args.max_stale_days)
        & (pd.to_numeric(out["median_value_recent"], errors="coerce").fillna(0.0) >= args.min_median_value)
    )

    eligible = out[out["strict_eligible"]].sort_values(
        ["median_value_recent", "trading_days", "coverage_ratio", "code"],
        ascending=[False, False, False, True],
    )
    selected = eligible.head(args.target_size).copy()
    if len(selected) < args.target_size:
        raise RuntimeError(
            f"Only {len(selected)} symbols passed strict filters; target_size={args.target_size}. "
            "Relax thresholds or reduce target size."
        )

    selected_codes = selected["code"].astype(str).str.upper().tolist()
    rank_map = {code: rank + 1 for rank, code in enumerate(selected_codes)}
    out["selected"] = out["code"].astype(str).str.upper().isin(selected_codes)
    out["selection_rank"] = out["code"].astype(str).str.upper().map(rank_map)
    return out.sort_values(["selected", "selection_rank", "median_value_recent"], ascending=[False, True, False])


def subset_sector_map(base_sector_map: Path, selected_codes: list[str]) -> pd.DataFrame:
    if not base_sector_map.exists():
        return pd.DataFrame({"ticker": selected_codes, "sector": "Unknown"})

    sector = pd.read_csv(base_sector_map)
    ticker_col = "ticker" if "ticker" in sector.columns else "code"
    sector[ticker_col] = sector[ticker_col].astype(str).str.upper()
    sector = sector[sector[ticker_col].isin(selected_codes)].copy()
    found = set(sector[ticker_col])
    missing = [code for code in selected_codes if code not in found]
    if missing:
        sector = pd.concat(
            [sector, pd.DataFrame({ticker_col: missing, "sector": "Unknown"})],
            ignore_index=True,
        )
    sector["selection_rank"] = sector[ticker_col].map({code: i + 1 for i, code in enumerate(selected_codes)})
    sector = sector.sort_values("selection_rank")
    if ticker_col != "ticker":
        sector = sector.rename(columns={ticker_col: "ticker"})
    return sector


def main() -> None:
    args = parse_args()
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    args.configs_dir.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(args.base_report)
    strict_report = select_strict_universe(report, args)
    selected = strict_report[strict_report["selected"]].sort_values("selection_rank")
    selected_codes = selected["code"].astype(str).str.upper().tolist()

    raw = pd.read_csv(args.raw_long)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["code"] = raw["code"].astype(str).str.upper()
    selected_prices = raw[raw["code"].isin(selected_codes)].copy()
    if selected_prices.empty:
        raise RuntimeError("No price rows matched the strict selected universe.")

    paths = {
        "selection_report": args.dataset_dir / f"{args.output_prefix}_selection_report.csv",
        "selected_long": args.dataset_dir / f"{args.output_prefix}_ohlcv_selected_long.csv",
        "wide_close": args.dataset_dir / f"{args.output_prefix}_adclose_wide.csv",
        "daily_return": args.dataset_dir / f"{args.output_prefix}_daily_return.csv",
        "weekly_min_max": args.dataset_dir / f"{args.output_prefix}_weekly_min_max_return.csv",
        "sector_map": args.dataset_dir / f"{args.output_prefix}_sector_map.csv",
        "manifest": args.dataset_dir / f"{args.output_prefix}_manifest.json",
        "config_dynamic": args.configs_dir / f"{args.output_prefix}_dynamic.json",
        "config_hybrid": args.configs_dir / f"{args.output_prefix}_hybrid.json",
    }

    strict_report.to_csv(paths["selection_report"], index=False)
    selected_prices.to_csv(paths["selected_long"], index=False)
    wide = write_wide_prices(selected_prices, selected_codes, paths["wide_close"], args.wide_price_field)
    write_daily_returns(wide, paths["daily_return"])
    write_weekly_min_max(wide, paths["weekly_min_max"])

    sector = subset_sector_map(args.base_sector_map, selected_codes)
    sector.to_csv(paths["sector_map"], index=False)

    write_recommended_config(
        path=paths["config_dynamic"],
        daily_file=paths["wide_close"],
        weekly_file=paths["weekly_min_max"],
        sector_file=paths["sector_map"],
        graph_mode="dynamic",
    )
    write_recommended_config(
        path=paths["config_hybrid"],
        daily_file=paths["wide_close"],
        weekly_file=paths["weekly_min_max"],
        sector_file=paths["sector_map"],
        graph_mode="hybrid",
    )

    missing_fraction = wide.isna().mean().sort_values(ascending=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Derived from VNDIRECT HOSE crawl",
        "base_raw_long": str(args.raw_long),
        "base_report": str(args.base_report),
        "filters": {
            "target_size": args.target_size,
            "min_trading_days": args.min_trading_days,
            "min_coverage_ratio": args.min_coverage_ratio,
            "max_stale_days": args.max_stale_days,
            "min_median_value": args.min_median_value,
            "wide_price_field": args.wide_price_field,
        },
        "counts": {
            "base_symbols": int(report["code"].nunique()),
            "strict_eligible_symbols": int(strict_report["strict_eligible"].sum()),
            "selected_symbols": int(len(selected_codes)),
            "selected_rows": int(len(selected_prices)),
            "wide_days": int(len(wide)),
            "total_cells": int(np.prod(wide.shape)),
            "missing_cells": int(wide.isna().sum().sum()),
        },
        "date_range": {
            "first": pd.Timestamp(wide.index.min()).date().isoformat(),
            "last": pd.Timestamp(wide.index.max()).date().isoformat(),
        },
        "quality": {
            "max_missing_fraction": float(missing_fraction.max()),
            "median_missing_fraction": float(missing_fraction.median()),
            "min_trading_days_selected": int(selected["trading_days"].min()),
            "min_coverage_ratio_selected": float(selected["coverage_ratio"].min()),
            "min_median_value_recent_selected": float(selected["median_value_recent"].min()),
        },
        "selected_symbols": selected_codes,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Selected {len(selected_codes)} strict symbols from {len(report)} HOSE symbols.")
    print(f"Wide file: {paths['wide_close']} ({wide.shape[0]} days x {wide.shape[1]} symbols)")
    print(f"Date range: {manifest['date_range']['first']} -> {manifest['date_range']['last']}")
    print(f"Max missing fraction: {manifest['quality']['max_missing_fraction']:.4f}")
    print(f"Median missing fraction: {manifest['quality']['median_missing_fraction']:.4f}")
    print(f"Recommended config: {paths['config_dynamic']}")


if __name__ == "__main__":
    main()
