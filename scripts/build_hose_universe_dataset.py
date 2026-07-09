"""Build a larger HOSE stock dataset for spatial-temporal experiments.

The legacy project dataset has 26 real-estate/infrastructure tickers. This
script builds a broader HOSE universe from VNDIRECT public endpoints, filters a
liquid subset, and exports files that remain compatible with the current
weekly GAT-LSTM pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_COLUMN = "History Price / Date"
VNDIRECT_BASE_URL = "https://api-finfo.vndirect.com.vn/v4"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://dstock.vndirect.com.vn/",
}
LEGACY_TICKERS = (
    "CCL",
    "CDC",
    "CRE",
    "D2D",
    "DIG",
    "DXG",
    "HDC",
    "HDG",
    "HPX",
    "IJC",
    "ITC",
    "KBC",
    "KDH",
    "LHG",
    "NBB",
    "NLG",
    "NTL",
    "NVL",
    "PDR",
    "SZL",
    "TCH",
    "TDC",
    "TIP",
    "TIX",
    "VHM",
    "VPI",
)
PRICE_FIELDS = (
    "basicPrice",
    "ceilingPrice",
    "floorPrice",
    "open",
    "high",
    "low",
    "close",
    "average",
    "adOpen",
    "adHigh",
    "adLow",
    "adClose",
    "adAverage",
    "change",
    "adChange",
)
NUMERIC_FIELDS = PRICE_FIELDS + (
    "nmVolume",
    "nmValue",
    "ptVolume",
    "ptValue",
    "pctChange",
)


@dataclass(slots=True)
class OutputPaths:
    symbols: Path
    raw_long: Path
    selected_long: Path
    wide_close: Path
    daily_return: Path
    weekly_min_max: Path
    selection_report: Path
    sector_map: Path
    manifest: Path
    config: Path | None


def _request_json(url: str, retries: int = 3, pause_seconds: float = 0.8) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - network/API failures are retried.
            last_error = exc
            time.sleep(pause_seconds * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def _build_url(path: str, params: dict[str, Any]) -> str:
    return f"{VNDIRECT_BASE_URL}/{path}?{urllib.parse.urlencode(params)}"


def fetch_hose_symbols(status: str = "listed") -> pd.DataFrame:
    """Fetch HOSE equity symbols from VNDIRECT stock metadata."""

    query_parts = ["type:STOCK", "floor:HOSE"]
    if status.lower() != "all":
        query_parts.append(f"status:{status.upper()}")

    payload = _request_json(
        _build_url(
            "stocks",
            {
                "q": "~".join(query_parts),
                "size": 2000,
            },
        )
    )
    symbols = pd.DataFrame(payload.get("data", []))
    if symbols.empty or "code" not in symbols.columns:
        raise RuntimeError("VNDIRECT did not return any HOSE stock symbols.")
    symbols["code"] = symbols["code"].astype(str).str.upper().str.strip()
    symbols = symbols.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    return symbols


def fetch_stock_prices(
    symbol: str,
    start: str,
    end: str,
    page_size: int = 5000,
    request_pause: float = 0.05,
) -> pd.DataFrame:
    """Fetch daily OHLCV rows for one symbol."""

    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        params = {
            "q": f"code:{symbol}~date:gte:{start}~date:lte:{end}",
            "sort": "date",
            "size": page_size,
            "page": page,
        }
        payload = _request_json(_build_url("stock_prices", params))
        page_rows = payload.get("data", [])
        rows.extend(page_rows)

        total_pages = int(payload.get("totalPages") or 1)
        if page >= total_pages or not page_rows:
            break
        page += 1
        time.sleep(request_pause)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["code"] = symbol
    return frame


def fetch_company_profile(symbol: str) -> dict[str, Any]:
    """Fetch company profile metadata for one symbol when available."""

    payload = _request_json(
        _build_url(
            "company_profiles",
            {
                "q": f"code:{symbol}",
                "size": 1,
            },
        ),
        retries=2,
    )
    rows = payload.get("data", [])
    return rows[0] if rows else {"code": symbol}


def normalise_text(value: object) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", text.lower())


SECTOR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Banking", ("ngan hang", " tmcp ", "bank")),
    ("Securities", ("chung khoan", "securities")),
    ("Insurance", ("bao hiem", "insurance")),
    ("RealEstate", ("bat dong san", "dia oc", "nha o", "khu do thi", "vinhomes")),
    ("IndustrialPark", ("khu cong nghiep", "ha tang cong nghiep", "idico", "sonadezi")),
    ("ConstructionMaterials", ("xay dung", "xi mang", "vat lieu", "da xay dung", "gach", "betong")),
    ("Steel", ("thep", "steel", "ton")),
    ("EnergyOilGas", ("dau khi", "xang dau", "gas", "dien", "nang luong", "thuy dien")),
    ("FoodBeverageAgriculture", ("thuc pham", "sua", "bia", "duong", "nong nghiep", "thuy san", "cao su")),
    ("RetailDistribution", ("ban le", "phan phoi", "thuong mai", "the gioi di dong", "fpt retail")),
    ("TechnologyTelecom", ("cong nghe", "vien thong", "phan mem", "fpt")),
    ("HealthcarePharma", ("duoc", "y te", "benh vien", "pharma")),
    ("LogisticsTransportation", ("cang", "van tai", "hang khong", "logistics", "shipping")),
    ("ChemicalsPlasticFertilizer", ("hoa chat", "nhua", "phan bon", "plastic")),
    ("UtilitiesWater", ("cap thoat nuoc", "nuoc sach", "moi truong")),
    ("Manufacturing", ("san xuat", "cong nghiep", "det may", "may mac", "go", "bao bi")),
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_sector(row: pd.Series) -> tuple[str, str, float]:
    """Infer a coarse sector from VNDIRECT company text fields."""

    name_text = " ".join(
        normalise_text(row.get(column))
        for column in (
            "companyName",
            "companyNameEng",
            "shortName",
            "vnName",
            "enName",
        )
    )
    summary_text = " ".join(
        normalise_text(row.get(column))
        for column in (
            "vnSummary",
            "enSummary",
        )
    )
    padded_name = f" {name_text} "
    padded_full = f" {name_text} {summary_text} "

    # "Chung khoan" appears in many company summaries as part of "stock
    # exchange", so securities classification must come from the company name.
    if _contains_any(padded_name, ("chung khoan", "securities")):
        return "Securities", "vndirect_profile_keywords", 0.85

    for sector, keywords in SECTOR_RULES:
        if sector == "Securities":
            continue
        if _contains_any(padded_full, keywords):
            return sector, "vndirect_profile_keywords", 0.75
    return "Unknown", "vndirect_profile_keywords", 0.0


def prepare_price_frame(raw_prices: pd.DataFrame, price_multiplier: float) -> pd.DataFrame:
    """Clean and scale raw API prices into a single long frame."""

    if raw_prices.empty:
        return raw_prices

    frame = raw_prices.copy()
    frame["code"] = frame["code"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "code"])

    for column in NUMERIC_FIELDS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in PRICE_FIELDS:
        if column in frame.columns:
            frame[column] = frame[column] * price_multiplier

    frame = frame.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    return frame.reset_index(drop=True)


def build_selection_report(
    prices: pd.DataFrame,
    symbols: pd.DataFrame,
    liquidity_window_days: int,
) -> pd.DataFrame:
    """Compute coverage and liquidity metrics for all fetched symbols."""

    if prices.empty:
        raise RuntimeError("No price rows were fetched.")

    metric_rows: list[dict[str, Any]] = []
    calendar = pd.Index(sorted(prices["date"].dropna().unique()))
    calendar_size = len(calendar)
    max_date = pd.Timestamp(calendar.max())

    for symbol, group in prices.groupby("code", sort=True):
        usable = group.dropna(subset=["adClose"]).sort_values("date")
        if usable.empty:
            metric_rows.append({"code": symbol, "trading_days": 0})
            continue

        recent = usable.tail(liquidity_window_days)
        first_date = pd.Timestamp(usable["date"].min())
        last_date = pd.Timestamp(usable["date"].max())
        metric_rows.append(
            {
                "code": symbol,
                "first_date": first_date.date().isoformat(),
                "last_date": last_date.date().isoformat(),
                "trading_days": int(usable["date"].nunique()),
                "coverage_ratio": float(usable["date"].nunique() / calendar_size) if calendar_size else 0.0,
                "last_gap_days": int((max_date - last_date).days),
                "median_value_recent": float(recent["nmValue"].median(skipna=True)),
                "mean_value_recent": float(recent["nmValue"].mean(skipna=True)),
                "median_volume_recent": float(recent["nmVolume"].median(skipna=True)),
                "mean_volume_recent": float(recent["nmVolume"].mean(skipna=True)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    return metrics.merge(symbols, on="code", how="left")


def choose_universe(
    report: pd.DataFrame,
    target_size: int,
    min_trading_days: int,
    min_coverage_ratio: float,
    max_stale_days: int,
    min_median_value: float,
    force_include_tickers: tuple[str, ...],
) -> pd.DataFrame:
    """Select a liquid universe, preserving requested legacy tickers when possible."""

    filtered = report.copy()
    filtered["median_value_recent"] = pd.to_numeric(filtered["median_value_recent"], errors="coerce").fillna(0.0)
    filtered["trading_days"] = pd.to_numeric(filtered["trading_days"], errors="coerce").fillna(0).astype(int)
    filtered["coverage_ratio"] = pd.to_numeric(filtered["coverage_ratio"], errors="coerce").fillna(0.0)
    filtered["last_gap_days"] = pd.to_numeric(filtered["last_gap_days"], errors="coerce").fillna(9999).astype(int)

    eligible_mask = (
        (filtered["trading_days"] >= min_trading_days)
        & (filtered["coverage_ratio"] >= min_coverage_ratio)
        & (filtered["last_gap_days"] <= max_stale_days)
        & (filtered["median_value_recent"] >= min_median_value)
    )
    filtered["eligible"] = eligible_mask
    filtered["forced_include"] = filtered["code"].isin(force_include_tickers)

    forced = filtered[
        filtered["forced_include"]
        & (filtered["trading_days"] > 0)
        & (filtered["last_gap_days"] <= max_stale_days)
    ].copy()
    eligible = filtered[filtered["eligible"]].copy()

    sort_columns = ["median_value_recent", "trading_days", "coverage_ratio", "code"]
    eligible = eligible.sort_values(sort_columns, ascending=[False, False, False, True])
    forced = forced.sort_values(sort_columns, ascending=[False, False, False, True])

    if target_size <= 0:
        selected_codes = list(dict.fromkeys([*forced["code"].tolist(), *eligible["code"].tolist()]))
    else:
        selected_codes = list(forced["code"].tolist())
        for code in eligible["code"].tolist():
            if code not in selected_codes:
                selected_codes.append(code)
            if len(selected_codes) >= target_size:
                break

    filtered["selected"] = filtered["code"].isin(selected_codes)
    filtered["selection_rank"] = np.nan
    rank_map = {code: rank + 1 for rank, code in enumerate(selected_codes)}
    filtered.loc[filtered["selected"], "selection_rank"] = filtered.loc[filtered["selected"], "code"].map(rank_map)
    return filtered.sort_values(["selected", "selection_rank", "median_value_recent"], ascending=[False, True, False])


def format_date(value: pd.Timestamp) -> str:
    return f"{value.month}/{value.day}/{value.year}"


def write_wide_prices(prices: pd.DataFrame, symbols: list[str], output_path: Path, value_column: str) -> pd.DataFrame:
    wide = prices[prices["code"].isin(symbols)].pivot(index="date", columns="code", values=value_column)
    wide = wide.reindex(columns=symbols).sort_index()

    output = wide.round(0).astype("Int64").reset_index()
    output.insert(0, DATE_COLUMN, output["date"].map(format_date))
    output = output.drop(columns=["date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return wide


def write_daily_returns(prices: pd.DataFrame, output_path: Path) -> None:
    returns = prices.pct_change(fill_method=None)
    output = returns.reset_index()
    output.insert(0, DATE_COLUMN, output["date"].map(format_date))
    output = output.drop(columns=["date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def write_weekly_min_max(prices: pd.DataFrame, output_path: Path) -> None:
    weekly_groups = prices.resample("W-FRI")
    weekly_open = weekly_groups.first()
    weekly_min = weekly_groups.min()
    weekly_max = weekly_groups.max()
    weekly_min_return = (weekly_min / weekly_open) - 1.0
    weekly_max_return = (weekly_max / weekly_open) - 1.0
    output = pd.concat({"Min Return": weekly_min_return, "Max Return": weekly_max_return}, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index_label="Week End")


def build_sector_map(
    selected_report: pd.DataFrame,
    request_pause: float,
    skip_profiles: bool,
) -> pd.DataFrame:
    profile_rows: list[dict[str, Any]] = []
    for row in selected_report.sort_values("selection_rank").itertuples(index=False):
        base = row._asdict()
        profile: dict[str, Any] = {"code": base["code"]}
        if not skip_profiles:
            try:
                profile = fetch_company_profile(base["code"])
                time.sleep(request_pause)
            except Exception as exc:  # noqa: BLE001 - profile metadata is non-critical.
                profile = {"code": base["code"], "profile_error": str(exc)}
        profile_rows.append({**base, **profile})

    profiles = pd.DataFrame(profile_rows)
    sector_rows = []
    for _, row in profiles.iterrows():
        sector, source, confidence = infer_sector(row)
        sector_rows.append(
            {
                "ticker": row.get("code"),
                "sector": sector,
                "sector_source": source,
                "sector_confidence": confidence,
            }
        )
    sector_frame = pd.DataFrame(sector_rows)
    return sector_frame.merge(profiles, left_on="ticker", right_on="code", how="left")


def write_recommended_config(
    path: Path,
    daily_file: Path,
    weekly_file: Path,
    sector_file: Path,
    graph_mode: str,
) -> None:
    config = {
        "experiment_name": f"hose_liquid_{graph_mode}",
        "paths": {
            "project_root": ".",
            "dataset_dir": "dataset",
            "outputs_dir": "outputs",
            "logs_dir": "logs",
            "configs_dir": "configs",
        },
        "data": {
            "daily_file": str(daily_file).replace("\\", "/"),
            "target_file": str(weekly_file).replace("\\", "/"),
            "aggregation_mode": "weekly",
            "feature_set": "baseline4",
            "window_size": 8,
            "split_idx": -12,
        },
        "graph": {
            "graph_mode": graph_mode,
            "sector_map_file": str(sector_file).replace("\\", "/"),
            "top_k": 5,
            "use_arm": False,
            "use_static_graph": graph_mode != "dynamic",
            "similarity_metric": "pearson",
            "corr_threshold": 0.55,
        },
        "model": {
            "model_type": "deep",
            "in_features": 4,
            "gnn_hidden": 128,
            "lstm_hidden": 256,
            "num_layers": 1,
            "heads": 2,
            "dropout": 0.5,
        },
        "loss": {
            "name": "correlation",
            "alpha": 0.5,
            "weight_mse": 0.45,
            "weight_corr": 0.45,
            "weight_penalty": 0.1,
        },
        "training": {
            "epochs": 30,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "print_every": 1,
            "early_stopping_patience": 10,
            "warmup_epochs": 5,
            "min_delta": 0.0001,
            "use_scheduler": True,
        },
        "runtime": {
            "random_seed": 42,
            "device": "auto",
            "save_checkpoint": True,
            "save_plots": True,
            "show_plots": False,
        },
        "benchmark": {
            "output_dir": "logs",
            "protocol": "walk_forward",
            "seeds": [42, 52, 62],
            "gnn_epochs": 80,
            "baseline_epochs": 80,
            "device": "auto",
            "n_folds": 5,
            "n_runs": 3,
            "include_statistical_models": True,
            "validation_split": 0.1,
            "early_stopping_patience": 8,
            "min_train_size": 150,
            "test_step": 15,
        },
        "optuna": {
            "n_trials": 40,
            "direction": "minimize",
            "metric": "mae_interval",
            "study_name": f"hose_liquid_{graph_mode}",
            "storage_url": None,
            "persist_study": True,
            "resume_study": True,
            "sampler_seed": 42,
            "max_epochs_per_trial": 10,
            "prune_after_epochs": 5,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def output_paths(dataset_dir: Path, configs_dir: Path, prefix: str, write_config: bool) -> OutputPaths:
    config_path = configs_dir / f"{prefix}_hybrid.json" if write_config else None
    return OutputPaths(
        symbols=dataset_dir / f"{prefix}_symbols.csv",
        raw_long=dataset_dir / f"{prefix}_ohlcv_raw_long.csv",
        selected_long=dataset_dir / f"{prefix}_ohlcv_selected_long.csv",
        wide_close=dataset_dir / f"{prefix}_adclose_wide.csv",
        daily_return=dataset_dir / f"{prefix}_daily_return.csv",
        weekly_min_max=dataset_dir / f"{prefix}_weekly_min_max_return.csv",
        selection_report=dataset_dir / f"{prefix}_selection_report.csv",
        sector_map=dataset_dir / f"{prefix}_sector_map.csv",
        manifest=dataset_dir / f"{prefix}_manifest.json",
        config=config_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--status", choices=["listed", "all"], default="listed")
    parser.add_argument("--target-size", type=int, default=150)
    parser.add_argument("--min-trading-days", type=int, default=1000)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.45)
    parser.add_argument("--max-stale-days", type=int, default=14)
    parser.add_argument("--min-median-value", type=float, default=0.0)
    parser.add_argument("--liquidity-window-days", type=int, default=252)
    parser.add_argument("--request-pause", type=float, default=0.05)
    parser.add_argument("--page-size", type=int, default=5000)
    parser.add_argument("--price-multiplier", type=float, default=1000.0)
    parser.add_argument("--wide-price-field", default="adClose")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-prefix", default="hose_liquid_150")
    parser.add_argument("--skip-raw-output", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument(
        "--force-include-tickers",
        default=",".join(LEGACY_TICKERS),
        help="Comma-separated tickers to preserve from the legacy 26-stock study when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now().isoformat(timespec="seconds")
    paths = output_paths(args.dataset_dir, args.configs_dir, args.output_prefix, not args.no_config)

    print(f"Fetching HOSE {args.status} stock list...")
    symbols = fetch_hose_symbols(args.status)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    symbols.to_csv(paths.symbols, index=False)
    print(f"Fetched {len(symbols)} HOSE stock symbols -> {paths.symbols}")

    price_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for idx, symbol in enumerate(symbols["code"].tolist(), start=1):
        print(f"[{idx:03d}/{len(symbols)}] Fetching {symbol}")
        try:
            frame = fetch_stock_prices(
                symbol=symbol,
                start=args.start,
                end=args.end,
                page_size=args.page_size,
                request_pause=args.request_pause,
            )
            if not frame.empty:
                price_frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - keep the crawl progressing.
            failures.append({"code": symbol, "error": str(exc)})
        time.sleep(args.request_pause)

    if not price_frames:
        raise RuntimeError("No stock price data was fetched.")

    raw_prices = prepare_price_frame(pd.concat(price_frames, ignore_index=True), args.price_multiplier)
    if not args.skip_raw_output:
        raw_prices.to_csv(paths.raw_long, index=False)
        print(f"Wrote raw long OHLCV: {paths.raw_long} ({len(raw_prices):,} rows)")

    report = build_selection_report(raw_prices, symbols, args.liquidity_window_days)
    force_include = tuple(
        ticker.strip().upper()
        for ticker in args.force_include_tickers.split(",")
        if ticker.strip()
    )
    report = choose_universe(
        report=report,
        target_size=args.target_size,
        min_trading_days=args.min_trading_days,
        min_coverage_ratio=args.min_coverage_ratio,
        max_stale_days=args.max_stale_days,
        min_median_value=args.min_median_value,
        force_include_tickers=force_include,
    )
    report.to_csv(paths.selection_report, index=False)

    selected = report[report["selected"]].sort_values("selection_rank")
    selected_symbols = selected["code"].tolist()
    if not selected_symbols:
        raise RuntimeError("Filtering selected no symbols. Relax the thresholds and rerun.")

    selected_prices = raw_prices[raw_prices["code"].isin(selected_symbols)].copy()
    selected_prices.to_csv(paths.selected_long, index=False)
    wide_prices = write_wide_prices(selected_prices, selected_symbols, paths.wide_close, args.wide_price_field)
    write_daily_returns(wide_prices, paths.daily_return)
    write_weekly_min_max(wide_prices, paths.weekly_min_max)

    sector_map = build_sector_map(
        selected_report=selected,
        request_pause=args.request_pause,
        skip_profiles=args.skip_profiles,
    )
    sector_map.to_csv(paths.sector_map, index=False)

    if paths.config is not None:
        write_recommended_config(
            path=paths.config,
            daily_file=paths.wide_close,
            weekly_file=paths.weekly_min_max,
            sector_file=paths.sector_map,
            graph_mode="hybrid",
        )

    manifest = {
        "created_at": started_at,
        "source": {
            "provider": "VNDIRECT public api-finfo endpoints",
            "base_url": VNDIRECT_BASE_URL,
            "symbol_status": args.status,
            "start": args.start,
            "end": args.end,
            "price_multiplier": args.price_multiplier,
            "wide_price_field": args.wide_price_field,
        },
        "filters": {
            "target_size": args.target_size,
            "min_trading_days": args.min_trading_days,
            "min_coverage_ratio": args.min_coverage_ratio,
            "max_stale_days": args.max_stale_days,
            "min_median_value": args.min_median_value,
            "liquidity_window_days": args.liquidity_window_days,
            "force_include_tickers": force_include,
        },
        "counts": {
            "symbols": int(len(symbols)),
            "fetched_symbols": int(raw_prices["code"].nunique()),
            "raw_rows": int(len(raw_prices)),
            "selected_symbols": int(len(selected_symbols)),
            "selected_rows": int(len(selected_prices)),
            "wide_days": int(len(wide_prices)),
            "failures": int(len(failures)),
        },
        "selected_symbols": selected_symbols,
        "fetch_failures": failures,
        "outputs": {key: str(value) if value is not None else None for key, value in asdict(paths).items()},
    }
    with paths.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Selected {len(selected_symbols)} symbols from {len(symbols)} HOSE symbols.")
    print(f"Wide price file: {paths.wide_close} ({wide_prices.shape[0]} days x {wide_prices.shape[1]} symbols)")
    print(f"Weekly target file: {paths.weekly_min_max}")
    print(f"Sector map: {paths.sector_map}")
    if paths.config is not None:
        print(f"Recommended config: {paths.config}")
    print(f"Manifest: {paths.manifest}")


if __name__ == "__main__":
    main()
