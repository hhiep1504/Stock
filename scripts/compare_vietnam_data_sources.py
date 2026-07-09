"""Compare Vietnamese stock price datasets across independent sources.

This script audits the HOSE liquid universe against:

* VNDIRECT api-finfo output already built by build_hose_universe_dataset.py
* CafeF adjusted "Upto 3 san" AmiBroker/MetaStock zip
* Yahoo Finance chart data for .VN tickers

It writes source-specific normalized long files and pairwise discrepancy
reports so model experiments can cite data quality checks explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.VN"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}


@dataclass(slots=True)
class SourceStatus:
    source: str
    status: str
    details: str


def selected_symbols(path: Path) -> list[str]:
    report = pd.read_csv(path)
    selected = report[report["selected"] == True].sort_values("selection_rank")  # noqa: E712
    if selected.empty:
        raise ValueError(f"No selected symbols in {path}")
    return selected["code"].astype(str).str.upper().tolist()


def read_vndirect(path: Path, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["code"].astype(str).str.upper()
    mask = (
        frame["ticker"].isin(symbols)
        & (frame["date"] >= pd.Timestamp(start))
        & (frame["date"] <= pd.Timestamp(end))
    )
    columns = ["date", "ticker", "adOpen", "adHigh", "adLow", "adClose", "nmVolume", "nmValue"]
    out = frame.loc[mask, columns].rename(
        columns={
            "adOpen": "open",
            "adHigh": "high",
            "adLow": "low",
            "adClose": "close",
            "nmVolume": "volume",
            "nmValue": "value",
        }
    )
    for column in ("open", "high", "low", "close", "volume", "value"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["source"] = "vndirect"
    return out.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])


def read_cafef_zip(
    path: Path,
    symbols: list[str],
    start: str,
    end: str,
    price_multiplier: float,
) -> pd.DataFrame:
    selected = set(symbols)
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.lower().endswith(".csv"):
                continue
            with archive.open(info.filename) as handle:
                frame = pd.read_csv(handle, encoding="utf-8-sig")
            frame.columns = [column.strip("<>") for column in frame.columns]
            if "Ticker" not in frame.columns:
                continue
            frame["ticker"] = frame["Ticker"].astype(str).str.upper()
            frame = frame[frame["ticker"].isin(selected)]
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["DTYYYYMMDD"].astype(str), format="%Y%m%d", errors="coerce")
            frame = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
            if frame.empty:
                continue
            out = frame.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
            for column in PRICE_COLUMNS:
                out[column] = pd.to_numeric(out[column], errors="coerce") * price_multiplier
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
            out["value"] = np.nan
            out["source"] = "cafef_adjusted"
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "value", "source"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])


def _to_epoch(value: str, end_inclusive: bool = False) -> int:
    timestamp = pd.Timestamp(value)
    if end_inclusive:
        timestamp = timestamp + pd.Timedelta(days=1)
    dt = datetime(timestamp.year, timestamp.month, timestamp.day, tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_yahoo_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    params = (
        f"?period1={_to_epoch(start)}"
        f"&period2={_to_epoch(end, end_inclusive=True)}"
        "&interval=1d&events=history"
    )
    url = YAHOO_CHART_URL.format(symbol=symbol) + params
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    result = payload.get("chart", {}).get("result")
    if not result:
        return pd.DataFrame()

    result0 = result[0]
    timestamps = result0.get("timestamp") or []
    quote = (result0.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (result0.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Bangkok").date,
            "ticker": symbol,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": adjclose if adjclose is not None else quote.get("close"),
            "raw_close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = np.nan
    frame["source"] = "yahoo_adjclose"
    for column in ("open", "high", "low", "close", "raw_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "ticker", "close"])


def fetch_yahoo(symbols: list[str], start: str, end: str, request_pause: float) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for index, symbol in enumerate(symbols, start=1):
        print(f"[Yahoo {index:03d}/{len(symbols)}] {symbol}")
        try:
            frame = fetch_yahoo_symbol(symbol, start, end)
            if frame.empty:
                failures.append({"ticker": symbol, "error": "empty"})
            else:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - keep source audit progressing.
            failures.append({"ticker": symbol, "error": str(exc)})
        time.sleep(request_pause)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "value", "source"]), failures
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]), failures


def coverage_summary(frame: pd.DataFrame, source: str, symbols: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {
            "source": source,
            "symbols_requested": len(symbols),
            "symbols_with_data": 0,
            "rows": 0,
            "first_date": None,
            "last_date": None,
        }
    return {
        "source": source,
        "symbols_requested": len(symbols),
        "symbols_with_data": int(frame["ticker"].nunique()),
        "rows": int(len(frame)),
        "first_date": pd.Timestamp(frame["date"].min()).date().isoformat(),
        "last_date": pd.Timestamp(frame["date"].max()).date().isoformat(),
        "median_rows_per_symbol": float(frame.groupby("ticker")["date"].nunique().median()),
        "min_rows_per_symbol": int(frame.groupby("ticker")["date"].nunique().min()),
        "max_rows_per_symbol": int(frame.groupby("ticker")["date"].nunique().max()),
    }


def compare_pair(base: pd.DataFrame, other: pd.DataFrame, base_name: str, other_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = base[["date", "ticker", "close", "volume"]].rename(
        columns={"close": f"close_{base_name}", "volume": f"volume_{base_name}"}
    )
    right = other[["date", "ticker", "close", "volume"]].rename(
        columns={"close": f"close_{other_name}", "volume": f"volume_{other_name}"}
    )
    merged = left.merge(right, on=["date", "ticker"], how="inner")
    if merged.empty:
        return merged, {
            "pair": f"{base_name}_vs_{other_name}",
            "common_rows": 0,
            "common_symbols": 0,
        }

    close_a = merged[f"close_{base_name}"].astype(float)
    close_b = merged[f"close_{other_name}"].astype(float)
    diff = close_a - close_b
    pct_diff = diff / close_a.replace(0.0, np.nan)
    merged["close_abs_diff"] = diff.abs()
    merged["close_pct_diff_abs"] = pct_diff.abs()
    merged["close_pct_diff_signed"] = pct_diff

    summary = {
        "pair": f"{base_name}_vs_{other_name}",
        "common_rows": int(len(merged)),
        "common_symbols": int(merged["ticker"].nunique()),
        "first_common_date": pd.Timestamp(merged["date"].min()).date().isoformat(),
        "last_common_date": pd.Timestamp(merged["date"].max()).date().isoformat(),
        "median_abs_close_diff_vnd": float(merged["close_abs_diff"].median()),
        "p95_abs_close_diff_vnd": float(merged["close_abs_diff"].quantile(0.95)),
        "median_abs_pct_diff": float(merged["close_pct_diff_abs"].median()),
        "p95_abs_pct_diff": float(merged["close_pct_diff_abs"].quantile(0.95)),
        "exact_or_1dong_match_rate": float((merged["close_abs_diff"] <= 1.0).mean()),
        "within_0p1pct_rate": float((merged["close_pct_diff_abs"] <= 0.001).mean()),
        "within_1pct_rate": float((merged["close_pct_diff_abs"] <= 0.01).mean()),
    }
    return merged, summary


def add_daily_returns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])
    out["daily_return"] = out.groupby("ticker")["close"].pct_change(fill_method=None)
    return out.dropna(subset=["daily_return"])


def compare_return_pair(
    base: pd.DataFrame,
    other: pd.DataFrame,
    base_name: str,
    other_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left_returns = add_daily_returns(base)
    right_returns = add_daily_returns(other)
    left = left_returns[["date", "ticker", "daily_return"]].rename(
        columns={"daily_return": f"return_{base_name}"}
    )
    right = right_returns[["date", "ticker", "daily_return"]].rename(
        columns={"daily_return": f"return_{other_name}"}
    )
    merged = left.merge(right, on=["date", "ticker"], how="inner")
    if merged.empty:
        return merged, {
            "pair": f"{base_name}_vs_{other_name}",
            "common_return_rows": 0,
            "common_symbols": 0,
        }

    diff = merged[f"return_{base_name}"].astype(float) - merged[f"return_{other_name}"].astype(float)
    merged["abs_return_diff"] = diff.abs()
    summary = {
        "pair": f"{base_name}_vs_{other_name}",
        "common_return_rows": int(len(merged)),
        "common_symbols": int(merged["ticker"].nunique()),
        "first_common_date": pd.Timestamp(merged["date"].min()).date().isoformat(),
        "last_common_date": pd.Timestamp(merged["date"].max()).date().isoformat(),
        "return_corr": float(merged[[f"return_{base_name}", f"return_{other_name}"]].corr().iloc[0, 1]),
        "median_abs_return_diff": float(merged["abs_return_diff"].median()),
        "p95_abs_return_diff": float(merged["abs_return_diff"].quantile(0.95)),
        "within_1bp_rate": float((merged["abs_return_diff"] <= 0.0001).mean()),
        "within_10bp_rate": float((merged["abs_return_diff"] <= 0.001).mean()),
        "within_100bp_rate": float((merged["abs_return_diff"] <= 0.01).mean()),
    }
    return merged, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, default=Path("dataset/hose_liquid_150_selection_report.csv"))
    parser.add_argument("--vndirect-long", type=Path, default=Path("dataset/hose_liquid_150_ohlcv_selected_long.csv"))
    parser.add_argument("--cafef-zip", type=Path, default=Path("dataset/cafef_upto_20260707.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/source_audit"))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means all selected symbols.")
    parser.add_argument("--request-pause", type=float, default=0.05)
    parser.add_argument("--cafef-price-multiplier", type=float, default=1000.0)
    parser.add_argument("--skip-yahoo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    symbols = selected_symbols(args.selection_report)
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    print(f"Auditing {len(symbols)} selected symbols from {args.start} to {args.end}")

    statuses = [
        SourceStatus("vndirect", "available", "Local dataset built from api-finfo stock_prices."),
        SourceStatus("cafef_adjusted", "available", f"Downloaded zip: {args.cafef_zip}"),
        SourceStatus("yahoo_adjclose", "available" if not args.skip_yahoo else "skipped", "Yahoo Finance chart .VN endpoint."),
        SourceStatus("ssi_fastconnect", "requires_auth", "DailyOhlc/DailyStockPrice returns Missing Authorization header without token."),
        SourceStatus("tcbs_public_old", "unavailable", "Old apipubaws bars-long-term endpoint returned 404 in this environment."),
    ]

    vndirect = read_vndirect(args.vndirect_long, symbols, args.start, args.end)
    cafef = read_cafef_zip(args.cafef_zip, symbols, args.start, args.end, args.cafef_price_multiplier)
    if args.skip_yahoo:
        yahoo = pd.DataFrame(columns=vndirect.columns)
        yahoo_failures: list[dict[str, str]] = []
    else:
        yahoo, yahoo_failures = fetch_yahoo(symbols, args.start, args.end, args.request_pause)

    source_frames = {
        "vndirect": vndirect,
        "cafef_adjusted": cafef,
        "yahoo_adjclose": yahoo,
    }
    for source_name, frame in source_frames.items():
        if frame.empty:
            continue
        frame.to_csv(args.output_dir / f"{source_name}_long.csv", index=False)

    coverage = [coverage_summary(frame, name, symbols) for name, frame in source_frames.items()]
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(args.output_dir / "coverage_summary.csv", index=False)

    pair_summaries = []
    pair_frames = []
    for left_name, right_name in (
        ("vndirect", "cafef_adjusted"),
        ("vndirect", "yahoo_adjclose"),
        ("cafef_adjusted", "yahoo_adjclose"),
    ):
        merged, summary = compare_pair(source_frames[left_name], source_frames[right_name], left_name, right_name)
        pair_summaries.append(summary)
        if not merged.empty:
            merged.insert(0, "pair", f"{left_name}_vs_{right_name}")
            pair_frames.append(merged)

    pair_summary_frame = pd.DataFrame(pair_summaries)
    pair_summary_frame.to_csv(args.output_dir / "pairwise_summary.csv", index=False)
    if pair_frames:
        pairwise_rows = pd.concat(pair_frames, ignore_index=True)
        pairwise_rows.to_csv(args.output_dir / "pairwise_common_rows.csv", index=False)
        worst = pairwise_rows.sort_values("close_pct_diff_abs", ascending=False).head(200)
        worst.to_csv(args.output_dir / "pairwise_worst_differences.csv", index=False)

    return_summaries = []
    return_frames = []
    for left_name, right_name in (
        ("vndirect", "cafef_adjusted"),
        ("vndirect", "yahoo_adjclose"),
        ("cafef_adjusted", "yahoo_adjclose"),
    ):
        merged, summary = compare_return_pair(source_frames[left_name], source_frames[right_name], left_name, right_name)
        return_summaries.append(summary)
        if not merged.empty:
            merged.insert(0, "pair", f"{left_name}_vs_{right_name}")
            return_frames.append(merged)

    return_summary_frame = pd.DataFrame(return_summaries)
    return_summary_frame.to_csv(args.output_dir / "pairwise_return_summary.csv", index=False)
    if return_frames:
        return_rows = pd.concat(return_frames, ignore_index=True)
        return_rows.to_csv(args.output_dir / "pairwise_return_common_rows.csv", index=False)
        return_worst = return_rows.sort_values("abs_return_diff", ascending=False).head(200)
        return_worst.to_csv(args.output_dir / "pairwise_worst_return_differences.csv", index=False)

    status_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": symbols,
        "sources": [asdict(status) for status in statuses],
        "yahoo_failures": yahoo_failures,
        "outputs": {
            "coverage_summary": str(args.output_dir / "coverage_summary.csv"),
            "pairwise_summary": str(args.output_dir / "pairwise_summary.csv"),
            "pairwise_common_rows": str(args.output_dir / "pairwise_common_rows.csv"),
            "pairwise_worst_differences": str(args.output_dir / "pairwise_worst_differences.csv"),
            "pairwise_return_summary": str(args.output_dir / "pairwise_return_summary.csv"),
            "pairwise_return_common_rows": str(args.output_dir / "pairwise_return_common_rows.csv"),
            "pairwise_worst_return_differences": str(args.output_dir / "pairwise_worst_return_differences.csv"),
        },
    }
    with (args.output_dir / "source_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status_payload, handle, indent=2, ensure_ascii=False)

    print("\nCoverage summary:")
    print(coverage_frame.to_string(index=False))
    print("\nPairwise summary:")
    print(pair_summary_frame.to_string(index=False))
    print("\nPairwise return summary:")
    print(return_summary_frame.to_string(index=False))
    print(f"\nWrote source audit outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
