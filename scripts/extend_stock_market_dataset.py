"""Extend the legacy HOSE wide price dataset using anchored VNDIRECT returns.

The legacy file appears to contain adjusted prices from a source that does not
match CafeF or VNDIRECT absolute prices for every ticker. To avoid artificial
level jumps, this script uses VNDIRECT close-to-close returns and anchors each
ticker to the first/last observed legacy price.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COLUMN = "History Price / Date"
VN_DIRECT_URL = "https://dchart-api.vndirect.com.vn/dchart/history"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://dchart.vndirect.com.vn/",
}


def parse_price(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).replace(",", "").strip()
    if not text:
        return np.nan
    return float(text)


def format_date(value: pd.Timestamp) -> str:
    return f"{value.month}/{value.day}/{value.year}"


def format_price(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{int(round(number)):,}"


def to_epoch_seconds(value: pd.Timestamp) -> int:
    dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_vndirect_close(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int = 3,
    pause_seconds: float = 0.8,
) -> pd.Series:
    params = {
        "resolution": "D",
        "symbol": symbol,
        "from": str(to_epoch_seconds(start)),
        "to": str(to_epoch_seconds(end + pd.Timedelta(days=1))),
    }
    url = f"{VN_DIRECT_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=HEADERS)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if payload.get("s") != "ok" or not payload.get("t"):
                return pd.Series(dtype=float, name=symbol)

            dates = pd.to_datetime(
                [datetime.fromtimestamp(ts, timezone.utc).date() for ts in payload["t"]]
            )
            closes = pd.to_numeric(pd.Series(payload["c"], dtype=float), errors="coerce") * 1000.0
            series = pd.Series(closes.to_numpy(), index=dates, name=symbol).sort_index()
            return series[~series.index.duplicated(keep="last")]
        except Exception as exc:  # noqa: BLE001 - retry network/API failures.
            last_error = exc
            time.sleep(pause_seconds * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {symbol} from VNDIRECT after {retries} attempts") from last_error


def build_extended_prices(
    old_path: Path,
    prepend_start: pd.Timestamp,
    append_end: pd.Timestamp,
    request_pause: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_raw = pd.read_csv(old_path, dtype=str).fillna("")
    if DATE_COLUMN not in old_raw.columns:
        raise ValueError(f"Expected '{DATE_COLUMN}' in {old_path}")

    tickers = [column for column in old_raw.columns if column != DATE_COLUMN]
    old_dates = pd.to_datetime(old_raw[DATE_COLUMN], errors="coerce")
    if old_dates.isna().any():
        bad_rows = old_raw.loc[old_dates.isna(), DATE_COLUMN].head().tolist()
        raise ValueError(f"Could not parse some legacy dates: {bad_rows}")

    old_raw = old_raw.copy()
    old_raw["_Date"] = old_dates
    old_raw = old_raw.sort_values("_Date").reset_index(drop=True)
    old_first = old_raw["_Date"].iloc[0]
    old_last = old_raw["_Date"].iloc[-1]

    source_start = min(prepend_start, old_first)
    source_end = max(append_end, old_last)

    before_frames: list[pd.Series] = []
    after_frames: list[pd.Series] = []
    report_rows: list[dict[str, object]] = []

    old_numeric = old_raw.set_index("_Date")[tickers].map(parse_price)

    for ticker in tickers:
        source = fetch_vndirect_close(ticker, source_start, source_end)
        time.sleep(request_pause)

        old_first_price = old_numeric[ticker].dropna().iloc[0]
        old_first_date = old_numeric[ticker].dropna().index[0]
        old_last_price = old_numeric[ticker].dropna().iloc[-1]
        old_last_date = old_numeric[ticker].dropna().index[-1]

        if old_first_date not in source.index:
            raise ValueError(f"{ticker}: missing VNDIRECT anchor date {old_first_date.date()}")
        if old_last_date not in source.index:
            raise ValueError(f"{ticker}: missing VNDIRECT anchor date {old_last_date.date()}")

        first_anchor = source.loc[old_first_date]
        last_anchor = source.loc[old_last_date]
        before = source[(source.index >= prepend_start) & (source.index < old_first_date)].copy()
        after = source[(source.index > old_last_date) & (source.index <= append_end)].copy()

        before = before * (old_first_price / first_anchor)
        after = after * (old_last_price / last_anchor)
        before.name = ticker
        after.name = ticker
        before_frames.append(before)
        after_frames.append(after)

        report_rows.append(
            {
                "ticker": ticker,
                "legacy_first_date": old_first_date.date().isoformat(),
                "legacy_last_date": old_last_date.date().isoformat(),
                "legacy_first_price": old_first_price,
                "legacy_last_price": old_last_price,
                "vndirect_first_anchor": first_anchor,
                "vndirect_last_anchor": last_anchor,
                "prepend_rows": int(before.notna().sum()),
                "append_rows": int(after.notna().sum()),
                "first_scale": old_first_price / first_anchor,
                "last_scale": old_last_price / last_anchor,
            }
        )

    before_df = pd.concat(before_frames, axis=1) if before_frames else pd.DataFrame(columns=tickers)
    after_df = pd.concat(after_frames, axis=1) if after_frames else pd.DataFrame(columns=tickers)

    old_values = old_raw.set_index("_Date")[tickers].copy()
    combined_index = before_df.index.union(old_values.index).union(after_df.index).sort_values()
    combined = pd.DataFrame(index=combined_index, columns=tickers, dtype=object)

    for ticker in tickers:
        combined.loc[before_df.index, ticker] = before_df[ticker].map(format_price)
        combined.loc[old_values.index, ticker] = old_values[ticker]
        combined.loc[after_df.index, ticker] = after_df[ticker].map(format_price)

    combined = combined.fillna("")
    combined.insert(0, DATE_COLUMN, [format_date(date) for date in combined.index])
    combined = combined.reset_index(drop=True)

    numeric_prices = combined.set_index(pd.to_datetime(combined[DATE_COLUMN]))[tickers].map(parse_price)
    report = pd.DataFrame(report_rows)
    return combined, numeric_prices, report


def write_daily_returns(prices: pd.DataFrame, output_path: Path) -> None:
    returns = prices.pct_change(fill_method=None)
    out = returns.reset_index(names=DATE_COLUMN)
    out[DATE_COLUMN] = out[DATE_COLUMN].map(format_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def write_weekly_min_max(prices: pd.DataFrame, output_path: Path) -> None:
    weekly_groups = prices.resample("W-FRI")
    weekly_open = weekly_groups.first()
    weekly_min = weekly_groups.min()
    weekly_max = weekly_groups.max()
    weekly_min_return = (weekly_min / weekly_open) - 1.0
    weekly_max_return = (weekly_max / weekly_open) - 1.0
    weekly_out = pd.concat({"Min Return": weekly_min_return, "Max Return": weekly_max_return}, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weekly_out.to_csv(output_path, index_label="Week End")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("dataset/stock_market_19_24.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/stock_market_18_26_vndirect_anchored.csv"),
    )
    parser.add_argument(
        "--daily-return-output",
        type=Path,
        default=Path("dataset/daily_return_18_26_vndirect_anchored.csv"),
    )
    parser.add_argument(
        "--weekly-output",
        type=Path,
        default=Path("dataset/weekly_min_max_return_18_26_vndirect_anchored.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("dataset/stock_market_18_26_vndirect_anchored_report.csv"),
    )
    parser.add_argument("--prepend-start", default="2018-09-05")
    parser.add_argument("--append-end", default=datetime.now().date().isoformat())
    parser.add_argument("--request-pause", type=float, default=0.5)
    args = parser.parse_args()

    prepend_start = pd.Timestamp(args.prepend_start)
    append_end = pd.Timestamp(args.append_end)

    combined, numeric_prices, report = build_extended_prices(
        old_path=args.old,
        prepend_start=prepend_start,
        append_end=append_end,
        request_pause=args.request_pause,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    write_daily_returns(numeric_prices, args.daily_return_output)
    write_weekly_min_max(numeric_prices, args.weekly_output)
    report.to_csv(args.report_output, index=False)

    print(f"Wrote extended prices: {args.output} ({combined.shape[0]} rows, {combined.shape[1] - 1} tickers)")
    print(f"Wrote daily returns: {args.daily_return_output}")
    print(f"Wrote weekly min/max returns: {args.weekly_output}")
    print(f"Wrote anchor report: {args.report_output}")
    print(f"Date range: {combined[DATE_COLUMN].iloc[0]} -> {combined[DATE_COLUMN].iloc[-1]}")


if __name__ == "__main__":
    main()
