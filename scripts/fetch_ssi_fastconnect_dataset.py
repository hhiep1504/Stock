"""Fetch HOSE-Liquid symbols from SSI FastConnect Data.

Credentials are read from an env-style file:

    SSI_CONSUMER_ID=...
    SSI_CONSUMER_SECRET=...

or, alternatively:

    SSI_ACCESS_TOKEN=...

The script uses FastConnect Data only. It does not require SSI trading account
passwords, PINs, OTPs, or order-entry permissions.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SSI_BASE_URL = "https://fc-data.ssi.com.vn/api/v2/Market"
DATE_COLUMN = "History Price / Date"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Content-Type": "application/json",
}


@dataclass(slots=True)
class Credentials:
    consumer_id: str | None = None
    consumer_secret: str | None = None
    access_token: str | None = None


def read_env_file(path: Path) -> Credentials:
    values: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(
            f"Credential file not found: {path}. Copy configs/ssi_fastconnect.env.example "
            "to configs/ssi_fastconnect.env and fill SSI_CONSUMER_ID/SSI_CONSUMER_SECRET."
        )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return Credentials(
        consumer_id=values.get("SSI_CONSUMER_ID") or values.get("consumerID"),
        consumer_secret=values.get("SSI_CONSUMER_SECRET") or values.get("consumerSecret"),
        access_token=values.get("SSI_ACCESS_TOKEN") or values.get("accessToken"),
    )


def selected_symbols(path: Path, max_symbols: int = 0) -> list[str]:
    report = pd.read_csv(path)
    selected = report[report["selected"] == True].sort_values("selection_rank")  # noqa: E712
    if selected.empty:
        raise ValueError(f"No selected symbols found in {path}")
    symbols = selected["code"].astype(str).str.upper().tolist()
    return symbols[:max_symbols] if max_symbols > 0 else symbols


def request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 40,
) -> dict[str, Any]:
    request_headers = {**HEADERS, **(headers or {})}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    return json.loads(text)


def get_access_token(credentials: Credentials) -> str:
    if credentials.access_token:
        return credentials.access_token
    if not credentials.consumer_id or not credentials.consumer_secret:
        raise ValueError("Provide SSI_CONSUMER_ID and SSI_CONSUMER_SECRET, or SSI_ACCESS_TOKEN.")

    payload = {
        "consumerID": credentials.consumer_id,
        "consumerSecret": credentials.consumer_secret,
    }
    data = request_json(f"{SSI_BASE_URL}/AccessToken", method="POST", payload=payload)
    token = (
        data.get("accessToken")
        or data.get("data", {}).get("accessToken")
        or data.get("Data", {}).get("AccessToken")
    )
    if not token:
        raise RuntimeError(f"SSI AccessToken response did not contain a token. Response keys: {list(data.keys())}")
    return str(token)


class SsiClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.auth_header_value = f"Bearer {access_token}"

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{SSI_BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
        try:
            return request_json(url, headers={"Authorization": self.auth_header_value})
        except urllib.error.HTTPError as exc:
            if exc.code != 401 or self.auth_header_value == self.access_token:
                raise

        # Some FastConnect client examples use the raw token in Authorization.
        # Retry once with that form if Bearer is rejected.
        self.auth_header_value = self.access_token
        return request_json(url, headers={"Authorization": self.auth_header_value})


def to_ssi_date(value: str) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%d/%m/%Y")


def parse_number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "None", "nan"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def extract_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    rows = payload.get("data") or payload.get("Data") or []
    total = payload.get("totalRecord") or payload.get("TotalRecord")
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("Data") or []
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None
    return list(rows), total_int


def fetch_daily_stock_price(
    client: SsiClient,
    symbol: str,
    start: str,
    end: str,
    market: str,
    page_size: int,
    max_pages: int,
    request_pause: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total_record: int | None = None
    for page_index in range(1, max_pages + 1):
        payload = client.get(
            "DailyStockPrice",
            {
                "Symbol": symbol,
                "market": market,
                "fromDate": to_ssi_date(start),
                "toDate": to_ssi_date(end),
                "pageIndex": page_index,
                "pageSize": page_size,
            },
        )
        rows, total_record = extract_rows(payload)
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        fetched = sum(len(frame) for frame in frames)
        if total_record is not None and fetched >= total_record:
            break
        time.sleep(request_pause)

    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["ticker"] = symbol
    return frame


def normalise_daily_stock_price(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    date_col = "Tradingdate" if "Tradingdate" in out.columns else "TradingDate"
    symbol_col = "Symbol" if "Symbol" in out.columns else "ticker"
    out["date"] = pd.to_datetime(out[date_col], dayfirst=True, errors="coerce")
    out["ticker"] = out[symbol_col].astype(str).str.upper()
    out["open"] = out.get("Openprice", np.nan).map(parse_number)
    out["high"] = out.get("Highestprice", np.nan).map(parse_number)
    out["low"] = out.get("Lowestprice", np.nan).map(parse_number)
    out["close"] = out.get("Closeprice", np.nan).map(parse_number)
    out["close_adjusted"] = out.get("Closepriceadjusted", out.get("Closeprice", np.nan)).map(parse_number)
    out["volume"] = out.get("Totalmatchvol", out.get("Totaltradedvol", np.nan)).map(parse_number)
    out["value"] = out.get("Totalmatchval", out.get("Totaltradedvalue", np.nan)).map(parse_number)
    out["source"] = "ssi_fastconnect_daily_stock_price"

    columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "close_adjusted",
        "volume",
        "value",
        "source",
    ]
    return out[columns].dropna(subset=["date", "ticker", "close_adjusted"]).sort_values(["ticker", "date"])


def format_date(value: pd.Timestamp) -> str:
    return f"{value.month}/{value.day}/{value.year}"


def write_wide(frame: pd.DataFrame, symbols: list[str], output_path: Path, value_column: str) -> pd.DataFrame:
    wide = frame.pivot_table(index="date", columns="ticker", values=value_column, aggfunc="last")
    wide = wide.reindex(columns=symbols).sort_index()
    output = wide.reset_index()
    output.insert(0, DATE_COLUMN, [format_date(pd.Timestamp(value)) for value in output["date"]])
    output = output.drop(columns=["date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return wide


def write_daily_return(wide: pd.DataFrame, output_path: Path) -> None:
    returns = wide.pct_change(fill_method=None).reset_index()
    returns.insert(0, DATE_COLUMN, [format_date(pd.Timestamp(value)) for value in returns["date"]])
    returns = returns.drop(columns=["date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("configs/ssi_fastconnect.env"))
    parser.add_argument("--selection-report", type=Path, default=Path("dataset/hose_liquid_150_selection_report.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/source_audit"))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--market", default="HOSE")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--request-pause", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    credentials = read_env_file(args.env_file)
    token = get_access_token(credentials)
    client = SsiClient(token)
    symbols = selected_symbols(args.selection_report, args.max_symbols)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for index, symbol in enumerate(symbols, start=1):
        print(f"[SSI {index:03d}/{len(symbols)}] {symbol}")
        try:
            raw = fetch_daily_stock_price(
                client=client,
                symbol=symbol,
                start=args.start,
                end=args.end,
                market=args.market,
                page_size=args.page_size,
                max_pages=args.max_pages,
                request_pause=args.request_pause,
            )
            normalised = normalise_daily_stock_price(raw)
            if normalised.empty:
                failures.append({"ticker": symbol, "error": "empty"})
            else:
                frames.append(normalised)
        except Exception as exc:  # noqa: BLE001 - continue fetching other symbols.
            failures.append({"ticker": symbol, "error": str(exc)})
        time.sleep(args.request_pause)

    if not frames:
        raise RuntimeError(f"SSI returned no usable rows. Failures: {failures[:5]}")

    long_frame = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    long_path = args.output_dir / "ssi_fastconnect_long.csv"
    wide_path = args.output_dir / "ssi_fastconnect_close_adjusted_wide.csv"
    return_path = args.output_dir / "ssi_fastconnect_daily_return_wide.csv"
    manifest_path = args.output_dir / "ssi_fastconnect_manifest.json"

    long_frame.to_csv(long_path, index=False)
    wide = write_wide(long_frame, symbols, wide_path, "close_adjusted")
    write_daily_return(wide, return_path)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "SSI FastConnect Data DailyStockPrice",
        "start": args.start,
        "end": args.end,
        "market": args.market,
        "symbols_requested": len(symbols),
        "symbols_with_data": int(long_frame["ticker"].nunique()),
        "rows": int(len(long_frame)),
        "first_date": pd.Timestamp(long_frame["date"].min()).date().isoformat(),
        "last_date": pd.Timestamp(long_frame["date"].max()).date().isoformat(),
        "failures": failures,
        "outputs": {
            "long": str(long_path),
            "wide": str(wide_path),
            "daily_return": str(return_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(f"Rows: {len(long_frame):,}; symbols: {long_frame['ticker'].nunique()}/{len(symbols)}")
    print(f"Date range: {manifest['first_date']} -> {manifest['last_date']}")
    print(f"Long: {long_path}")
    print(f"Wide adjusted close: {wide_path} ({wide.shape[0]} days x {wide.shape[1]} symbols)")
    print(f"Daily return: {return_path}")
    print(f"Manifest: {manifest_path}")
    if failures:
        print(f"Failures: {len(failures)}; see manifest for details.")


if __name__ == "__main__":
    main()
