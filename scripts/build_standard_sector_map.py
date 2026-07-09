"""Build a standardized sector/industry map for HOSE benchmark stocks.

The previous sector map was inferred from VNDIRECT profile keywords. This
script replaces that with TradingView Vietnam scanner classifications, which
provide explicit sector and industry fields for Vietnamese listed equities.

For graph construction, the static relation is the `industry` column. This
avoids over-connecting broad sectors such as Finance while preserving an
international-style sector/industry hierarchy.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


TRADINGVIEW_SCAN_URL = "https://scanner.tradingview.com/vietnam/scan"
DATE_COLUMN = "History Price / Date"
SCANNER_COLUMNS = [
    "name",
    "description",
    "exchange",
    "sector",
    "industry",
    "type",
    "subtype",
    "market_cap_basic",
    "volume",
    "close",
]


def request_tradingview_scan(range_start: int = 0, range_end: int = 1000) -> dict[str, Any]:
    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "HOSE"},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "options": {"lang": "en"},
        "markets": ["vietnam"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": SCANNER_COLUMNS,
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [range_start, range_end],
    }
    request = urllib.request.Request(
        TRADINGVIEW_SCAN_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Referer": "https://www.tradingview.com/",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def fetch_tradingview_hose() -> pd.DataFrame:
    payload = request_tradingview_scan(0, 1000)
    rows = []
    for item in payload.get("data", []):
        values = item.get("d", [])
        row = {"tv_symbol": item.get("s")}
        for column, value in zip(SCANNER_COLUMNS, values):
            row[column] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("TradingView scanner returned no HOSE stocks.")
    frame["ticker"] = frame["name"].astype(str).str.upper().str.strip()
    frame = frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    return frame


def selected_symbols(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        symbols = payload.get("selected_symbols", [])
        if symbols:
            return [str(symbol).upper() for symbol in symbols]

    frame = pd.read_csv(path)
    if "selected" in frame.columns:
        frame = frame[frame["selected"] == True].sort_values("selection_rank")  # noqa: E712
    code_col = "code" if "code" in frame.columns else "ticker"
    return frame[code_col].astype(str).str.upper().tolist()


def normalise_industry(row: pd.Series) -> str:
    industry = str(row.get("industry") or "").strip()
    sector = str(row.get("sector") or "").strip()
    if industry and industry.lower() != "none":
        return industry
    if sector and sector.lower() != "none":
        return sector
    return "Unknown"


def write_config(
    source_config: Path,
    output_config: Path,
    sector_map_file: Path,
    graph_mode: str,
) -> None:
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    experiment_name = output_config.stem
    payload["experiment_name"] = experiment_name
    payload.setdefault("graph", {})
    payload["graph"]["graph_mode"] = graph_mode
    payload["graph"]["sector_map_file"] = str(sector_map_file).replace("\\", "/")
    payload["graph"]["use_static_graph"] = graph_mode != "dynamic"
    payload.setdefault("optuna", {})
    payload["optuna"]["study_name"] = experiment_name
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("dataset/hose_liquid_100_strict_manifest.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/hose_liquid_100_strict_sector_industry_tradingview.csv"),
    )
    parser.add_argument(
        "--scanner-cache",
        type=Path,
        default=Path("dataset/tradingview_hose_sector_industry_raw.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/hose_liquid_100_strict_sector_industry_report.json"),
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use the existing TradingView scanner cache instead of fetching a fresh snapshot.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/hose_liquid_100_strict_hybrid.json"),
    )
    parser.add_argument(
        "--hybrid-config",
        type=Path,
        default=Path("configs/hose_liquid_100_strict_hybrid_industry.json"),
    )
    parser.add_argument(
        "--static-config",
        type=Path,
        default=Path("configs/hose_liquid_100_strict_static_industry.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = selected_symbols(args.selection)
    if args.use_cache:
        if not args.scanner_cache.exists():
            raise FileNotFoundError(f"Scanner cache not found: {args.scanner_cache}")
        tv = pd.read_csv(args.scanner_cache)
    else:
        tv = fetch_tradingview_hose()
        args.scanner_cache.parent.mkdir(parents=True, exist_ok=True)
        tv.to_csv(args.scanner_cache, index=False)

    selected = pd.DataFrame({"ticker": symbols, "selection_rank": range(1, len(symbols) + 1)})
    merged = selected.merge(tv, on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["industry"] = merged["industry"].fillna("Unknown")
    merged["industry"] = merged.apply(normalise_industry, axis=1)
    merged["sector_source"] = "TradingView Vietnam scanner"
    merged["sector_standard"] = "TradingView sector/industry"
    merged["coverage_status"] = merged["tv_symbol"].notna().map({True: "matched", False: "missing"})

    output_columns = [
        "ticker",
        "sector",
        "industry",
        "sector_source",
        "sector_standard",
        "coverage_status",
        "selection_rank",
        "tv_symbol",
        "description",
        "exchange",
        "type",
        "subtype",
        "market_cap_basic",
        "volume",
        "close",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged[output_columns].to_csv(args.output, index=False)

    if args.base_config.exists():
        write_config(args.base_config, args.hybrid_config, args.output, "hybrid")
        write_config(args.base_config, args.static_config, args.output, "static")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": TRADINGVIEW_SCAN_URL,
        "selected_symbols": len(symbols),
        "matched_symbols": int((merged["coverage_status"] == "matched").sum()),
        "missing_symbols": merged.loc[merged["coverage_status"] != "matched", "ticker"].tolist(),
        "sector_counts": merged["sector"].value_counts().to_dict(),
        "industry_counts": merged["industry"].value_counts().to_dict(),
        "static_relation_column": "industry",
        "outputs": {
            "sector_map": str(args.output),
            "scanner_cache": str(args.scanner_cache),
            "hybrid_config": str(args.hybrid_config),
            "static_config": str(args.static_config),
        },
        "note": (
            "Use industry for static graph construction; it avoids dense "
            "sector-level Finance edges."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Matched {report['matched_symbols']}/{report['selected_symbols']} selected symbols.")
    print(f"Sector map: {args.output}")
    print(f"Hybrid config: {args.hybrid_config}")
    print(f"Static config: {args.static_config}")
    print("Sector counts:")
    print(merged["sector"].value_counts().to_string())
    print("\nTop industry counts:")
    print(merged["industry"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
