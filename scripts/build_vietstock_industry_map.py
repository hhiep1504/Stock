"""Build a Vietstock sector/industry map for HOSE benchmark stocks.

VietstockFinance exposes a Corporate A-Z JSON endpoint and company detail pages
with GICS-style industry variables. This script fetches the selected HOSE100
symbols, resolves their Vietstock detail URLs, and writes a sector/industry CSV
that can be used directly by the graph loader.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_URL = "https://finance.vietstock.vn"
CORPORATE_AZ_URL = f"{BASE_URL}/data/corporateaz"
DATE_COLUMN = "History Price / Date"
BUSINESS_TYPE_IDS = {
    "non_financial": 1,
    "securities": 2,
    "bank": 3,
    "insurance": 5,
}
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def request_text(url: str, headers: dict[str, str] | None = None) -> tuple[str, str]:
    request = urllib.request.Request(url, headers=headers or HEADERS)
    with urllib.request.urlopen(request, timeout=40) as response:
        cookie_headers = response.headers.get_all("Set-Cookie") or []
        cookies = "; ".join(cookie.split(";", 1)[0] for cookie in cookie_headers)
        return response.read().decode("utf-8", errors="ignore"), cookies


def get_request_token() -> tuple[str, str, str]:
    page_url = f"{BASE_URL}/doanh-nghiep-a-z/doanh-nghiep-phi-tai-chinh?languageid=2"
    page, cookies = request_text(page_url)
    match = re.search(r"name=__RequestVerificationToken type=hidden value=([^>\s]+)", page)
    if not match:
        raise RuntimeError("Could not find Vietstock request verification token.")
    return match.group(1), cookies, page_url


def post_corporate_az(
    token: str,
    cookies: str,
    referer: str,
    business_type_id: int,
    page: int,
    page_size: int,
    code: str = "",
) -> list[dict[str, Any]]:
    payload = {
        "catID": "",
        "industryID": "",
        "page": page,
        "pageSize": page_size,
        "type": 0,
        "code": code,
        "businessTypeID": business_type_id,
        "orderBy": "Code",
        "orderDir": "ASC",
        "__RequestVerificationToken": token,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": referer,
        "Cookie": cookies,
    }
    request = urllib.request.Request(
        CORPORATE_AZ_URL,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def fetch_corporate_directory(page_size: int = 50, pause: float = 0.15) -> pd.DataFrame:
    token, cookies, referer = get_request_token()
    rows: list[dict[str, Any]] = []

    for business_type_name, business_type_id in BUSINESS_TYPE_IDS.items():
        page = 1
        total_pages = 1
        while page <= total_pages:
            data = post_corporate_az(
                token=token,
                cookies=cookies,
                referer=referer,
                business_type_id=business_type_id,
                page=page,
                page_size=page_size,
            )
            if not data:
                break
            total_records = int(data[0].get("TotalRecord") or len(data))
            total_pages = max(1, (total_records + page_size - 1) // page_size)
            for item in data:
                item["BusinessType"] = business_type_name
                rows.append(item)
            page += 1
            time.sleep(pause)

    directory = pd.DataFrame(rows)
    if directory.empty:
        raise RuntimeError("Vietstock Corporate A-Z returned no rows.")
    directory["Code"] = directory["Code"].astype(str).str.upper().str.strip()
    return directory.drop_duplicates(["Code", "Exchange", "URL"]).sort_values("Code").reset_index(drop=True)


def resolve_symbol_directory_rows(
    symbols: list[str],
    existing_directory: pd.DataFrame,
    pause: float = 0.15,
) -> pd.DataFrame:
    token, cookies, referer = get_request_token()
    rows: list[dict[str, Any]] = []
    existing_codes = set(existing_directory["Code"].astype(str).str.upper()) if not existing_directory.empty else set()
    missing_symbols = [symbol for symbol in symbols if symbol not in existing_codes]

    for symbol in missing_symbols:
        for business_type_name, business_type_id in BUSINESS_TYPE_IDS.items():
            data = post_corporate_az(
                token=token,
                cookies=cookies,
                referer=referer,
                business_type_id=business_type_id,
                page=1,
                page_size=50,
                code=symbol,
            )
            for item in data:
                if str(item.get("Code") or "").upper() != symbol:
                    continue
                item["BusinessType"] = business_type_name
                rows.append(item)
        time.sleep(pause)

    if not rows:
        return existing_directory
    resolved = pd.DataFrame(rows)
    resolved["Code"] = resolved["Code"].astype(str).str.upper().str.strip()
    combined = pd.concat([existing_directory, resolved], ignore_index=True)
    return combined.drop_duplicates(["Code", "Exchange", "URL"]).sort_values("Code").reset_index(drop=True)


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


def decode_js_string(value: str) -> str:
    value = value.replace('\\"', '"')
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return html.unescape(value)


def extract_vietstock_industry(page: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for level in (1, 2, 3, 4):
        code_match = re.search(rf"_gicsLevel{level}\s*=\s*([^;]+);", page)
        name_match = re.search(rf'_gicsNameLevel{level}\s*=\s*"((?:\\.|[^"])*)"', page)
        output[f"gics_level{level}"] = str(code_match.group(1)).split("||", 1)[0].strip() if code_match else ""
        output[f"gics_name_level{level}"] = decode_js_string(name_match.group(1)).strip() if name_match else ""
    return output


def choose_industry(row: pd.Series) -> str:
    for column in ("gics_name_level3", "gics_name_level2", "gics_name_level1"):
        value = str(row.get(column) or "").strip()
        if value:
            return value
    fallback = str(row.get("IndustryName") or "").strip()
    return fallback if fallback else "Unknown"


def write_config(source_config: Path, output_config: Path, sector_map_file: Path, graph_mode: str) -> None:
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
        default=Path("dataset/hose_liquid_100_strict_sector_industry_vietstock.csv"),
    )
    parser.add_argument(
        "--directory-cache",
        type=Path,
        default=Path("dataset/vietstock_corporate_az_raw.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/hose_liquid_100_strict_sector_industry_vietstock_report.json"),
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--request-pause", type=float, default=0.2)
    parser.add_argument("--base-config", type=Path, default=Path("configs/hose_liquid_100_strict_hybrid.json"))
    parser.add_argument(
        "--hybrid-config",
        type=Path,
        default=Path("configs/hose_liquid_100_strict_hybrid_vietstock_industry.json"),
    )
    parser.add_argument(
        "--static-config",
        type=Path,
        default=Path("configs/hose_liquid_100_strict_static_vietstock_industry.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = selected_symbols(args.selection)
    selected = pd.DataFrame({"ticker": symbols, "selection_rank": range(1, len(symbols) + 1)})

    if args.use_cache:
        if not args.directory_cache.exists():
            raise FileNotFoundError(f"Vietstock directory cache not found: {args.directory_cache}")
        directory = pd.read_csv(args.directory_cache)
    else:
        directory = fetch_corporate_directory()
        directory = resolve_symbol_directory_rows(symbols, directory, pause=args.request_pause)
        args.directory_cache.parent.mkdir(parents=True, exist_ok=True)
        directory.to_csv(args.directory_cache, index=False)

    directory["Code"] = directory["Code"].astype(str).str.upper().str.strip()
    hose_directory = directory[directory["Exchange"].astype(str).str.upper().eq("HOSE")]
    lookup = hose_directory.drop_duplicates("Code").set_index("Code")

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        if symbol not in lookup.index:
            rows.append({"ticker": symbol, "coverage_status": "missing_directory"})
            continue
        item = lookup.loc[symbol].to_dict()
        url = str(item.get("URL") or "")
        detail_url = url if "languageid=" in url else f"{url}?languageid=2"
        try:
            page, _ = request_text(detail_url)
            industry = extract_vietstock_industry(page)
            coverage_status = "matched"
        except Exception as exc:  # noqa: BLE001 - keep partial output for audit.
            industry = {}
            coverage_status = f"detail_error: {exc}"
        rows.append(
            {
                "ticker": symbol,
                "vietstock_url": detail_url,
                "vietstock_name": item.get("Name"),
                "exchange": item.get("Exchange"),
                "business_type": item.get("BusinessType"),
                "vietstock_industry_name": item.get("IndustryName"),
                "coverage_status": coverage_status,
                **industry,
            }
        )
        time.sleep(args.request_pause)

    merged = selected.merge(pd.DataFrame(rows), on="ticker", how="left")
    for column in ("gics_name_level1", "gics_name_level2", "gics_name_level3", "gics_name_level4"):
        if column not in merged.columns:
            merged[column] = ""
        merged[column] = merged[column].fillna("")
    merged["sector"] = merged["gics_name_level1"].replace("", "Unknown")
    merged["industry_group"] = merged["gics_name_level2"].replace("", "Unknown")
    merged["industry"] = merged.apply(choose_industry, axis=1)
    merged["sub_industry"] = merged["gics_name_level4"].replace("", "Unknown")
    merged["sector_source"] = "VietstockFinance"
    merged["sector_standard"] = "Vietstock GICS-style sector hierarchy"

    output_columns = [
        "ticker",
        "sector",
        "industry_group",
        "industry",
        "sub_industry",
        "sector_source",
        "sector_standard",
        "coverage_status",
        "selection_rank",
        "vietstock_url",
        "vietstock_name",
        "exchange",
        "business_type",
        "vietstock_industry_name",
        "gics_level1",
        "gics_name_level1",
        "gics_level2",
        "gics_name_level2",
        "gics_level3",
        "gics_name_level3",
        "gics_level4",
        "gics_name_level4",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged[output_columns].to_csv(args.output, index=False)

    if args.base_config.exists():
        write_config(args.base_config, args.hybrid_config, args.output, "hybrid")
        write_config(args.base_config, args.static_config, args.output, "static")

    matched = merged["coverage_status"].eq("matched")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "https://finance.vietstock.vn",
        "selected_symbols": len(symbols),
        "matched_symbols": int(matched.sum()),
        "missing_symbols": merged.loc[~matched, ["ticker", "coverage_status"]].to_dict("records"),
        "sector_counts": merged["sector"].value_counts().to_dict(),
        "industry_group_counts": merged["industry_group"].value_counts().to_dict(),
        "industry_counts": merged["industry"].value_counts().to_dict(),
        "sub_industry_counts": merged["sub_industry"].value_counts().to_dict(),
        "static_relation_column": "industry",
        "outputs": {
            "sector_map": str(args.output),
            "directory_cache": str(args.directory_cache),
            "hybrid_config": str(args.hybrid_config),
            "static_config": str(args.static_config),
        },
        "note": "Use industry for static graph construction; sub_industry is retained for finer audit.",
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
