import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_price_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = "History Price / Date"
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    return df


def transform_series(price: pd.Series, method: str) -> pd.Series:
    if method == "log_return":
        return np.log(price / price.shift(1))
    if method == "diff":
        return price.diff()
    raise ValueError("method must be one of: log_return, diff")


def minmax_past_only(x: pd.Series, window: int | None = None) -> pd.Series:
    # Use only historical values by shifting bounds by 1 step.
    if window is None:
        x_min = x.expanding(min_periods=2).min().shift(1)
        x_max = x.expanding(min_periods=2).max().shift(1)
    else:
        x_min = x.rolling(window=window, min_periods=max(2, window // 5)).min().shift(1)
        x_max = x.rolling(window=window, min_periods=max(2, window // 5)).max().shift(1)

    denom = x_max - x_min
    x_norm = (x - x_min) / denom

    # Avoid inf values when denominator is zero.
    x_norm = x_norm.where(np.isfinite(x_norm), np.nan)
    return x_norm


def plot_normalization_view(
    date_index: pd.Index,
    raw_close: pd.Series,
    transformed: pd.Series,
    normalized: pd.Series,
    ticker: str,
    method: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(date_index, raw_close, color="#1f77b4", linewidth=1.2)
    axes[0].set_title(f"Raw Close Price - {ticker}")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.2)

    axes[1].plot(date_index, transformed, color="#ff7f0e", linewidth=1.0)
    axes[1].set_title(f"Transformed Series ({method})")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.2)

    axes[2].plot(date_index, normalized, color="#2ca02c", linewidth=1.0)
    axes[2].axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    axes[2].axhline(1.0, color="gray", linewidth=0.8, alpha=0.6)
    axes[2].set_title("Leakage-safe Min-Max Normalized")
    axes[2].set_ylabel("x_norm")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize close-price derived features with past-only Min-Max scaling and visualize the effect."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/stock_market_19_24.csv"),
        help="Path to daily close price CSV",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="VHM",
        help="Ticker symbol column to visualize",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="log_return",
        choices=["log_return", "diff"],
        help="Transformation before normalization",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Rolling window for past-only min/max. Use 0 for expanding mode.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dataset/figs"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    df = load_price_data(args.input)
    if args.ticker not in df.columns:
        raise ValueError(f"Ticker {args.ticker} not found. Available tickers: {', '.join(df.columns)}")

    raw = df[args.ticker].dropna()
    transformed = transform_series(raw, args.method)
    window = None if args.window <= 0 else args.window
    normalized = minmax_past_only(transformed, window=window)

    out_table = pd.DataFrame(
        {
            "raw_close": raw,
            "transformed": transformed,
            "x_norm": normalized,
        }
    )

    table_path = args.outdir / f"normalized_{args.ticker}_{args.method}.csv"
    fig_path = args.outdir / f"normalized_{args.ticker}_{args.method}.png"
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_table.to_csv(table_path, index_label="date")

    plot_normalization_view(
        date_index=out_table.index,
        raw_close=out_table["raw_close"],
        transformed=out_table["transformed"],
        normalized=out_table["x_norm"],
        ticker=args.ticker,
        method=args.method,
        out_path=fig_path,
    )

    valid_ratio = out_table["x_norm"].notna().mean() * 100.0
    print(f"Saved normalized data: {table_path}")
    print(f"Saved visualization: {fig_path}")
    print(f"Valid normalized ratio: {valid_ratio:.2f}%")
    print("Formula used: x_norm = (x - x_min_past) / (x_max_past - x_min_past)")


if __name__ == "__main__":
    main()
