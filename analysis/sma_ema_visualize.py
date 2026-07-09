import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def compute_sma_ema(close: pd.Series, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()
    out[f"ema_{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()
    return out


def plot_sma_ema(df: pd.DataFrame, ticker: str, window: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df.index, df["close"], linewidth=1.0, color="#1f77b4", label="Close")
    ax.plot(df.index, df[f"sma_{window}"], linewidth=1.2, color="#ff7f0e", label=f"SMA({window})")
    ax.plot(df.index, df[f"ema_{window}"], linewidth=1.2, color="#2ca02c", label=f"EMA({window})")

    ax.set_title(f"Close vs SMA/EMA - {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SMA and EMA from close price and visualize.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/stock_market_19_24.csv"),
        help="Path to daily close-price CSV",
    )
    parser.add_argument("--ticker", type=str, default="VHM", help="Ticker column")
    parser.add_argument("--window", type=int, default=20, help="Window size for SMA/EMA")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dataset/figs"),
        help="Output directory for CSV and chart",
    )
    args = parser.parse_args()

    if args.window < 2:
        raise ValueError("window must be >= 2")

    prices = load_price_data(args.input)
    if args.ticker not in prices.columns:
        raise ValueError(f"Ticker {args.ticker} not found.")

    close = prices[args.ticker].dropna()
    result = compute_sma_ema(close, args.window)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / f"sma_ema_{args.ticker}_{args.window}.csv"
    fig_path = args.outdir / f"sma_ema_{args.ticker}_{args.window}.png"

    result.to_csv(csv_path, index_label="date")
    plot_sma_ema(result, args.ticker, args.window, fig_path)

    print(f"Saved SMA/EMA table: {csv_path}")
    print(f"Saved SMA/EMA chart: {fig_path}")
    print(f"Rows with both indicators available: {result.dropna().shape[0]}")


if __name__ == "__main__":
    main()
