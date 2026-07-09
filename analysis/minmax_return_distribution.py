import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture


def load_daily_prices(csv_path: Path) -> pd.DataFrame:
    """Load daily close prices and convert all ticker columns to numeric."""
    df = pd.read_csv(csv_path)
    date_col = "History Price / Date"
    if date_col not in df.columns:
        raise ValueError(f"Khong tim thay cot ngay: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    return df


def compute_weekly_min_max_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute weekly min/max return per ticker relative to each week's first valid close."""
    weekly_groups = prices.resample("W-FRI")
    weekly_open = weekly_groups.first()
    weekly_min = weekly_groups.min()
    weekly_max = weekly_groups.max()

    weekly_min_return = (weekly_min / weekly_open) - 1.0
    weekly_max_return = (weekly_max / weekly_open) - 1.0

    return weekly_min_return, weekly_max_return


def to_log_returns(values: pd.Series) -> pd.Series:
    """Convert simple returns r to log returns ln(1+r), ignoring invalid r <= -1."""
    clean = values.dropna()
    clean = clean[clean > -1.0]
    return np.log1p(clean)


def compute_bimodal_metrics(x: np.ndarray) -> dict:
    n = len(x)
    if n < 20:
        raise ValueError("So diem du lieu qua it de kiem dinh on dinh (can >= 20).")

    skewness = stats.skew(x, bias=False)
    kurtosis_pearson = stats.kurtosis(x, fisher=False, bias=False)
    ks_gap = kurtosis_pearson - skewness ** 2
    bc = (skewness ** 2 + 1.0) / kurtosis_pearson if kurtosis_pearson != 0 else np.nan

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    ks_stat, ks_p = stats.kstest(x, "norm", args=(mu, sigma if sigma > 0 else 1e-12))
    jb_stat, jb_p = stats.jarque_bera(x)

    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(x.reshape(-1, 1))
    gmm2 = GaussianMixture(n_components=2, random_state=42).fit(x.reshape(-1, 1))
    bic1 = gmm1.bic(x.reshape(-1, 1))
    bic2 = gmm2.bic(x.reshape(-1, 1))

    return {
        "n": n,
        "mean": mu,
        "std": sigma,
        "skewness": skewness,
        "kurtosis_pearson": kurtosis_pearson,
        "k_minus_s2": ks_gap,
        "k_minus_s2_pass": ks_gap >= 1.0,
        "bimodality_coefficient": bc,
        "bc_suggest_bimodal": bc > (5.0 / 9.0),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
        "jb_stat": jb_stat,
        "jb_pvalue": jb_p,
        "gmm_bic_1": bic1,
        "gmm_bic_2": bic2,
        "gmm_2_better": bic2 < bic1,
    }


def binomial_up_down_test(x: np.ndarray) -> dict:
    """Optional: test up/down signs as Bernoulli, then Binomial over sample count."""
    up = int(np.sum(x > 0))
    down = int(np.sum(x < 0))
    n = up + down
    if n == 0:
        return {"binom_n": 0, "up": 0, "down": 0, "p_up_hat": np.nan, "pvalue_vs_0_5": np.nan}

    result = stats.binomtest(k=up, n=n, p=0.5, alternative="two-sided")
    return {
        "binom_n": n,
        "up": up,
        "down": down,
        "p_up_hat": up / n,
        "pvalue_vs_0_5": result.pvalue,
    }


def save_distribution_plot(x: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=50, density=True, alpha=0.6, color="#4C78A8", edgecolor="white")

    xs = np.linspace(np.min(x), np.max(x), 400)
    #kde = stats.gaussian_kde(x)
    #ax.plot(xs, kde(xs), color="#F58518", linewidth=2.0, label="KDE")

    ax.set_title(title)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_report(name: str, metrics: dict, binom: dict) -> None:
    print(f"\n{'=' * 80}")
    print(f"BAO CAO: {name}")
    print(f"{'=' * 80}")
    print(f"So mau: {metrics['n']}")
    print(f"Mean: {metrics['mean']:.6f} | Std: {metrics['std']:.6f}")
    print(f"Skewness (S): {metrics['skewness']:.6f}")
    print(f"Kurtosis Pearson (K): {metrics['kurtosis_pearson']:.6f}")
    print(f"K - S^2: {metrics['k_minus_s2']:.6f} | Thoa K - S^2 >= 1: {metrics['k_minus_s2_pass']}")
    print(
        "Bimodality Coefficient: "
        f"{metrics['bimodality_coefficient']:.6f} | Goi y bimodal (BC > 5/9): {metrics['bc_suggest_bimodal']}"
    )
    print(
        f"K-S normality p-value: {metrics['ks_pvalue']:.6g} | "
        f"Jarque-Bera p-value: {metrics['jb_pvalue']:.6g}"
    )
    print(
        f"GMM BIC (1 component): {metrics['gmm_bic_1']:.2f} | "
        f"GMM BIC (2 components): {metrics['gmm_bic_2']:.2f} | "
        f"2 components tot hon: {metrics['gmm_2_better']}"
    )
    print(
        f"Binomial up/down test (p=0.5): n={binom['binom_n']}, up={binom['up']}, down={binom['down']}, "
        f"p_up_hat={binom['p_up_hat']:.4f}, p-value={binom['pvalue_vs_0_5']:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Kiem tra log return weekly co tinh chat bimodal/fat-tail theo K-S^2 va cac kiem dinh bo sung."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/stock_market_19_24.csv"),
        help="Duong dan den file daily gia dong cua",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dataset/figs"),
        help="Thu muc luu hinh histogram + KDE",
    )
    parser.add_argument(
        "--save-weekly",
        type=Path,
        default=Path("dataset/weekly_min_max_return.csv"),
        help="Noi luu bang weekly min/max return de kiem tra lai",
    )
    args = parser.parse_args()

    prices = load_daily_prices(args.input)
    min_df, max_df = compute_weekly_min_max_returns(prices)

    # Save a flattened weekly table for auditability.
    weekly_out = pd.concat(
        {
            "Min Return": min_df,
            "Max Return": max_df,
        },
        axis=1,
    )
    args.save_weekly.parent.mkdir(parents=True, exist_ok=True)
    weekly_out.to_csv(args.save_weekly, index_label="Week End")
    print(f"Da luu bang weekly min/max: {args.save_weekly}")

    min_log = to_log_returns(min_df.stack())
    max_log = to_log_returns(max_df.stack())
    both_log = pd.concat([min_log, max_log], ignore_index=True)

    datasets = {
        "MIN only": min_log.to_numpy(),
        "MAX only": max_log.to_numpy(),
        "MIN + MAX": both_log.to_numpy(),
    }

    for name, values in datasets.items():
        metrics = compute_bimodal_metrics(values)
        binom = binomial_up_down_test(values)
        print_report(f"{name} (WEEKLY)", metrics, binom)

        out_file = args.outdir / f"weekly_log_return_{name.lower().replace(' ', '_').replace('+', 'plus')}.png"
        save_distribution_plot(values, f"Weekly log return distribution - {name}", out_file)
        print(f"Da luu hinh: {out_file}")


if __name__ == "__main__":
    main()
