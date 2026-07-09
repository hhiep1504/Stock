import argparse
import errno
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_daily(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = "History Price / Date"
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    return df


def minmax_past_only(series: pd.Series, window: int = 52) -> pd.Series:
    s_min = series.rolling(window=window, min_periods=max(5, window // 5)).min().shift(1)
    s_max = series.rolling(window=window, min_periods=max(5, window // 5)).max().shift(1)
    denom = s_max - s_min
    out = (series - s_min) / denom
    return out.where(np.isfinite(out), np.nan)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def hurst_exponent_simple(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or np.std(x) < 1e-12:
        return np.nan
    lags = np.arange(2, 10)
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    tau = np.asarray(tau)
    if np.any(tau <= 0):
        return np.nan
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(slope)


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= m + 1:
        return np.nan
    sd = np.std(x)
    if sd < 1e-12:
        return np.nan
    r = r_ratio * sd

    def _count(mm: int) -> int:
        count = 0
        for i in range(n - mm):
            template = x[i : i + mm]
            for j in range(i + 1, n - mm + 1):
                compare = x[j : j + mm]
                if np.max(np.abs(template - compare)) <= r:
                    count += 1
        return count

    b = _count(m)
    a = _count(m + 1)
    if b == 0 or a == 0:
        return np.nan
    return float(-np.log(a / b))


def build_weekly_panel(daily_prices: pd.DataFrame) -> pd.DataFrame:
    records = []

    weekly_open_all = daily_prices.resample("W-FRI").first()
    weekly_close_all = daily_prices.resample("W-FRI").last()
    market_candidates = ["VNINDEX", "VN-INDEX", "VN_INDEX"]

    market_col = next((col for col in market_candidates if col in daily_prices.columns), None)
    if market_col is not None:
        market_ret = (weekly_close_all[market_col] - weekly_open_all[market_col]) / (weekly_open_all[market_col] + 1e-8)
    else:
        # Fallback: equal-weight market proxy from all available stocks.
        market_ret = ((weekly_close_all - weekly_open_all) / (weekly_open_all + 1e-8)).mean(axis=1)

    has_volume = any(col.upper().endswith("_VOLUME") for col in daily_prices.columns)

    for ticker in daily_prices.columns:
        s = daily_prices[ticker].dropna()
        if s.empty:
            continue

        weekly_open = s.resample("W-FRI").first()
        weekly_high = s.resample("W-FRI").max()
        weekly_low = s.resample("W-FRI").min()
        weekly_close = s.resample("W-FRI").last()
        weekly_mean = s.resample("W-FRI").mean()
        weekly_std = s.resample("W-FRI").std()

        weekly_skew = s.resample("W-FRI").apply(
            lambda z: skew(pd.Series(z).pct_change().dropna()) if len(z) > 2 else np.nan
        )

        feat_return = np.log((weekly_close + 1e-8) / (weekly_open + 1e-8))
        feat_std = np.log1p(weekly_std)
        feat_mean = np.log1p(weekly_mean)
        feat_skew = weekly_skew

        simple_ret = (weekly_close - weekly_open) / (weekly_open + 1e-8)
        alpha_excess = simple_ret - market_ret.reindex(simple_ret.index)

        weekly_range = (weekly_high - weekly_low) / (weekly_open + 1e-8)
        weekly_range_lag1 = weekly_range.shift(1)

        min_4w = weekly_close.rolling(4, min_periods=4).min()
        max_4w = weekly_close.rolling(4, min_periods=4).max()
        rel_pos_4w = (weekly_close - min_4w) / (max_4w - min_4w + 1e-8)

        log_ret = np.log((weekly_close + 1e-8) / (weekly_close.shift(1) + 1e-8))

        sma_4 = weekly_close.rolling(4, min_periods=4).mean()
        ema_4 = weekly_close.ewm(span=4, adjust=False, min_periods=4).mean()
        sma_ratio_4 = (weekly_close / (sma_4 + 1e-8)) - 1.0
        ema_ratio_4 = (weekly_close / (ema_4 + 1e-8)) - 1.0

        rolling_std_4 = weekly_close.rolling(4, min_periods=4).std()
        bollinger_upper_4 = sma_4 + 2.0 * rolling_std_4
        bollinger_lower_4 = sma_4 - 2.0 * rolling_std_4
        bollinger_width_4 = (bollinger_upper_4 - bollinger_lower_4) / (sma_4 + 1e-8)
        bollinger_percent_b_4 = (weekly_close - bollinger_lower_4) / (
            bollinger_upper_4 - bollinger_lower_4 + 1e-8
        )

        rsi_14 = rsi(weekly_close, window=14)
        ema_fast_12 = weekly_close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow_26 = weekly_close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_norm = (ema_fast_12 - ema_slow_26) / (weekly_close + 1e-8)

        weekly_log_close = np.log(weekly_close + 1e-8)
        hurst_20 = weekly_log_close.rolling(20, min_periods=20).apply(
            lambda values: hurst_exponent_simple(np.asarray(values, dtype=float)),
            raw=False,
        )
        entropy_20 = log_ret.rolling(20, min_periods=20).apply(
            lambda values: sample_entropy(np.asarray(values, dtype=float)),
            raw=False,
        )

        # Optional feature: volume-weighted weekly return if volume data exists.
        if has_volume and f"{ticker}_VOLUME" in daily_prices.columns:
            vol = daily_prices[f"{ticker}_VOLUME"].dropna()
            aligned = pd.concat([s.rename("p"), vol.rename("v")], axis=1).dropna()
            vw_price = (aligned["p"] * aligned["v"]).resample("W-FRI").sum() / (
                aligned["v"].resample("W-FRI").sum() + 1e-8
            )
            vw_return = np.log((vw_price + 1e-8) / (vw_price.shift(1) + 1e-8))
        else:
            vw_return = pd.Series(np.nan, index=weekly_close.index)

        target_min = (weekly_low - weekly_open) / (weekly_open + 1e-8)
        target_max = (weekly_high - weekly_open) / (weekly_open + 1e-8)

        frame = pd.DataFrame(
            {
                "week": weekly_close.index,
                "ticker": ticker,
                "target_min": target_min,
                "target_max": target_max,
                "f_std": feat_std,
                "f_mean": feat_mean,
                "f_return": feat_return,
                "f_skew": feat_skew,
                "alpha_excess": alpha_excess,
                "weekly_range_lag1": weekly_range_lag1,
                "rel_pos_4w": rel_pos_4w,
                "vw_return": vw_return,
                "log_return": log_ret,
                "sma_ratio_4": sma_ratio_4,
                "ema_ratio_4": ema_ratio_4,
                "bollinger_width_4": bollinger_width_4,
                "bollinger_percent_b_4": bollinger_percent_b_4,
                "rsi_14": rsi_14,
                "macd_norm": macd_norm,
                "hurst_20": hurst_20,
                "entropy_20": entropy_20,
            }
        )
        records.append(frame)

    panel = pd.concat(records, ignore_index=True)
    panel = panel.sort_values(["week", "ticker"]).reset_index(drop=True)
    return panel


def align_panel_fair(panel: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    """Use one common sample set for every setup to ensure fair comparison."""
    needed = ["target_min", "target_max"] + required_features
    aligned = panel.dropna(subset=needed).copy()
    aligned = aligned.sort_values(["week", "ticker"]).reset_index(drop=True)
    return aligned


class SequenceRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, rnn_type: str = "gru"):
        super().__init__()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


def prepare_sequence_split(
    panel: pd.DataFrame,
    features: list[str],
    train_weeks: set,
    test_weeks: set,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for _, grp in panel.groupby("ticker"):
        g = grp.sort_values("week").dropna(subset=features + ["target_min", "target_max"])
        if len(g) <= lookback:
            continue

        x_full = g[features].to_numpy()
        y_full = g[["target_min", "target_max"]].to_numpy()
        w_full = g["week"].to_numpy()

        for idx in range(lookback, len(g)):
            seq = x_full[idx - lookback : idx]
            target = y_full[idx]
            week = w_full[idx]

            if week in train_weeks:
                x_train_list.append(seq)
                y_train_list.append(target)
            elif week in test_weeks:
                x_test_list.append(seq)
                y_test_list.append(target)

    if len(x_train_list) == 0 or len(x_test_list) == 0:
        return np.empty((0, lookback, len(features))), np.empty((0, 2)), np.empty((0, lookback, len(features))), np.empty((0, 2))

    return (
        np.asarray(x_train_list, dtype=np.float32),
        np.asarray(y_train_list, dtype=np.float32),
        np.asarray(x_test_list, dtype=np.float32),
        np.asarray(y_test_list, dtype=np.float32),
    )


def correlation_top_k(train_df: pd.DataFrame, features: list[str], k: int) -> list[str]:
    scores = []
    for f in features:
        if train_df[f].std() < 1e-12:
            scores.append((f, -np.inf))
            continue
        corr_min = np.abs(train_df[f].corr(train_df["target_min"]))
        corr_max = np.abs(train_df[f].corr(train_df["target_max"]))
        score = float(np.nan_to_num(corr_min, nan=0.0) + np.nan_to_num(corr_max, nan=0.0))
        scores.append((f, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:k]]


def train_predict_sequence_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    model_type: str,
    epochs: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SequenceRegressor(
        input_dim=x_train.shape[2],
        hidden_dim=64,
        rnn_type=model_type,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_t).cpu().numpy()
    return y_pred


def evaluate_time_splits(
    panel: pd.DataFrame,
    features: list[str],
    n_splits: int,
    top_k_prune: int | None = None,
    model_type: str = "gru",
    lookback: int = 8,
    epochs: int = 40,
    lr: float = 1e-3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Counter]:
    weeks = np.array(sorted(panel["week"].dropna().unique()))
    if len(weeks) < n_splits + 2:
        raise ValueError("Not enough weekly points for requested n_splits")

    fold_sizes = len(weeks) // (n_splits + 1)
    results = []
    selected_counter: Counter = Counter()
    
    # Diagnostic: show which features we're using
    print(f"  Evaluating {len(features)} features: {', '.join(features)}")

    for fold in range(1, n_splits + 1):
        train_end = fold * fold_sizes
        test_end = (fold + 1) * fold_sizes if fold < n_splits else len(weeks)
        train_weeks = weeks[:train_end]
        test_weeks = weeks[train_end:test_end]

        train_df = panel[panel["week"].isin(train_weeks)].copy()
        test_df = panel[panel["week"].isin(test_weeks)].copy()

        n_train_before = len(train_df)
        n_test_before = len(test_df)
        
        train_df = train_df.dropna(subset=features + ["target_min", "target_max"])
        test_df = test_df.dropna(subset=features + ["target_min", "target_max"])
        
        n_train_after = len(train_df)
        n_test_after = len(test_df)
        
        print(f"    Fold {fold}: train {n_train_before}->{n_train_after}, test {n_test_before}->{n_test_after}")
        
        if train_df.empty or test_df.empty:
            print(f"      -> SKIP (train or test empty after dropna)")
            continue

        use_features = list(features)
        if top_k_prune is not None and top_k_prune < len(use_features):
            use_features = correlation_top_k(train_df, use_features, top_k_prune)
            selected_counter.update(use_features)

        x_train, y_train, x_test, y_test = prepare_sequence_split(
            panel=panel,
            features=use_features,
            train_weeks=set(train_weeks),
            test_weeks=set(test_weeks),
            lookback=lookback,
        )
        
        print(f"      Sequences: train {x_train.shape[0]}, test {x_test.shape[0]}")
        
        if x_train.shape[0] == 0 or x_test.shape[0] == 0:
            print(f"      -> SKIP (no sequences)")
            continue

        scaler = StandardScaler()
        x_train_2d = x_train.reshape(-1, x_train.shape[2])
        x_test_2d = x_test.reshape(-1, x_test.shape[2])
        x_train = scaler.fit_transform(x_train_2d).reshape(x_train.shape)
        x_test = scaler.transform(x_test_2d).reshape(x_test.shape)

        y_pred = train_predict_sequence_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            model_type=model_type,
            epochs=epochs,
            lr=lr,
            seed=random_state,
        )

        r2_min = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_max = r2_score(y_test[:, 1], y_pred[:, 1])
        mae_min = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mae_max = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

        results.append(
            {
                "fold": fold,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "r2_min": r2_min,
                "r2_max": r2_max,
                "r2_avg": 0.5 * (r2_min + r2_max),
                "mae_min": mae_min,
                "mae_max": mae_max,
                "mae_avg": 0.5 * (mae_min + mae_max),
                "used_features": ",".join(use_features),
            }
        )

    return pd.DataFrame(results), selected_counter


def evaluate_feature_subset(
    panel: pd.DataFrame,
    feature_subset: list[str],
    n_splits: int,
    model_type: str,
    lookback: int,
    epochs: int,
    lr: float,
    random_state: int,
) -> float:
    """Return mean CV score used by Optuna objective (higher is better)."""
    if len(feature_subset) == 0:
        return -1e9

    res, _ = evaluate_time_splits(
        panel=panel,
        features=feature_subset,
        n_splits=n_splits,
        top_k_prune=None,
        model_type=model_type,
        lookback=lookback,
        epochs=epochs,
        lr=lr,
        random_state=random_state,
    )
    if res.empty:
        return -1e9
    return float(res["r2_avg"].mean())


def run_optuna_subset_search(
    panel: pd.DataFrame,
    candidate_features: list[str],
    n_splits: int,
    n_trials: int,
    min_features: int,
    max_features: int,
    model_type: str,
    lookback: int,
    epochs: int,
    lr: float,
    random_state: int,
) -> tuple[optuna.study.Study, pd.DataFrame]:
    logs: list[dict] = []

    def objective(trial: optuna.trial.Trial) -> float:
        chosen = []
        for feat in candidate_features:
            if trial.suggest_int(f"use_{feat}", 0, 1) == 1:
                chosen.append(feat)

        if len(chosen) < min_features or len(chosen) > max_features:
            score = -1e9
        else:
            score = evaluate_feature_subset(
                panel=panel,
                feature_subset=chosen,
                n_splits=n_splits,
                model_type=model_type,
                lookback=lookback,
                epochs=epochs,
                lr=lr,
                random_state=random_state,
            )

        logs.append(
            {
                "trial": trial.number,
                "score_r2_avg": score,
                "n_features": len(chosen),
                "features": ",".join(chosen),
            }
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = pd.DataFrame(logs).sort_values("score_r2_avg", ascending=False).reset_index(drop=True)
    return study, trials_df


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "setup": label,
            "folds": 0,
            "r2_avg_mean": np.nan,
            "r2_avg_std": np.nan,
            "mae_avg_mean": np.nan,
            "mae_avg_std": np.nan,
        }

    return {
        "setup": label,
        "folds": int(df.shape[0]),
        "r2_avg_mean": float(df["r2_avg"].mean()),
        "r2_avg_std": float(df["r2_avg"].std(ddof=1)),
        "mae_avg_mean": float(df["mae_avg"].mean()),
        "mae_avg_std": float(df["mae_avg"].std(ddof=1)),
    }


def resolve_writable_outdir(outdir: Path) -> Path:
    """Return a writable output folder, falling back on Kaggle working dir when needed."""
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    except OSError as exc:
        is_read_only = exc.errno in {errno.EROFS, errno.EPERM, errno.EACCES}
        is_kaggle_input = str(outdir).startswith("/kaggle/input/")
        if not (is_read_only or is_kaggle_input):
            raise

    fallback = Path("/kaggle/working") / "feature_pruning_outputs" / outdir.name
    fallback.mkdir(parents=True, exist_ok=True)
    print(f"[WARN] Outdir is not writable: {outdir}")
    print(f"[INFO] Using fallback outdir: {fallback}")
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline 4 features vs finance-justified extended set on weekly targets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/stock_market_19_24.csv"),
        help="Path to daily close-price table",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of expanding time folds")
    parser.add_argument("--top-k", type=int, default=6, help="Top-k features kept in pruning mode")
    parser.add_argument("--fair-mode", action="store_true", help="Use common valid rows for all setups")
    parser.add_argument("--optuna-trials", type=int, default=10, help="Optuna trials for subset search")
    parser.add_argument("--min-features", type=int, default=5, help="Min number of features in Optuna subset")
    parser.add_argument("--max-features", type=int, default=8, help="Max number of features in Optuna subset")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm"], help="Sequence model")
    parser.add_argument("--lookback", type=int, default=8, help="Weekly lookback window")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for sequence model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--exclude-features",
        type=str,
        default="",
        help="Comma-separated feature names to exclude from the extended feature set",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dataset/figs"),
        help="Output folder for reports",
    )
    args = parser.parse_args()

    baseline4 = ["f_std", "f_mean", "f_return", "f_skew"]
    feature_plus = [
        "alpha_excess",
        "weekly_range_lag1",
        "rel_pos_4w",
        "vw_return",
        "sma_ratio_4",
        "ema_ratio_4",
        "bollinger_width_4",
        "bollinger_percent_b_4",
        "rsi_14",
        "macd_norm",
        "hurst_20",
        "entropy_20",
    ]

    daily = load_daily(args.input)
    panel = build_weekly_panel(daily)

    # If volume-based feature is unavailable, drop it from candidate list.
    feature_plus = [f for f in feature_plus if panel[f].notna().any()]
    excluded_features = {item.strip() for item in args.exclude_features.split(",") if item.strip()}
    feature_plus = [f for f in feature_plus if f not in excluded_features]
    extended_features = baseline4 + feature_plus

    all_features = extended_features
    panel = panel.dropna(subset=["target_min", "target_max"]).copy()
    
    # Diagnostic: show NaN count per feature before alignment
    print("\n=== Feature NaN Analysis (before alignment) ===")
    for feat in all_features:
        n_nan = panel[feat].isna().sum()
        n_total = len(panel)
        pct_nan = 100.0 * n_nan / n_total if n_total > 0 else 0.0
        print(f"  {feat:20s}: {n_nan:6d}/{n_total:6d} NaN ({pct_nan:5.1f}%)")

    if args.fair_mode:
        print(f"\nDropping rows with ANY NaN in all features...")
        panel_eval = align_panel_fair(panel, all_features)
        print(f"Fair mode ON: common rows = {len(panel_eval)} (dropped {len(panel) - len(panel_eval)} rows)")
    else:
        panel_eval = panel
        print("Fair mode OFF: each setup uses its own available rows")

    res4, _ = evaluate_time_splits(
        panel_eval,
        baseline4,
        n_splits=args.splits,
        top_k_prune=None,
        model_type=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        lr=args.lr,
        random_state=args.seed,
    )
    res11, _ = evaluate_time_splits(
        panel_eval,
        extended_features,
        n_splits=args.splits,
        top_k_prune=None,
        model_type=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        lr=args.lr,
        random_state=args.seed,
    )
    res11_pruned, picked = evaluate_time_splits(
        panel_eval,
        extended_features,
        n_splits=args.splits,
        top_k_prune=args.top_k,
        model_type=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        lr=args.lr,
        random_state=args.seed,
    )

    study, trials_df = run_optuna_subset_search(
        panel=panel_eval,
        candidate_features=extended_features,
        n_splits=args.splits,
        n_trials=args.optuna_trials,
        min_features=args.min_features,
        max_features=args.max_features,
        model_type=args.model,
        lookback=args.lookback,
        epochs=max(8, args.epochs // 2),
        lr=args.lr,
        random_state=args.seed,
    )

    best_features = [
        feat for feat in extended_features if study.best_params.get(f"use_{feat}", 0) == 1
    ]
    res_optuna_best, _ = evaluate_time_splits(
        panel_eval,
        best_features,
        n_splits=args.splits,
        top_k_prune=None,
        model_type=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        lr=args.lr,
        random_state=args.seed,
    )

    summary = pd.DataFrame(
        [
            summarize(res4, "baseline_4"),
            summarize(res11, "extended_features"),
            summarize(res11_pruned, f"pruned_extended_to_{args.top_k}"),
            summarize(res_optuna_best, "optuna_best_subset"),
        ]
    )

    outdir = resolve_writable_outdir(args.outdir)
    summary_path = outdir / "feature_pruning_weekly_summary.csv"
    detail_path = outdir / "feature_pruning_weekly_details.csv"
    picked_path = outdir / "feature_pruning_selected_counts.csv"
    optuna_trials_path = outdir / "feature_pruning_optuna_trials.csv"
    optuna_best_path = outdir / "feature_pruning_optuna_best_features.csv"

    detail = pd.concat(
        [
            res4.assign(setup="baseline_4"),
            res11.assign(setup="extended_features"),
            res11_pruned.assign(setup=f"pruned_extended_to_{args.top_k}"),
            res_optuna_best.assign(setup="optuna_best_subset"),
        ],
        ignore_index=True,
    )

    picked_df = pd.DataFrame(
        {
            "feature": list(picked.keys()),
            "selected_count": list(picked.values()),
        }
    ).sort_values("selected_count", ascending=False)

    print("=== Weekly Feature Pruning Test ===")
    print(summary.to_string(index=False))
    if not picked_df.empty:
        print("\nTop selected features in pruning mode:")
        print(picked_df.head(10).to_string(index=False))

    print("\nBest Optuna subset:")
    print(", ".join(best_features) if best_features else "No valid subset found")
    print(f"Best Optuna objective (mean r2_avg): {study.best_value:.6f}")

    try:
        summary.to_csv(summary_path, index=False)
        detail.to_csv(detail_path, index=False)
        picked_df.to_csv(picked_path, index=False)
        trials_df.to_csv(optuna_trials_path, index=False)
        pd.DataFrame({"feature": best_features}).to_csv(optuna_best_path, index=False)
    except PermissionError as exc:
        print(f"\n[WARN] Could not save CSV outputs: {exc}")
        return

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved details: {detail_path}")
    print(f"Saved selected feature counts: {picked_path}")
    print(f"Saved Optuna trials: {optuna_trials_path}")
    print(f"Saved Optuna best subset: {optuna_best_path}")


if __name__ == "__main__":
    main()
