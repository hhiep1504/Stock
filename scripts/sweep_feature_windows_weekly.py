#!/usr/bin/env python3
"""Sweep technical-feature parameters on weekly stock data.

The script is intentionally a screening tool: it tests all configured
single-feature variants against the baseline on the same valid rows, then runs
a greedy forward selection over the best variants. The selected set should be
re-tested with the official GAT/GCN/LSTM benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


DATE_COLUMN = "History Price / Date"
EPSILON = 1e-8
BASELINE_FEATURES = ("f_std", "f_mean", "f_return", "f_skew")


@dataclass(slots=True)
class Candidate:
    name: str
    family: str
    params: str


@dataclass(slots=True)
class EvalResult:
    feature_set: str
    family: str
    params: str
    features: str
    n_features: int
    n_periods: int
    n_sequences: int
    first_period: str
    last_period: str
    mae_interval_mean: float
    mae_interval_std: float
    baseline_same_rows_mae: float
    delta_vs_baseline: float
    status: str


class SequenceRegressor(nn.Module):
    """Small GRU/LSTM used only for feature screening."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, rnn_type: str = "gru"):
        super().__init__()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :])


def load_daily(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"Missing date column: {DATE_COLUMN}")

    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    for column in frame.columns:
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    return frame.ffill().bfill()


def hurst_exponent(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 12 or np.std(values) < 1e-12:
        return np.nan
    max_lag = min(10, len(values) // 2)
    lags = np.arange(2, max_lag)
    if len(lags) < 2:
        return np.nan
    tau = np.asarray([np.std(values[lag:] - values[:-lag]) for lag in lags])
    if np.any(tau <= 0):
        return np.nan
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def sample_entropy(values: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n <= m + 1:
        return np.nan
    scale = np.std(values)
    if scale < 1e-12:
        return np.nan
    radius = r_ratio * scale

    def _count(order: int) -> int:
        count = 0
        for i in range(n - order):
            template = values[i : i + order]
            for j in range(i + 1, n - order + 1):
                compare = values[j : j + order]
                if np.max(np.abs(template - compare)) <= radius:
                    count += 1
        return count

    b_count = _count(m)
    a_count = _count(m + 1)
    if b_count == 0 or a_count == 0:
        return np.nan
    return float(-np.log(a_count / b_count))


def compute_rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_feature_frames(
    daily_prices: pd.DataFrame,
    sma_windows: Iterable[int],
    ema_windows: Iterable[int],
    bollinger_windows: Iterable[int],
    bollinger_ks: Iterable[float],
    rsi_windows: Iterable[int],
    macd_pairs: Iterable[tuple[int, int]],
    hurst_windows: Iterable[int],
    entropy_windows: Iterable[int],
    rel_pos_windows: Iterable[int],
    include_entropy: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    weekly_open = daily_prices.resample("W-FRI").first()
    weekly_high = daily_prices.resample("W-FRI").max()
    weekly_low = daily_prices.resample("W-FRI").min()
    weekly_close = daily_prices.resample("W-FRI").last()
    weekly_mean = daily_prices.resample("W-FRI").mean()
    weekly_std = daily_prices.resample("W-FRI").std()

    simple_return = (weekly_close - weekly_open) / (weekly_open + EPSILON)
    market_return = simple_return.mean(axis=1)
    log_return = np.log((weekly_close + EPSILON) / (weekly_close.shift(1) + EPSILON))
    log_close = np.log(weekly_close + EPSILON)

    frames: dict[str, pd.DataFrame] = {
        "f_std": np.log1p(weekly_std),
        "f_mean": np.log1p(weekly_mean),
        "f_return": np.log((weekly_close + EPSILON) / (weekly_open + EPSILON)),
        "f_skew": daily_prices.pct_change().resample("W-FRI").apply(lambda values: values.dropna().skew()),
        "alpha_excess": simple_return.sub(market_return, axis=0),
        "weekly_range_lag1": ((weekly_high - weekly_low) / (weekly_open + EPSILON)).shift(1),
    }

    for window in rel_pos_windows:
        rolling_high = weekly_high.rolling(window, min_periods=window).max()
        rolling_low = weekly_low.rolling(window, min_periods=window).min()
        frames[f"rel_pos_w{window}"] = (weekly_close - rolling_low) / (rolling_high - rolling_low + EPSILON)

    for window in sma_windows:
        sma = weekly_close.rolling(window, min_periods=window).mean()
        frames[f"sma_ratio_w{window}"] = (weekly_close / (sma + EPSILON)) - 1.0

    for window in ema_windows:
        ema = weekly_close.ewm(span=window, adjust=False, min_periods=window).mean()
        frames[f"ema_ratio_w{window}"] = (weekly_close / (ema + EPSILON)) - 1.0

    for window in bollinger_windows:
        middle = weekly_close.rolling(window, min_periods=window).mean()
        sigma = weekly_close.rolling(window, min_periods=window).std()
        for k_value in bollinger_ks:
            upper = middle + k_value * sigma
            lower = middle - k_value * sigma
            k_label = str(k_value).replace(".", "p")
            frames[f"bollinger_width_w{window}_k{k_label}"] = (upper - lower) / (middle + EPSILON)
            frames[f"bollinger_percent_b_w{window}_k{k_label}"] = (
                (weekly_close - lower) / (upper - lower + EPSILON)
            )

    for window in rsi_windows:
        frames[f"rsi_w{window}"] = compute_rsi(weekly_close, window)

    for fast, slow in macd_pairs:
        fast_ema = weekly_close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        slow_ema = weekly_close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        frames[f"macd_norm_f{fast}_s{slow}"] = (fast_ema - slow_ema) / (weekly_close + EPSILON)

    for window in hurst_windows:
        frames[f"hurst_w{window}"] = log_close.rolling(window, min_periods=window).apply(
            lambda values: hurst_exponent(np.asarray(values, dtype=float)),
            raw=False,
        )

    if include_entropy:
        for window in entropy_windows:
            frames[f"entropy_w{window}"] = log_return.rolling(window, min_periods=window).apply(
                lambda values: sample_entropy(np.asarray(values, dtype=float)),
                raw=False,
            )

    target_min = (weekly_low - weekly_open) / (weekly_open + EPSILON)
    target_max = (weekly_high - weekly_open) / (weekly_open + EPSILON)
    return frames, target_min, target_max


def build_candidates(frames: dict[str, pd.DataFrame]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for name in sorted(frames):
        if name in BASELINE_FEATURES:
            continue
        if name.startswith("sma_ratio"):
            family = "sma_ratio"
        elif name.startswith("ema_ratio"):
            family = "ema_ratio"
        elif name.startswith("bollinger_width"):
            family = "bollinger_width"
        elif name.startswith("bollinger_percent_b"):
            family = "bollinger_percent_b"
        elif name.startswith("rsi"):
            family = "rsi"
        elif name.startswith("macd"):
            family = "macd"
        elif name.startswith("hurst"):
            family = "hurst"
        elif name.startswith("entropy"):
            family = "entropy"
        elif name.startswith("rel_pos"):
            family = "rel_pos"
        else:
            family = name
        candidates.append(Candidate(name=name, family=family, params=name.removeprefix(family).strip("_")))
    return candidates


def make_tensors(
    feature_frames: dict[str, pd.DataFrame],
    target_min: pd.DataFrame,
    target_max: pd.DataFrame,
    feature_names: tuple[str, ...],
    period_filter: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common = set(target_min.index) & set(target_max.index)
    for name in feature_names:
        common &= set(feature_frames[name].index)

    x_values = []
    y_values = []
    period_labels: list[str] = []
    for idx in sorted(common):
        period_label = str(idx.date()) if hasattr(idx, "date") else str(idx)
        if period_filter is not None and period_label not in period_filter:
            continue
        feature_stack = np.stack([feature_frames[name].loc[idx].to_numpy() for name in feature_names], axis=1)
        target_stack = np.stack([target_min.loc[idx].to_numpy(), target_max.loc[idx].to_numpy()], axis=1)
        if not np.isfinite(feature_stack).all() or not np.isfinite(target_stack).all():
            continue
        x_values.append(feature_stack.astype(np.float32))
        y_values.append(target_stack.astype(np.float32))
        period_labels.append(period_label)

    if not x_values:
        num_nodes = target_min.shape[1]
        return (
            np.empty((0, num_nodes, len(feature_names)), dtype=np.float32),
            np.empty((0, num_nodes, 2), dtype=np.float32),
            [],
        )
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32), period_labels


def fit_transform_for_train(x_full: np.ndarray, train_sequence_count: int, lookback: int) -> np.ndarray:
    train_period_end = min(x_full.shape[0], train_sequence_count + lookback - 1)
    scaler = StandardScaler()
    scaler.fit(x_full[:train_period_end].reshape(-1, x_full.shape[-1]))
    return scaler.transform(x_full.reshape(-1, x_full.shape[-1])).reshape(x_full.shape)


def flatten_sequences(x_full: np.ndarray, y_full: np.ndarray, starts: range, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for start in starts:
        x_window = x_full[start : start + lookback]
        y_target = y_full[start + lookback]
        for node_index in range(x_full.shape[1]):
            x_rows.append(x_window[:, node_index, :])
            y_rows.append(y_target[node_index, :])
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    epochs: int,
    hidden_dim: int,
    rnn_type: str,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SequenceRegressor(x_train.shape[-1], hidden_dim=hidden_dim, rnn_type=rnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(x_test_t).detach().cpu().numpy()


def evaluate_feature_set(
    feature_frames: dict[str, pd.DataFrame],
    target_min: pd.DataFrame,
    target_max: pd.DataFrame,
    feature_names: tuple[str, ...],
    lookback: int,
    min_train_size: int,
    test_step: int,
    seeds: tuple[int, ...],
    runs: int,
    epochs: int,
    hidden_dim: int,
    rnn_type: str,
    device: torch.device,
    period_filter: set[str] | None = None,
) -> tuple[float, float, int, int, str, str, tuple[str, ...]]:
    x_raw, y_full, period_labels = make_tensors(
        feature_frames,
        target_min,
        target_max,
        feature_names,
        period_filter=period_filter,
    )
    n_periods = int(x_raw.shape[0])
    n_sequences = max(0, n_periods - lookback)
    if n_sequences <= min_train_size:
        raise ValueError(f"not enough sequences: {n_sequences}")

    fold_maes = []
    for test_start in range(min_train_size, n_sequences, test_step):
        test_end = min(n_sequences, test_start + test_step)
        x_scaled = fit_transform_for_train(x_raw, train_sequence_count=test_start, lookback=lookback)
        x_train, y_train = flatten_sequences(x_scaled, y_full, range(0, test_start), lookback)
        x_test, y_test = flatten_sequences(x_scaled, y_full, range(test_start, test_end), lookback)
        if len(x_train) == 0 or len(x_test) == 0:
            continue
        for run_index in range(runs):
            seed = seeds[run_index % len(seeds)]
            prediction = train_predict(
                x_train,
                y_train,
                x_test,
                seed=seed,
                epochs=epochs,
                hidden_dim=hidden_dim,
                rnn_type=rnn_type,
                device=device,
            )
            mae_min = mean_absolute_error(y_test[:, 0], prediction[:, 0])
            mae_max = mean_absolute_error(y_test[:, 1], prediction[:, 1])
            fold_maes.append(0.5 * (mae_min + mae_max))

    if not fold_maes:
        raise ValueError("no valid folds")
    return (
        float(np.mean(fold_maes)),
        float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0,
        n_periods,
        n_sequences,
        period_labels[0],
        period_labels[-1],
        tuple(period_labels),
    )


def choose_output_dir(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = [path / f"feature_window_sweep_{stamp}", Path(tempfile.gettempdir()) / f"feature_window_sweep_{stamp}"]
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            try:
                probe.unlink()
            except OSError:
                pass
            return candidate
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("could not create output directory")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_macd_pairs(value: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        fast, slow = item.split(":")
        pairs.append((int(fast), int(slow)))
    return tuple(pairs)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep weekly technical feature windows")
    parser.add_argument("--daily-file", type=Path, default=Path("dataset/stock_market_19_24.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("logs/feature_window_sweep"))
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--min-train-size", type=int, default=150)
    parser.add_argument("--test-step", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sma-windows", default="2,3,4,5,6,8,12")
    parser.add_argument("--ema-windows", default="2,3,4,5,6,8,12")
    parser.add_argument("--bollinger-windows", default="4,5,6,8,12")
    parser.add_argument("--bollinger-ks", default="1.5,2.0")
    parser.add_argument("--rsi-windows", default="4,5,8,14,20")
    parser.add_argument("--macd-pairs", default="3:6,4:8,5:10,6:12,8:17,12:26")
    parser.add_argument("--hurst-windows", default="12,20,26")
    parser.add_argument("--entropy-windows", default="12,20,26")
    parser.add_argument("--rel-pos-windows", default="2,3,4,5,6,8,12")
    parser.add_argument("--exclude-entropy", action="store_true")
    parser.add_argument("--top-candidates", type=int, default=20)
    parser.add_argument("--max-greedy-features", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=None, help="Debug limit for smoke tests")
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    daily = load_daily(args.daily_file)
    frames, target_min, target_max = build_feature_frames(
        daily,
        sma_windows=parse_int_list(args.sma_windows),
        ema_windows=parse_int_list(args.ema_windows),
        bollinger_windows=parse_int_list(args.bollinger_windows),
        bollinger_ks=parse_float_list(args.bollinger_ks),
        rsi_windows=parse_int_list(args.rsi_windows),
        macd_pairs=parse_macd_pairs(args.macd_pairs),
        hurst_windows=parse_int_list(args.hurst_windows),
        entropy_windows=parse_int_list(args.entropy_windows),
        rel_pos_windows=parse_int_list(args.rel_pos_windows),
        include_entropy=not args.exclude_entropy,
    )
    candidates = build_candidates(frames)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    seeds = parse_int_list(args.seeds)
    output_dir = choose_output_dir(args.outdir)
    print(f"[sweep] device={device}")
    print(f"[sweep] candidates={len(candidates)}")
    print(f"[sweep] output_dir={output_dir}")

    baseline_cache: dict[tuple[str, ...], tuple[float, float, int, int, str, str, tuple[str, ...]]] = {}

    def evaluate_cached(
        feature_names: tuple[str, ...],
        period_filter: set[str] | None = None,
    ) -> tuple[float, float, int, int, str, str, tuple[str, ...]]:
        return evaluate_feature_set(
            frames,
            target_min,
            target_max,
            feature_names,
            lookback=args.lookback,
            min_train_size=args.min_train_size,
            test_step=args.test_step,
            seeds=seeds,
            runs=args.runs,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            rnn_type=args.rnn_type,
            device=device,
            period_filter=period_filter,
        )

    screen_rows: list[EvalResult] = []
    for index, candidate in enumerate(candidates, start=1):
        feature_names = BASELINE_FEATURES + (candidate.name,)
        try:
            candidate_metrics = evaluate_cached(feature_names)
            _, _, _, _, first_period, last_period, period_labels = candidate_metrics
            baseline_key = tuple(period_labels)
            if baseline_key not in baseline_cache:
                baseline_metrics = evaluate_cached(BASELINE_FEATURES, period_filter=set(period_labels))
                baseline_cache[baseline_key] = baseline_metrics
            baseline_mae = baseline_cache[baseline_key][0]
            delta = candidate_metrics[0] - baseline_mae
            row = EvalResult(
                feature_set=f"baseline_plus_{candidate.name}",
                family=candidate.family,
                params=candidate.params,
                features=",".join(feature_names),
                n_features=len(feature_names),
                n_periods=candidate_metrics[2],
                n_sequences=candidate_metrics[3],
                first_period=first_period,
                last_period=last_period,
                mae_interval_mean=candidate_metrics[0],
                mae_interval_std=candidate_metrics[1],
                baseline_same_rows_mae=baseline_mae,
                delta_vs_baseline=delta,
                status="ok",
            )
        except Exception as exc:
            row = EvalResult(
                feature_set=f"baseline_plus_{candidate.name}",
                family=candidate.family,
                params=candidate.params,
                features=",".join(feature_names),
                n_features=len(feature_names),
                n_periods=0,
                n_sequences=0,
                first_period="",
                last_period="",
                mae_interval_mean=math.nan,
                mae_interval_std=math.nan,
                baseline_same_rows_mae=math.nan,
                delta_vs_baseline=math.nan,
                status=f"skipped: {exc}",
            )
        screen_rows.append(row)
        print(
            f"[{index:03d}/{len(candidates):03d}] {candidate.name}: "
            f"mae={row.mae_interval_mean:.6f} delta={row.delta_vs_baseline:.6f} {row.status}"
        )

    screen_df = pd.DataFrame([asdict(row) for row in screen_rows])
    screen_df = screen_df.sort_values(["status", "delta_vs_baseline", "mae_interval_mean"], na_position="last")
    screen_path = output_dir / "single_feature_window_sweep.csv"
    screen_df.to_csv(screen_path, index=False)

    ok_df = screen_df[screen_df["status"].eq("ok") & np.isfinite(screen_df["delta_vs_baseline"])].copy()
    ok_df = ok_df.sort_values("delta_vs_baseline")
    pool = list(ok_df.head(args.top_candidates)["feature_set"].str.replace("^baseline_plus_", "", regex=True))

    selected: list[str] = []
    greedy_rows = []
    best_metrics = evaluate_cached(BASELINE_FEATURES)
    best_mae = best_metrics[0]
    greedy_rows.append(
        {
            "step": 0,
            "added_feature": "",
            "mae_interval_mean": best_mae,
            "mae_interval_std": best_metrics[1],
            "n_periods": best_metrics[2],
            "n_sequences": best_metrics[3],
            "features": ",".join(BASELINE_FEATURES),
        }
    )

    for step in range(1, args.max_greedy_features + 1):
        step_best = None
        for candidate_name in pool:
            if candidate_name in selected:
                continue
            trial_features = BASELINE_FEATURES + tuple(selected) + (candidate_name,)
            try:
                metrics = evaluate_cached(trial_features)
            except Exception:
                continue
            if step_best is None or metrics[0] < step_best[1][0]:
                step_best = (candidate_name, metrics)
        if step_best is None or step_best[1][0] >= best_mae:
            break
        selected.append(step_best[0])
        best_metrics = step_best[1]
        best_mae = best_metrics[0]
        greedy_rows.append(
            {
                "step": step,
                "added_feature": step_best[0],
                "mae_interval_mean": best_mae,
                "mae_interval_std": best_metrics[1],
                "n_periods": best_metrics[2],
                "n_sequences": best_metrics[3],
                "features": ",".join(BASELINE_FEATURES + tuple(selected)),
            }
        )
        print(f"[greedy] step={step} add={step_best[0]} mae={best_mae:.6f}")

    greedy_df = pd.DataFrame(greedy_rows)
    greedy_path = output_dir / "greedy_selected_features.csv"
    greedy_df.to_csv(greedy_path, index=False)

    payload = {
        "args": json_safe(vars(args)),
        "device": str(device),
        "single_feature_csv": str(screen_path),
        "greedy_csv": str(greedy_path),
        "selected_extra_features": selected,
        "selected_all_features": list(BASELINE_FEATURES + tuple(selected)),
        "best_mae_interval_mean": best_mae,
    }
    (output_dir / "selected_feature_set.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[sweep] best")
    print(greedy_df.tail(1).to_string(index=False))
    print(f"[sweep] wrote {screen_path}")
    print(f"[sweep] wrote {greedy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
