"""Data loading and feature engineering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.feature_sets import BASELINE_FEATURES, FEATURE_SET_DEFINITIONS


def _hurst_exponent_simple(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20 or np.std(values) < 1e-12:
        return np.nan

    lags = np.arange(2, 10)
    tau = np.asarray([np.std(values[lag:] - values[:-lag]) for lag in lags])
    if np.any(tau <= 0):
        return np.nan
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def _sample_entropy(values: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
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


class DataLoader:
    """Load and preprocess HOSE stock data."""

    DATE_COLUMN = "History Price / Date"

    def __init__(self, daily_file_path: str | Path, target_file_path: str | Path | None = None):
        self.daily_file_path = Path(daily_file_path)
        self.target_file_path = Path(target_file_path) if target_file_path else None
        self.stock_codes: list[str] | None = None
        self.df_daily: pd.DataFrame | None = None
        self.df_daily_merged: pd.DataFrame | None = None

    @staticmethod
    def clean_float(value: Any) -> float:
        """Convert mixed spreadsheet-style values to float safely."""

        try:
            if isinstance(value, str):
                cleaned = value.replace(",", "").replace("#DIV/0!", "").strip()
                if cleaned in {"", "-"}:
                    return 0.0
                return float(cleaned)
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def clean_price(value: Any) -> float:
        """Convert a price cell to float."""

        if isinstance(value, str):
            return float(value.replace(",", "").strip())
        return float(value)

    def load_daily_data(self) -> pd.DataFrame:
        """Read the daily price table and add canonical time columns."""

        if not self.daily_file_path.exists():
            raise FileNotFoundError(f"Daily price file not found: {self.daily_file_path}")

        frame = pd.read_csv(self.daily_file_path)
        if self.DATE_COLUMN not in frame.columns:
            raise ValueError(
                f"Expected '{self.DATE_COLUMN}' in {self.daily_file_path.name}, "
                f"but only found: {list(frame.columns[:5])}"
            )

        frame["Date"] = pd.to_datetime(frame[self.DATE_COLUMN], errors="coerce")
        frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        frame["Quarter"] = frame["Date"].dt.quarter
        frame["Year"] = frame["Date"].dt.year
        iso_calendar = frame["Date"].dt.isocalendar()
        frame["ISO_Year"] = iso_calendar.year.astype(int)
        frame["ISO_Week"] = iso_calendar.week.astype(int)
        frame["Year_Week"] = (
            frame["ISO_Year"].astype(str) + "_W" + frame["ISO_Week"].astype(str).str.zfill(2)
        )

        non_stock_columns = {
            self.DATE_COLUMN,
            "Date",
            "Quarter",
            "Year",
            "ISO_Year",
            "ISO_Week",
            "Year_Week",
        }
        self.stock_codes = [column for column in frame.columns if column not in non_stock_columns]

        clean_prices = frame[self.stock_codes].apply(lambda column: column.map(self.clean_price)).ffill().bfill()
        self.df_daily = frame
        self.df_daily_merged = pd.concat(
            [frame[["Date", "Year", "Quarter", "ISO_Year", "ISO_Week", "Year_Week"]], clean_prices],
            axis=1,
        )
        return self.df_daily_merged

    def load_target_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the optional min/max return target file."""

        if self.target_file_path is None:
            raise ValueError("Target file path not provided.")
        if self.stock_codes is None:
            raise ValueError("Load daily data before loading target data.")

        try:
            target_multi = pd.read_csv(self.target_file_path, header=[0, 1])
            min_columns = [
                column
                for column in target_multi.columns
                if str(column[0]).strip().startswith("Min Return")
            ]
            max_columns = [
                column
                for column in target_multi.columns
                if str(column[0]).strip().startswith("Max Return")
            ]
            min_by_ticker = {str(column[1]).strip(): column for column in min_columns}
            max_by_ticker = {str(column[1]).strip(): column for column in max_columns}
            if all(ticker in min_by_ticker and ticker in max_by_ticker for ticker in self.stock_codes):
                min_returns = target_multi[[min_by_ticker[ticker] for ticker in self.stock_codes]].copy()
                max_returns = target_multi[[max_by_ticker[ticker] for ticker in self.stock_codes]].copy()
                min_returns.columns = self.stock_codes
                max_returns.columns = self.stock_codes
                min_returns = min_returns.apply(lambda column: column.map(self.clean_float))
                max_returns = max_returns.apply(lambda column: column.map(self.clean_float))
                return min_returns, max_returns
        except (pd.errors.ParserError, ValueError):
            pass

        target_frame = pd.read_csv(self.target_file_path, header=1).apply(
            lambda column: column.map(self.clean_float)
        )
        num_stocks = len(self.stock_codes)
        min_start = 2
        min_end = min_start + num_stocks
        max_end = min_end + num_stocks
        if target_frame.shape[1] < max_end:
            min_start = 1
            min_end = min_start + num_stocks
            max_end = min_end + num_stocks
        min_returns = target_frame.iloc[:, min_start:min_end].copy()
        max_returns = target_frame.iloc[:, min_end:max_end].copy()
        min_returns.columns = self.stock_codes
        max_returns.columns = self.stock_codes
        return min_returns, max_returns

    def get_date_range(self, year_start: int, q_start: int, year_end: int, q_end: int) -> pd.DataFrame:
        """Return price data between two quarter labels."""

        if self.df_daily_merged is None or self.stock_codes is None:
            raise ValueError("Daily data not loaded. Call load_daily_data() first.")

        frame = self.df_daily_merged
        if year_start == year_end:
            mask = (
                (frame["Year"] == year_start)
                & (frame["Quarter"] >= q_start)
                & (frame["Quarter"] <= q_end)
            )
        else:
            mask = (
                ((frame["Year"] == year_start) & (frame["Quarter"] >= q_start))
                | ((frame["Year"] == year_end) & (frame["Quarter"] <= q_end))
                | ((frame["Year"] > year_start) & (frame["Year"] < year_end))
            )
        return frame.loc[mask, self.stock_codes]

    def get_week_range(self, start_week: str, end_week: str) -> pd.DataFrame:
        """Return price data between two ISO week labels."""

        if self.df_daily_merged is None or self.stock_codes is None:
            raise ValueError("Daily data not loaded. Call load_daily_data() first.")

        frame = self.df_daily_merged
        mask = (frame["Year_Week"] >= start_week) & (frame["Year_Week"] <= end_week)
        return frame.loc[mask, self.stock_codes]

    def get_stock_codes(self) -> list[str]:
        """Expose the detected ticker list."""

        if self.stock_codes is None:
            raise ValueError("Load daily data before requesting stock codes.")
        return self.stock_codes


class FeatureEngineer:
    """Create quarterly or weekly tensors from the raw price table."""

    def __init__(self, df_daily_merged: pd.DataFrame, stock_codes: list[str], aggregation_mode: str = "quarterly"):
        if aggregation_mode not in {"quarterly", "weekly"}:
            raise ValueError("aggregation_mode must be either 'quarterly' or 'weekly'.")

        self.df_daily_merged = df_daily_merged.copy()
        self.stock_codes = stock_codes
        self.aggregation_mode = aggregation_mode
        self.scaler = StandardScaler()

        if "Date" in self.df_daily_merged.columns:
            self.df_daily_merged["Date"] = pd.to_datetime(self.df_daily_merged["Date"])

    def _grouped_prices(self) -> Any:
        if self.aggregation_mode == "weekly":
            return self.df_daily_merged.groupby("Year_Week")[self.stock_codes]
        return self.df_daily_merged.groupby(["Year", "Quarter"])[self.stock_codes]

    def compute_features(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute the four node-level input features."""

        feature_frames = self.compute_feature_frames("baseline4")
        return (
            feature_frames["f_std"],
            feature_frames["f_mean"],
            feature_frames["f_return"],
            feature_frames["f_skew"],
        )

    def compute_feature_frames(self, feature_set: str = "baseline4") -> dict[str, pd.DataFrame]:
        """Compute all feature frames required by a named feature set."""

        if feature_set not in FEATURE_SET_DEFINITIONS:
            supported = ", ".join(FEATURE_SET_DEFINITIONS)
            raise ValueError(f"Unsupported feature_set '{feature_set}'. Choose one of: {supported}.")

        grouped = self._grouped_prices()
        input_std = grouped.std().fillna(0.0)
        input_mean = grouped.mean().fillna(0.0)
        price_start = grouped.first().fillna(0.0)
        price_high = grouped.max().fillna(0.0)
        price_low = grouped.min().fillna(0.0)
        price_end = grouped.last().fillna(0.0)

        epsilon = 1e-8
        feature_frames: dict[str, pd.DataFrame] = {
            "f_std": np.log1p(input_std),
            "f_mean": np.log1p(input_mean),
            "f_return": np.log((price_end + epsilon) / (price_start + epsilon)),
            "f_skew": grouped.agg(lambda series: series.pct_change().dropna().skew()).fillna(0.0),
        }

        selected_features = set(FEATURE_SET_DEFINITIONS[feature_set])
        if selected_features.difference(BASELINE_FEATURES):
            simple_return = (price_end - price_start) / (price_start + epsilon)
            market_return = simple_return.mean(axis=1)
            period_range = (price_high - price_low) / (price_start + epsilon)

            rolling_high_4 = price_high.rolling(4, min_periods=4).max()
            rolling_low_4 = price_low.rolling(4, min_periods=4).min()
            sma_4 = price_end.rolling(4, min_periods=4).mean()
            ema_4 = price_end.ewm(span=4, adjust=False, min_periods=4).mean()
            rolling_std_4 = price_end.rolling(4, min_periods=4).std()
            bollinger_upper_4 = sma_4 + 2.0 * rolling_std_4
            bollinger_lower_4 = sma_4 - 2.0 * rolling_std_4

            price_delta = price_end.diff()
            gain = price_delta.clip(lower=0.0)
            loss = (-price_delta).clip(lower=0.0)
            avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
            rsi_14 = 100.0 - (100.0 / (1.0 + (avg_gain / (avg_loss + 1e-12))))

            ema_fast_12 = price_end.ewm(span=12, adjust=False, min_periods=12).mean()
            ema_slow_26 = price_end.ewm(span=26, adjust=False, min_periods=26).mean()
            feature_frames.update(
                {
                    "alpha_excess": simple_return.sub(market_return, axis=0),
                    "weekly_range_lag1": period_range.shift(1),
                    "rel_pos_4w": (price_end - rolling_low_4) / (rolling_high_4 - rolling_low_4 + epsilon),
                    "sma_ratio_4": (price_end / (sma_4 + epsilon)) - 1.0,
                    "ema_ratio_4": (price_end / (ema_4 + epsilon)) - 1.0,
                    "bollinger_width_4": (bollinger_upper_4 - bollinger_lower_4) / (sma_4 + epsilon),
                    "bollinger_percent_b_4": (
                        (price_end - bollinger_lower_4) / (bollinger_upper_4 - bollinger_lower_4 + epsilon)
                    ),
                    "rsi_14": rsi_14,
                    "macd_norm": (ema_fast_12 - ema_slow_26) / (price_end + epsilon),
                }
            )
            if "hurst_20" in selected_features:
                log_close = np.log(price_end + epsilon)
                feature_frames["hurst_20"] = log_close.rolling(20, min_periods=20).apply(
                    lambda values: _hurst_exponent_simple(np.asarray(values, dtype=float)),
                    raw=False,
                )
            if selected_features & {
                "sma_ratio_w12",
                "rsi_w20",
                "hurst_w26",
                "bollinger_width_w12_k2p0",
                "macd_norm_f12_s26",
                "rel_pos_w12",
            }:
                sma_12 = price_end.rolling(12, min_periods=12).mean()
                rolling_std_12 = price_end.rolling(12, min_periods=12).std()
                rolling_high_12 = price_high.rolling(12, min_periods=12).max()
                rolling_low_12 = price_low.rolling(12, min_periods=12).min()
                ema_fast_12w = price_end.ewm(span=12, adjust=False, min_periods=12).mean()
                ema_slow_26w = price_end.ewm(span=26, adjust=False, min_periods=26).mean()
                price_delta_20 = price_end.diff()
                gain_20 = price_delta_20.clip(lower=0.0)
                loss_20 = (-price_delta_20).clip(lower=0.0)
                avg_gain_20 = gain_20.ewm(alpha=1.0 / 20, adjust=False, min_periods=20).mean()
                avg_loss_20 = loss_20.ewm(alpha=1.0 / 20, adjust=False, min_periods=20).mean()
                bollinger_upper_12 = sma_12 + 2.0 * rolling_std_12
                bollinger_lower_12 = sma_12 - 2.0 * rolling_std_12

                feature_frames.update(
                    {
                        "sma_ratio_w12": (price_end / (sma_12 + epsilon)) - 1.0,
                        "rsi_w20": 100.0
                        - (100.0 / (1.0 + (avg_gain_20 / (avg_loss_20 + 1e-12)))),
                        "bollinger_width_w12_k2p0": (
                            (bollinger_upper_12 - bollinger_lower_12) / (sma_12 + epsilon)
                        ),
                        "macd_norm_f12_s26": (ema_fast_12w - ema_slow_26w) / (price_end + epsilon),
                        "rel_pos_w12": (
                            (price_end - rolling_low_12) / (rolling_high_12 - rolling_low_12 + epsilon)
                        ),
                    }
                )
                if "hurst_w26" in selected_features:
                    log_close = np.log(price_end + epsilon)
                    feature_frames["hurst_w26"] = log_close.rolling(26, min_periods=26).apply(
                        lambda values: _hurst_exponent_simple(np.asarray(values, dtype=float)),
                        raw=False,
                    )
            if "entropy_20" in selected_features:
                log_return = np.log((price_end + epsilon) / (price_end.shift(1) + epsilon))
                feature_frames["entropy_20"] = log_return.rolling(20, min_periods=20).apply(
                    lambda values: _sample_entropy(np.asarray(values, dtype=float)),
                    raw=False,
                )

        return {name: feature_frames[name] for name in FEATURE_SET_DEFINITIONS[feature_set]}

    def compute_targets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute min/max return targets for each aggregation period."""

        grouped = self._grouped_prices()
        price_start = grouped.first().fillna(0.0)
        price_min = grouped.min().fillna(0.0)
        price_max = grouped.max().fillna(0.0)

        target_min = (price_min - price_start) / (price_start + 1e-8)
        target_max = (price_max - price_start) / (price_start + 1e-8)
        return target_min, target_max

    def create_tensors(
        self,
        feat_std: pd.DataFrame,
        feat_mean: pd.DataFrame,
        feat_return: pd.DataFrame,
        feat_skew: pd.DataFrame,
        target_min: pd.DataFrame,
        target_max: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
        """Create raw PyTorch tensors aligned on common time periods.

        Scaling is intentionally deferred until after a chronological train split
        is known, so the scaler can be fitted on past data only.
        """

        feature_frames = {
            "f_std": feat_std,
            "f_mean": feat_mean,
            "f_return": feat_return,
            "f_skew": feat_skew,
        }
        return self.create_tensors_from_features(
            feature_frames=feature_frames,
            target_min=target_min,
            target_max=target_max,
            feature_names=BASELINE_FEATURES,
        )

    def create_tensors_from_features(
        self,
        feature_frames: dict[str, pd.DataFrame],
        target_min: pd.DataFrame,
        target_max: pd.DataFrame,
        feature_names: tuple[str, ...] | list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
        """Create raw tensors from an ordered collection of feature frames."""

        required_indices = [set(frame.index) for frame in feature_frames.values()]
        required_indices.extend([set(target_min.index), set(target_max.index)])
        common_indices = sorted(set.intersection(*required_indices))

        x_values: list[np.ndarray] = []
        y_values: list[np.ndarray] = []
        valid_indices_map: list[Any] = []

        for idx in common_indices:
            feature_stack = np.stack(
                [feature_frames[name].loc[idx].to_numpy() for name in feature_names],
                axis=1,
            ).astype(np.float32)
            target_stack = np.stack(
                [target_min.loc[idx].to_numpy(), target_max.loc[idx].to_numpy()],
                axis=1,
            ).astype(np.float32)

            if not np.isfinite(feature_stack).all() or not np.isfinite(target_stack).all():
                continue

            x_values.append(feature_stack)
            y_values.append(target_stack)
            valid_indices_map.append(idx)

        if not x_values:
            x_full = torch.empty((0, len(self.stock_codes), len(feature_names)), dtype=torch.float32)
            y_full = torch.empty((0, len(self.stock_codes), 2), dtype=torch.float32)
            return x_full, y_full, valid_indices_map

        x_full = torch.tensor(np.asarray(x_values, dtype=np.float32), dtype=torch.float32)
        y_full = torch.tensor(np.asarray(y_values, dtype=np.float32), dtype=torch.float32)
        return x_full, y_full, valid_indices_map

    @staticmethod
    def fit_scaler_on_train_sequences(
        x_full: torch.Tensor,
        train_sequence_count: int,
        window_size: int,
    ) -> StandardScaler:
        """Fit a scaler using only periods visible to the training sequences."""

        if train_sequence_count < 1:
            raise ValueError("train_sequence_count must be at least 1 to fit the feature scaler.")
        if window_size < 1:
            raise ValueError("window_size must be at least 1 to fit the feature scaler.")

        num_periods = int(x_full.size(0))
        train_period_end = min(num_periods, train_sequence_count + window_size - 1)
        train_slice = x_full[:train_period_end].detach().cpu().numpy()
        _, _, num_features = train_slice.shape

        scaler = StandardScaler()
        scaler.fit(train_slice.reshape(-1, num_features))
        return scaler

    @staticmethod
    def transform_with_scaler(x_full: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
        """Apply a fitted scaler to the full chronological feature tensor."""

        x_np = x_full.detach().cpu().numpy()
        num_periods, num_nodes, num_features = x_np.shape
        x_scaled = scaler.transform(x_np.reshape(-1, num_features))
        return torch.tensor(
            x_scaled.reshape(num_periods, num_nodes, num_features),
            dtype=torch.float32,
        )
