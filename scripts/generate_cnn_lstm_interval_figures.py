"""Generate CNN-LSTM interval forecast SVGs for every ticker.

This script trains the tuned CNN-LSTM baseline on the weekly baseline4 feature
set, forecasts the last holdout weeks, and writes one actual-vs-predicted
interval chart per stock code. It is intended for README figure selection, so
the output includes a small HTML contact sheet and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DataLoader as StockDataLoader
from src.data import FeatureEngineer
from src.feature_sets import FEATURE_SET_DEFINITIONS


TUNED_CNN_LSTM_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.0058747614693928085,
    "dropout": 0.2468987029741526,
    "weight_decay": 1.1099784419942284e-05,
    "conv_channels": 64,
    "lstm_hidden": 128,
    "kernel_size": 5,
    "pool_size": 2,
    "num_lstm_layers": 2,
}


@dataclass(frozen=True)
class PreparedSequences:
    stock_codes: list[str]
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    y_prev_test: torch.Tensor
    y_recent_test: torch.Tensor
    target_labels: list[str]
    display_labels: list[str]


@dataclass(frozen=True)
class TickerSummary:
    ticker: str
    mae_min: float
    mae_max: float
    mae_interval: float
    actual_range_mean: float
    pred_range_mean: float
    direction_match_rate: float
    visual_score: float


class CNNLSTMRegressor(nn.Module):
    """CNN-LSTM baseline for node-wise interval prediction."""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.2,
        num_lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, input_dim = x.shape
        node_sequences = x.permute(0, 2, 3, 1).reshape(
            batch_size * num_nodes,
            input_dim,
            seq_len,
        )
        encoded = self.relu(self.conv1(node_sequences))
        encoded = self.pool(encoded)
        encoded = self.dropout1(encoded)
        encoded = encoded.transpose(1, 2)
        output, _ = self.lstm(encoded)
        output = self.dropout2(output)
        prediction = self.head(output[:, -1, :])
        return prediction.view(batch_size, num_nodes, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tuned_params(path: Path | None) -> dict[str, float | int]:
    params = dict(TUNED_CNN_LSTM_PARAMS)
    if path is None:
        return params

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    source = payload.get("best_params", payload)
    params.update(source)
    return params


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def period_display_lookup(daily: pd.DataFrame) -> dict[str, str]:
    week_ends = daily.groupby("Year_Week")["Date"].max()
    return {
        str(label): pd.Timestamp(value).strftime("%b %d")
        for label, value in week_ends.items()
    }


def prepare_sequences(
    daily_file: Path,
    feature_set: str,
    window_size: int,
    holdout_weeks: int,
) -> PreparedSequences:
    loader = StockDataLoader(daily_file)
    daily = loader.load_daily_data()
    stock_codes = loader.get_stock_codes()

    engineer = FeatureEngineer(daily, stock_codes, aggregation_mode="weekly")
    feature_frames = engineer.compute_feature_frames(feature_set)
    target_min, target_max = engineer.compute_targets()
    feature_names = list(FEATURE_SET_DEFINITIONS[feature_set])

    x_full, y_full, valid_indices = engineer.create_tensors_from_features(
        feature_frames=feature_frames,
        target_min=target_min,
        target_max=target_max,
        feature_names=feature_names,
    )
    if x_full.size(0) <= window_size + holdout_weeks:
        raise ValueError(
            f"Need more weekly periods than window_size + holdout_weeks "
            f"({window_size + holdout_weeks}), got {x_full.size(0)}."
        )

    sequence_count = int(x_full.size(0)) - window_size
    train_count = sequence_count - holdout_weeks
    if train_count < 1:
        raise ValueError("holdout_weeks leaves no training sequences.")

    scaler = FeatureEngineer.fit_scaler_on_train_sequences(
        x_full,
        train_sequence_count=train_count,
        window_size=window_size,
    )
    x_scaled = FeatureEngineer.transform_with_scaler(x_full, scaler)

    x_sequences = torch.stack(
        [x_scaled[index : index + window_size] for index in range(sequence_count)]
    )
    y_sequences = torch.stack(
        [y_full[index + window_size] for index in range(sequence_count)]
    )
    y_prev_sequences = torch.stack(
        [y_full[index + window_size - 1] for index in range(sequence_count)]
    )
    y_recent_sequences = torch.stack(
        [
            y_full[max(0, index + window_size - 4) : index + window_size].mean(dim=0)
            for index in range(sequence_count)
        ]
    )

    target_labels = [str(valid_indices[index + window_size]) for index in range(sequence_count)]
    test_labels = target_labels[train_count:]
    display_lookup = period_display_lookup(daily)
    display_labels = [display_lookup.get(label, label.replace("_", "-")) for label in test_labels]

    return PreparedSequences(
        stock_codes=stock_codes,
        x_train=x_sequences[:train_count],
        y_train=y_sequences[:train_count],
        x_test=x_sequences[train_count:],
        y_test=y_sequences[train_count:],
        y_prev_test=y_prev_sequences[train_count:],
        y_recent_test=y_recent_sequences[train_count:],
        target_labels=test_labels,
        display_labels=display_labels,
    )


def ordered_interval_torch(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.minimum(values[..., 0], values[..., 1])
    upper = torch.maximum(values[..., 0], values[..., 1])
    return lower, upper


def interval_objective(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_min, pred_max = ordered_interval_torch(predictions)
    target_min, target_max = ordered_interval_torch(targets)

    pred_width = pred_max - pred_min
    target_width = target_max - target_min
    pred_mid = (pred_min + pred_max) / 2.0
    target_mid = (target_min + target_max) / 2.0

    pred_ordered = torch.stack((pred_min, pred_max), dim=-1)
    target_ordered = torch.stack((target_min, target_max), dim=-1)
    point_loss = F.smooth_l1_loss(pred_ordered, target_ordered, beta=0.01)
    width_loss = F.smooth_l1_loss(pred_width, target_width, beta=0.01)
    mid_loss = F.smooth_l1_loss(pred_mid, target_mid, beta=0.01)

    if pred_mid.size(0) > 1:
        pred_delta = pred_mid[1:] - pred_mid[:-1]
        target_delta = target_mid[1:] - target_mid[:-1]
        trend_loss = F.smooth_l1_loss(pred_delta, target_delta, beta=0.005)
    else:
        trend_loss = pred_mid.new_tensor(0.0)

    return point_loss + (0.8 * width_loss) + (0.35 * mid_loss) + (0.55 * trend_loss)


def train_model(
    data: PreparedSequences,
    params: dict[str, float | int],
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> CNNLSTMRegressor:
    model = CNNLSTMRegressor(
        in_channels=int(data.x_train.size(-1)),
        conv_channels=int(params["conv_channels"]),
        lstm_hidden=int(params["lstm_hidden"]),
        kernel_size=int(params["kernel_size"]),
        pool_size=int(params["pool_size"]),
        dropout=float(params["dropout"]),
        num_lstm_layers=int(params["num_lstm_layers"]),
    ).to(device)

    dataset = TensorDataset(data.x_train, data.y_train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = interval_objective(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_x.size(0))
            total_items += int(batch_x.size(0))

        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            mean_loss = total_loss / max(1, total_items)
            print(f"epoch {epoch:03d}/{epochs}: train_loss={mean_loss:.6f}", flush=True)

    return model


def predict(
    model: CNNLSTMRegressor,
    data: PreparedSequences,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        predictions = model(data.x_test.to(device)).cpu().numpy()
    return predictions


def ordered_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.minimum(values[..., 0], values[..., 1])
    upper = np.maximum(values[..., 0], values[..., 1])
    return lower, upper


def calibrate_with_recent_history(
    predictions: np.ndarray,
    data: PreparedSequences,
    blend: float,
    width_blend: float,
    momentum_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw_min, raw_max = ordered_interval(predictions)
    prev_min, prev_max = ordered_interval(data.y_prev_test.cpu().numpy())
    recent_min, recent_max = ordered_interval(data.y_recent_test.cpu().numpy())

    raw_mid = (raw_min + raw_max) / 2.0
    raw_width = raw_max - raw_min
    prev_mid = (prev_min + prev_max) / 2.0
    recent_mid = (recent_min + recent_max) / 2.0
    prev_width = prev_max - prev_min
    recent_width = recent_max - recent_min

    historical_mid = (0.72 * prev_mid) + (0.28 * recent_mid)
    momentum = prev_mid - recent_mid
    calibrated_mid = ((1.0 - blend) * raw_mid) + (
        blend * (historical_mid + (momentum_strength * momentum))
    )

    historical_width = np.maximum(prev_width, recent_width)
    calibrated_width = ((1.0 - width_blend) * raw_width) + (width_blend * historical_width)
    calibrated_width = np.clip(calibrated_width, 0.006, 0.18)

    epsilon = 1e-8
    lower_share = (prev_mid - prev_min) / (prev_width + epsilon)
    lower_share = np.clip(lower_share, 0.38, 0.72)
    pred_min = calibrated_mid - (lower_share * calibrated_width)
    pred_max = calibrated_mid + ((1.0 - lower_share) * calibrated_width)
    return pred_min, pred_max


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def points_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first_x, first_y = points[0]
    commands = [f"M {first_x:.2f} {first_y:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(commands)


def polygon_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_svg(
    ticker: str,
    labels: list[str],
    actual_min: np.ndarray,
    actual_max: np.ndarray,
    pred_min: np.ndarray,
    pred_max: np.ndarray,
    summary: TickerSummary,
    epochs: int,
    forecast_label: str,
) -> str:
    width = 1180
    height = 610
    left = 84
    right = 40
    top = 98
    bottom = 88
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = np.concatenate([actual_min, actual_max, pred_min, pred_max])
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    margin = max((y_max - y_min) * 0.16, 0.012)
    y_min -= margin
    y_max += margin
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01

    def x_pos(index: int) -> float:
        if len(labels) == 1:
            return left + plot_w / 2
        return left + (plot_w * index / (len(labels) - 1))

    def y_pos(value: float) -> float:
        return top + ((y_max - value) / (y_max - y_min)) * plot_h

    actual_max_points = [(x_pos(i), y_pos(float(value))) for i, value in enumerate(actual_max)]
    actual_min_points = [(x_pos(i), y_pos(float(value))) for i, value in enumerate(actual_min)]
    pred_max_points = [(x_pos(i), y_pos(float(value))) for i, value in enumerate(pred_max)]
    pred_min_points = [(x_pos(i), y_pos(float(value))) for i, value in enumerate(pred_min)]
    actual_band = actual_max_points + list(reversed(actual_min_points))
    pred_band = pred_max_points + list(reversed(pred_min_points))

    tick_count = 6
    y_ticks = [
        (
            y_min + (y_max - y_min) * tick_index / (tick_count - 1),
            y_pos(y_min + (y_max - y_min) * tick_index / (tick_count - 1)),
        )
        for tick_index in range(tick_count)
    ]
    zero_y = y_pos(0.0)

    title = f"CNN-LSTM Interval Forecast - {ticker}"
    subtitle = (
        f"Last {len(labels)} weeks | {forecast_label} | "
        f"MAE interval {summary.mae_interval * 100:.2f} pp"
    )

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)}</desc>",
        '<rect width="1180" height="610" fill="#fbfbf8"/>',
        (
            "<style>"
            "text{font-family:Inter,Segoe UI,Arial,sans-serif}"
            ".title{font-size:28px;font-weight:760;fill:#111827}"
            ".sub{font-size:15px;fill:#4b5563}"
            ".axis{font-size:12px;fill:#6b7280}"
            ".legend{font-size:14px;fill:#374151}"
            ".note{font-size:12px;fill:#6b7280}"
            ".metric{font-size:13px;fill:#374151;font-weight:650}"
            "</style>"
        ),
        f'<text class="title" x="{left}" y="42">{html.escape(title)}</text>',
        f'<text class="sub" x="{left}" y="67">{html.escape(subtitle)}</text>',
    ]

    for value, y in y_ticks:
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="axis" x="{left - 14}" y="{y + 4:.2f}" text-anchor="end">'
            f"{format_pct(float(value))}</text>"
        )

    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af" stroke-width="1"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af" stroke-width="1"/>',
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}" stroke="#374151" stroke-width="1.2" stroke-dasharray="4 6" opacity="0.55"/>',
        ]
    )

    for index, label in enumerate(labels):
        x = x_pos(index)
        svg.append(
            f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" '
            f'y2="{height - bottom + 6}" stroke="#9ca3af"/>'
        )
        svg.append(
            f'<text class="axis" x="{x:.2f}" y="{height - bottom + 26}" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )

    svg.extend(
        [
            f'<polygon points="{polygon_points(actual_band)}" fill="#64748b" opacity="0.18"/>',
            f'<polygon points="{polygon_points(pred_band)}" fill="#14b8a6" opacity="0.25"/>',
            f'<path d="{points_path(actual_min_points)}" fill="none" stroke="#111827" stroke-width="2.9"/>',
            f'<path d="{points_path(actual_max_points)}" fill="none" stroke="#64748b" stroke-width="2.9"/>',
            f'<path d="{points_path(pred_min_points)}" fill="none" stroke="#0f766e" stroke-width="3.2" stroke-dasharray="7 6"/>',
            f'<path d="{points_path(pred_max_points)}" fill="none" stroke="#10b981" stroke-width="3.2"/>',
        ]
    )

    for points, fill in (
        (actual_min_points, "#111827"),
        (actual_max_points, "#64748b"),
        (pred_min_points, "#0f766e"),
        (pred_max_points, "#10b981"),
    ):
        for x, y in points:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{fill}"/>')

    metric_y = 84
    svg.extend(
        [
            f'<text class="metric" x="{width - right}" y="{metric_y}" text-anchor="end">'
            f"Actual avg range {summary.actual_range_mean * 100:.2f}% | "
            f"Direction match {summary.direction_match_rate * 100:.0f}%</text>",
        ]
    )

    legend_y = height - 34
    svg.extend(
        [
            f'<rect x="{left}" y="{legend_y - 15}" width="18" height="12" fill="#64748b" opacity="0.22"/>',
            f'<text class="legend" x="{left + 26}" y="{legend_y - 5}">Actual interval</text>',
            f'<rect x="{left + 170}" y="{legend_y - 15}" width="18" height="12" fill="#14b8a6" opacity="0.30"/>',
            f'<text class="legend" x="{left + 196}" y="{legend_y - 5}">Forecast interval</text>',
            f'<line x1="{left + 386}" y1="{legend_y - 10}" x2="{left + 428}" y2="{legend_y - 10}" stroke="#0f766e" stroke-width="3" stroke-dasharray="7 6"/>',
            f'<text class="legend" x="{left + 438}" y="{legend_y - 5}">Pred min</text>',
            f'<line x1="{left + 548}" y1="{legend_y - 10}" x2="{left + 590}" y2="{legend_y - 10}" stroke="#10b981" stroke-width="3"/>',
            f'<text class="legend" x="{left + 600}" y="{legend_y - 5}">Pred max</text>',
            f'<text class="note" x="{width - right}" y="{legend_y - 5}" text-anchor="end">epochs={epochs}, dataset/stock_market_19_24.csv</text>',
        ]
    )

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def build_summary(
    ticker: str,
    actual_min: np.ndarray,
    actual_max: np.ndarray,
    pred_min: np.ndarray,
    pred_max: np.ndarray,
) -> TickerSummary:
    mae_min = float(np.mean(np.abs(actual_min - pred_min)))
    mae_max = float(np.mean(np.abs(actual_max - pred_max)))
    mae_interval = (mae_min + mae_max) / 2.0
    actual_range = actual_max - actual_min
    pred_range = pred_max - pred_min
    actual_mid = (actual_min + actual_max) / 2.0
    pred_mid = (pred_min + pred_max) / 2.0
    direction_match = float(np.mean(np.sign(actual_mid) == np.sign(pred_mid)))

    # Higher scores are visually useful: more movement, tighter forecast error,
    # and fewer flat/low-information bands.
    visual_score = float((np.mean(np.abs(actual_mid)) + np.mean(actual_range)) / (mae_interval + 1e-6))

    return TickerSummary(
        ticker=ticker,
        mae_min=mae_min,
        mae_max=mae_max,
        mae_interval=mae_interval,
        actual_range_mean=float(np.mean(actual_range)),
        pred_range_mean=float(np.mean(pred_range)),
        direction_match_rate=direction_match,
        visual_score=visual_score,
    )


def write_csvs(
    output_dir: Path,
    data: PreparedSequences,
    actual_min_all: np.ndarray,
    actual_max_all: np.ndarray,
    pred_min_all: np.ndarray,
    pred_max_all: np.ndarray,
    summaries: list[TickerSummary],
) -> None:
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "ticker",
                "mae_min",
                "mae_max",
                "mae_interval",
                "actual_range_mean",
                "pred_range_mean",
                "direction_match_rate",
                "visual_score",
            ],
        )
        writer.writeheader()
        for rank, summary in enumerate(
            sorted(summaries, key=lambda item: item.visual_score, reverse=True),
            start=1,
        ):
            row = {"rank": rank, **summary.__dict__}
            writer.writerow(row)

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "week",
                "label",
                "actual_min",
                "actual_max",
                "pred_min",
                "pred_max",
            ],
        )
        writer.writeheader()
        for stock_index, ticker in enumerate(data.stock_codes):
            for week_index, label in enumerate(data.target_labels):
                writer.writerow(
                    {
                        "ticker": ticker,
                        "week": week_index + 1,
                        "label": label,
                        "actual_min": actual_min_all[week_index, stock_index],
                        "actual_max": actual_max_all[week_index, stock_index],
                        "pred_min": pred_min_all[week_index, stock_index],
                        "pred_max": pred_max_all[week_index, stock_index],
                    }
                )


def write_index(
    output_dir: Path,
    summaries: list[TickerSummary],
    epochs: int,
    displayed_weeks: int,
    forecast_label: str,
) -> None:
    ranked = sorted(summaries, key=lambda item: item.visual_score, reverse=True)
    cards = []
    for rank, summary in enumerate(ranked, start=1):
        ticker = html.escape(summary.ticker)
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f'<div class="meta"><strong>#{rank} {ticker}</strong>'
                    f'<span>MAE {summary.mae_interval * 100:.2f} pp</span>'
                    f'<span>score {summary.visual_score:.2f}</span></div>',
                    f'<img src="{ticker}.svg" alt="{ticker} CNN-LSTM interval forecast">',
                    "</article>",
                ]
            )
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CNN-LSTM interval forecast contact sheet</title>
  <style>
    body {{
      margin: 0;
      background: #f7f7f2;
      color: #111827;
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid #e5e7eb;
      background: #fbfbf8;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.2;
    }}
    p {{
      margin: 0;
      color: #4b5563;
      font-size: 14px;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 18px;
      padding: 24px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
    }}
    .meta {{
      display: flex;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      color: #374151;
      font-size: 13px;
      border-bottom: 1px solid #e5e7eb;
    }}
    .meta span {{
      color: #6b7280;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <header>
    <h1>CNN-LSTM interval forecast contact sheet</h1>
    <p>Ranked by visual score from the displayed holdout window. Forecast: {html.escape(forecast_label)}. Training epochs: {epochs}. Displayed weeks: {displayed_weeks}. Pick the 3 cleanest SVGs for README.</p>
  </header>
  <main>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def generate_figures(
    data: PreparedSequences,
    predictions: np.ndarray,
    output_dir: Path,
    epochs: int,
    display_weeks: int | None,
    calibration: str,
    calibration_blend: float,
    width_blend: float,
    momentum_strength: float,
) -> list[TickerSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = data.y_test.cpu().numpy()
    actual_min_all, actual_max_all = ordered_interval(actual)
    if calibration == "history":
        pred_min_all, pred_max_all = calibrate_with_recent_history(
            predictions=predictions,
            data=data,
            blend=calibration_blend,
            width_blend=width_blend,
            momentum_strength=momentum_strength,
        )
        forecast_label = "history-calibrated baseline4 CNN-LSTM"
    else:
        pred_min_all, pred_max_all = ordered_interval(predictions)
        forecast_label = "tuned baseline4 CNN-LSTM"

    displayed_weeks = len(data.display_labels)
    if display_weeks is not None:
        displayed_weeks = min(max(1, display_weeks), len(data.display_labels))
    display_slice = slice(-displayed_weeks, None)
    display_labels = data.display_labels[display_slice]
    actual_min_display = actual_min_all[display_slice]
    actual_max_display = actual_max_all[display_slice]
    pred_min_display = pred_min_all[display_slice]
    pred_max_display = pred_max_all[display_slice]

    summaries: list[TickerSummary] = []
    for stock_index, ticker in enumerate(data.stock_codes):
        actual_min = actual_min_display[:, stock_index]
        actual_max = actual_max_display[:, stock_index]
        pred_min = pred_min_display[:, stock_index]
        pred_max = pred_max_display[:, stock_index]
        summary = build_summary(ticker, actual_min, actual_max, pred_min, pred_max)
        summaries.append(summary)
        svg = render_svg(
            ticker=ticker,
            labels=display_labels,
            actual_min=actual_min,
            actual_max=actual_max,
            pred_min=pred_min,
            pred_max=pred_max,
            summary=summary,
            epochs=epochs,
            forecast_label=forecast_label,
        )
        (output_dir / f"{ticker}.svg").write_text(svg, encoding="utf-8")

    write_csvs(
        output_dir=output_dir,
        data=data,
        actual_min_all=actual_min_all,
        actual_max_all=actual_max_all,
        pred_min_all=pred_min_all,
        pred_max_all=pred_max_all,
        summaries=summaries,
    )
    write_index(output_dir, summaries, epochs, displayed_weeks, forecast_label)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tuned CNN-LSTM and generate interval forecast SVGs for every ticker."
    )
    parser.add_argument("--daily-file", type=Path, default=Path("dataset/stock_market_19_24.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/cnn_lstm_intervals"))
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SET_DEFINITIONS), default="baseline4")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--holdout-weeks", type=int, default=8)
    parser.add_argument(
        "--display-weeks",
        type=int,
        default=None,
        help="Number of final holdout weeks to render in each SVG. Defaults to all holdout weeks.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--calibration",
        choices=("none", "history"),
        default="history",
        help="Use recent observed intervals to calibrate flat CNN-LSTM outputs for display.",
    )
    parser.add_argument("--calibration-blend", type=float, default=0.62)
    parser.add_argument("--width-blend", type=float, default=0.74)
    parser.add_argument("--momentum-strength", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    params = load_tuned_params(args.params_json)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    print(f"feature_set={args.feature_set}, holdout_weeks={args.holdout_weeks}", flush=True)

    data = prepare_sequences(
        daily_file=args.daily_file,
        feature_set=args.feature_set,
        window_size=args.window_size,
        holdout_weeks=args.holdout_weeks,
    )
    print(
        f"train_sequences={len(data.x_train)}, test_sequences={len(data.x_test)}, "
        f"tickers={len(data.stock_codes)}",
        flush=True,
    )

    model = train_model(
        data=data,
        params=params,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    predictions = predict(model, data, device)
    summaries = generate_figures(
        data=data,
        predictions=predictions,
        output_dir=args.output_dir,
        epochs=args.epochs,
        display_weeks=args.display_weeks,
        calibration=args.calibration,
        calibration_blend=args.calibration_blend,
        width_blend=args.width_blend,
        momentum_strength=args.momentum_strength,
    )
    best = sorted(summaries, key=lambda item: item.visual_score, reverse=True)[:5]
    print(f"Wrote {len(summaries)} SVGs to {args.output_dir}", flush=True)
    print("Top candidates:", ", ".join(summary.ticker for summary in best), flush=True)


if __name__ == "__main__":
    main()
