"""Create a README-friendly actual-vs-forecast return interval figure.

The script intentionally uses only the Python standard library so the figure can
be regenerated in a fresh clone before installing the ML stack.
"""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean


@dataclass(frozen=True)
class WeeklyInterval:
    label: str
    week_end: datetime
    actual_min: float
    actual_max: float
    pred_min: float | None = None
    pred_max: float | None = None


def _parse_price(value: str) -> float | None:
    if not value:
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _load_daily_prices(path: Path, ticker: str) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if ticker not in (reader.fieldnames or []):
            available = ", ".join((reader.fieldnames or [])[1:12])
            raise ValueError(f"Ticker {ticker!r} not found. Available examples: {available}")

        for row in reader:
            raw_date = row.get("History Price / Date", "")
            try:
                date = datetime.strptime(raw_date, "%m/%d/%Y")
            except ValueError:
                continue
            price = _parse_price(row.get(ticker, ""))
            if price is not None and price > 0:
                rows.append((date, price))
    return sorted(rows, key=lambda item: item[0])


def _weekly_intervals(daily_prices: list[tuple[datetime, float]]) -> list[WeeklyInterval]:
    grouped: dict[tuple[int, int], list[tuple[datetime, float]]] = {}
    for date, price in daily_prices:
        iso_year, iso_week, _ = date.isocalendar()
        grouped.setdefault((iso_year, iso_week), []).append((date, price))

    intervals: list[WeeklyInterval] = []
    for (year, week), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item[0])
        if len(rows) < 2:
            continue
        open_price = rows[0][1]
        prices = [price for _, price in rows]
        week_end = rows[-1][0]
        intervals.append(
            WeeklyInterval(
                label=f"{year}-W{week:02d}",
                week_end=week_end,
                actual_min=(min(prices) / open_price) - 1.0,
                actual_max=(max(prices) / open_price) - 1.0,
            )
        )
    return intervals


def _add_rolling_forecast(intervals: list[WeeklyInterval], window: int) -> list[WeeklyInterval]:
    forecasted: list[WeeklyInterval] = []
    for index, interval in enumerate(intervals):
        if index < window:
            forecasted.append(interval)
            continue

        history = intervals[index - window : index]
        pred_min = fmean(item.actual_min for item in history)
        pred_max = fmean(item.actual_max for item in history)
        if pred_min > pred_max:
            pred_min, pred_max = pred_max, pred_min

        forecasted.append(
            WeeklyInterval(
                label=interval.label,
                week_end=interval.week_end,
                actual_min=interval.actual_min,
                actual_max=interval.actual_max,
                pred_min=pred_min,
                pred_max=pred_max,
            )
        )
    return forecasted


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _points_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first_x, first_y = points[0]
    commands = [f"M {first_x:.2f} {first_y:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(commands)


def _polygon_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _render_svg(intervals: list[WeeklyInterval], ticker: str, window: int) -> str:
    width = 1180
    height = 610
    left = 82
    right = 36
    top = 92
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    values: list[float] = []
    for item in intervals:
        values.extend([item.actual_min, item.actual_max])
        if item.pred_min is not None and item.pred_max is not None:
            values.extend([item.pred_min, item.pred_max])

    y_min = min(values)
    y_max = max(values)
    margin = max((y_max - y_min) * 0.14, 0.012)
    y_min -= margin
    y_max += margin

    def x_pos(index: int) -> float:
        if len(intervals) == 1:
            return left + plot_w / 2
        return left + (plot_w * index / (len(intervals) - 1))

    def y_pos(value: float) -> float:
        return top + ((y_max - value) / (y_max - y_min)) * plot_h

    actual_max_points = [(x_pos(i), y_pos(item.actual_max)) for i, item in enumerate(intervals)]
    actual_min_points = [(x_pos(i), y_pos(item.actual_min)) for i, item in enumerate(intervals)]
    pred_items = [(i, item) for i, item in enumerate(intervals) if item.pred_min is not None and item.pred_max is not None]
    pred_max_points = [(x_pos(i), y_pos(item.pred_max or 0.0)) for i, item in pred_items]
    pred_min_points = [(x_pos(i), y_pos(item.pred_min or 0.0)) for i, item in pred_items]

    actual_band = actual_max_points + list(reversed(actual_min_points))
    pred_band = pred_max_points + list(reversed(pred_min_points))

    ticks = []
    tick_count = 6
    for tick_index in range(tick_count):
        value = y_min + (y_max - y_min) * tick_index / (tick_count - 1)
        y = y_pos(value)
        ticks.append((value, y))

    label_step = max(1, len(intervals) // 7)
    x_labels = [
        (index, item.week_end.strftime("%b %d"))
        for index, item in enumerate(intervals)
        if index % label_step == 0 or index == len(intervals) - 1
    ]

    zero_y = y_pos(0.0)
    title = f"Weekly Return Interval Forecast - {ticker}"
    subtitle = f"Actual min-max band vs walk-forward {window}-week forecast band"

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)}</desc>",
        '<rect width="1180" height="610" fill="#fbfbf8"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif}.title{font-size:28px;font-weight:760;fill:#111827}.sub{font-size:15px;fill:#4b5563}.axis{font-size:12px;fill:#6b7280}.legend{font-size:14px;fill:#374151}.note{font-size:12px;fill:#6b7280}</style>',
        f'<text class="title" x="{left}" y="42">{html.escape(title)}</text>',
        f'<text class="sub" x="{left}" y="66">{html.escape(subtitle)}</text>',
    ]

    for value, y in ticks:
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text class="axis" x="{left - 14}" y="{y + 4:.2f}" text-anchor="end">{_format_pct(value)}</text>')

    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af" stroke-width="1"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af" stroke-width="1"/>',
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}" stroke="#374151" stroke-width="1.2" stroke-dasharray="4 6" opacity="0.55"/>',
        ]
    )

    for index, label in x_labels:
        x = x_pos(index)
        svg.append(f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" y2="{height - bottom + 6}" stroke="#9ca3af"/>')
        svg.append(
            f'<text class="axis" x="{x:.2f}" y="{height - bottom + 26}" text-anchor="middle" transform="rotate(-28 {x:.2f} {height - bottom + 26})">{html.escape(label)}</text>'
        )

    svg.extend(
        [
            f'<polygon points="{_polygon_points(actual_band)}" fill="#64748b" opacity="0.18"/>',
            f'<polygon points="{_polygon_points(pred_band)}" fill="#14b8a6" opacity="0.22"/>',
            f'<path d="{_points_path(actual_min_points)}" fill="none" stroke="#1f2937" stroke-width="2.8"/>',
            f'<path d="{_points_path(actual_max_points)}" fill="none" stroke="#475569" stroke-width="2.8"/>',
            f'<path d="{_points_path(pred_min_points)}" fill="none" stroke="#0f766e" stroke-width="3.2" stroke-dasharray="7 6"/>',
            f'<path d="{_points_path(pred_max_points)}" fill="none" stroke="#10b981" stroke-width="3.2"/>',
        ]
    )

    for x, y in actual_min_points[:: max(1, len(actual_min_points) // 18)]:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#1f2937"/>')
    for x, y in actual_max_points[:: max(1, len(actual_max_points) // 18)]:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#475569"/>')

    legend_y = height - 34
    svg.extend(
        [
            f'<rect x="{left}" y="{legend_y - 15}" width="18" height="12" fill="#64748b" opacity="0.22"/>',
            f'<text class="legend" x="{left + 26}" y="{legend_y - 5}">Actual interval</text>',
            f'<rect x="{left + 170}" y="{legend_y - 15}" width="18" height="12" fill="#14b8a6" opacity="0.28"/>',
            f'<text class="legend" x="{left + 196}" y="{legend_y - 5}">Forecast interval</text>',
            f'<line x1="{left + 356}" y1="{legend_y - 10}" x2="{left + 398}" y2="{legend_y - 10}" stroke="#0f766e" stroke-width="3" stroke-dasharray="7 6"/>',
            f'<text class="legend" x="{left + 408}" y="{legend_y - 5}">Forecast min</text>',
            f'<line x1="{left + 540}" y1="{legend_y - 10}" x2="{left + 582}" y2="{legend_y - 10}" stroke="#10b981" stroke-width="3"/>',
            f'<text class="legend" x="{left + 592}" y="{legend_y - 5}">Forecast max</text>',
            f'<text class="note" x="{width - right}" y="{legend_y - 5}" text-anchor="end">Generated from dataset/stock_market_19_24.csv</text>',
        ]
    )

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def build_figure(input_csv: Path, output_path: Path, ticker: str, forecast_window: int, weeks: int) -> Path:
    prices = _load_daily_prices(input_csv, ticker)
    intervals = _add_rolling_forecast(_weekly_intervals(prices), forecast_window)
    display_intervals = [item for item in intervals if item.pred_min is not None and item.pred_max is not None][-weeks:]
    if len(display_intervals) < 4:
        raise ValueError("Not enough weekly intervals to plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_svg(display_intervals, ticker=ticker, window=forecast_window), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an actual-vs-forecast interval SVG for README/docs.")
    parser.add_argument("--input", type=Path, default=Path("dataset/stock_market_19_24.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures/interval_forecast_band.svg"))
    parser.add_argument("--ticker", type=str, default="DXG")
    parser.add_argument("--forecast-window", type=int, default=8)
    parser.add_argument("--weeks", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_figure(
        input_csv=args.input,
        output_path=args.output,
        ticker=args.ticker.upper(),
        forecast_window=args.forecast_window,
        weeks=args.weeks,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
