"""Named feature sets used by the stock prediction pipeline."""

from __future__ import annotations

BASELINE_FEATURES: tuple[str, ...] = ("f_std", "f_mean", "f_return", "f_skew")

COMPACT_TECHNICAL_FEATURES: tuple[str, ...] = (
    "f_skew",
    "weekly_range_lag1",
    "rel_pos_4w",
    "bollinger_width_4",
)

TECHNICAL_FEATURES_NO_ENTROPY: tuple[str, ...] = (
    "alpha_excess",
    "weekly_range_lag1",
    "rel_pos_4w",
    "sma_ratio_4",
    "ema_ratio_4",
    "bollinger_width_4",
    "bollinger_percent_b_4",
    "rsi_14",
    "macd_norm",
    "hurst_20",
)

FEATURE_SET_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "baseline4": BASELINE_FEATURES,
    "compact_technical": COMPACT_TECHNICAL_FEATURES,
    "baseline_plus_compact": BASELINE_FEATURES
    + ("weekly_range_lag1", "rel_pos_4w", "bollinger_width_4"),
    "technical_no_entropy": BASELINE_FEATURES + TECHNICAL_FEATURES_NO_ENTROPY,
    "baseline_plus_hurst26": BASELINE_FEATURES + ("hurst_w26",),
    "baseline_plus_sma12": BASELINE_FEATURES + ("sma_ratio_w12",),
    "screened_liquid_top5": BASELINE_FEATURES
    + (
        "sma_ratio_w12",
        "rsi_w20",
        "bollinger_width_w12_k2p0",
        "macd_norm_f12_s26",
        "rel_pos_w12",
    ),
    "screened_with_hurst": BASELINE_FEATURES
    + (
        "hurst_w26",
        "sma_ratio_w12",
        "rsi_w20",
        "bollinger_width_w12_k2p0",
        "macd_norm_f12_s26",
        "rel_pos_w12",
    ),
}


def supported_feature_sets() -> tuple[str, ...]:
    """Return the supported feature-set names in a stable order."""

    return tuple(FEATURE_SET_DEFINITIONS)
