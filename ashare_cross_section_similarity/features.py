from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "区间收益",
    "波动率",
    "最大回撤",
    "趋势斜率",
    "下跌放量占比",
    "量价相关",
    "成交规模",
)


def normalized_close_path(window: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(window["close"], errors="coerce").astype(float).to_numpy()
    if len(values) == 0:
        return np.array([], dtype=float)
    first = values[0]
    if not np.isfinite(first) or first == 0:
        return np.zeros(len(values), dtype=float)
    return values / first * 100.0


def z_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def resample_path(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == length:
        return values
    if len(values) == 0:
        return np.zeros(length, dtype=float)
    if length <= 1:
        return np.array([float(values[-1])], dtype=float)
    source_x = np.linspace(0.0, 1.0, len(values))
    target_x = np.linspace(0.0, 1.0, length)
    return np.interp(target_x, source_x, values)


def max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = values / np.where(running_max == 0, np.nan, running_max) - 1.0
    if np.isnan(drawdowns).all():
        return 0.0
    return float(np.nanmin(drawdowns))


def window_features(window: pd.DataFrame) -> dict[str, float]:
    path = normalized_close_path(window)
    returns = pd.to_numeric(window["close"], errors="coerce").astype(float).pct_change()
    amount = pd.to_numeric(window.get("amount"), errors="coerce")
    volume = pd.to_numeric(window.get("volume"), errors="coerce")
    liquidity = amount.fillna(volume).astype(float)
    down_mask = returns < 0
    total_liquidity = float(liquidity.sum()) if len(liquidity.dropna()) else 0.0
    down_share = float(liquidity.loc[down_mask].sum() / total_liquidity) if total_liquidity else 0.0
    corr = _safe_corr(returns, liquidity)
    slope = 0.0
    if len(path) >= 2:
        x_values = np.arange(len(path), dtype=float)
        slope = float(np.polyfit(x_values, path, 1)[0])
    return {
        "区间收益": float(path[-1] / path[0] - 1.0) if len(path) >= 2 and path[0] else 0.0,
        "波动率": float(returns.dropna().std(ddof=0)) if len(returns.dropna()) else 0.0,
        "最大回撤": max_drawdown(path),
        "趋势斜率": slope,
        "下跌放量占比": down_share,
        "量价相关": corr,
        "成交规模": float(np.log1p(liquidity.mean())) if len(liquidity.dropna()) else 0.0,
    }


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pairs = pd.concat([left, right], axis=1).dropna()
    if len(pairs) < 2:
        return 0.0
    left_values = pairs.iloc[:, 0].astype(float).to_numpy()
    right_values = pairs.iloc[:, 1].astype(float).to_numpy()
    left_centered = left_values - float(left_values.mean())
    right_centered = right_values - float(right_values.mean())
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    corr = float(np.sum(left_centered * right_centered) / denominator)
    return corr if np.isfinite(corr) else 0.0
