from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.data import inclusive_end_timestamp
from ashare_cross_section_similarity.features import (
    FEATURE_COLUMNS,
    normalized_close_path,
    window_features,
    z_normalize,
)
from ashare_cross_section_similarity.similarity_algorithms import (
    BASELINE_ALGORITHM,
    build_algorithm_target,
    distance_for_close_matrix,
    ensure_algorithm_available,
)
from ashare_cross_section_similarity.similarity import _prepare_bars, _score_results
from ashare_cross_section_similarity.universe import normalize_symbol


@dataclass(frozen=True)
class HistorySearchConfig:
    symbol: str
    as_of: str | pd.Timestamp
    window_size: int
    forward_windows: tuple[int, ...] = (5, 20, 60)
    candidate_n: int = 100
    top_n: int = 10
    exclusion_bars: int = 20
    nearby_gap_days: int = 20
    path_weight: float = 0.7
    window_start: str | pd.Timestamp | None = None
    algorithm: str = BASELINE_ALGORITHM


@dataclass(frozen=True)
class HistorySearchResult:
    symbol: str
    as_of: pd.Timestamp
    window_size: int
    current_window: pd.DataFrame
    historical_windows: list[pd.DataFrame]
    results: pd.DataFrame


def search_history(bars: pd.DataFrame, config: HistorySearchConfig) -> HistorySearchResult:
    if config.window_start is None and config.window_size < 2:
        raise ValueError("window_size 至少需要 2。")
    if config.candidate_n < 1:
        raise ValueError("candidate_n 至少需要 1。")
    if config.top_n < 1:
        raise ValueError("top_n 至少需要 1。")
    if config.exclusion_bars < 0:
        raise ValueError("exclusion_bars 不能为负数。")
    if config.nearby_gap_days < 0:
        raise ValueError("nearby_gap_days 不能为负数。")
    if any(horizon <= 0 for horizon in config.forward_windows):
        raise ValueError("forward_windows 必须为正整数。")
    if not 0 <= config.path_weight <= 1:
        raise ValueError("path_weight 必须在 0 到 1 之间。")
    algorithm = ensure_algorithm_available(config.algorithm, mode="history")
    prepared = _prepare_bars(bars)
    symbol = normalize_symbol(config.symbol)
    prepared = prepared.loc[prepared["stock_code"] == symbol].sort_values("date").reset_index(drop=True)
    if prepared.empty:
        raise ValueError(f"未找到 {symbol} 的本地行情。")

    as_of = inclusive_end_timestamp(config.as_of)
    available = prepared.loc[prepared["date"] <= as_of].copy()
    if config.window_start is None and len(available) < config.window_size:
        raise ValueError("as-of 之前数据不足，无法形成当前窗口。")

    current_window, current_start, as_of_index, window_size = _current_history_window(prepared, available, config)
    target_path = z_normalize(normalized_close_path(current_window))
    target_metric = build_algorithm_target(current_window, algorithm)
    target_features = window_features(current_window)
    max_forward = max(config.forward_windows) if config.forward_windows else 0

    starts = _candidate_starts(
        as_of_index=as_of_index,
        current_start=current_start,
        window_size=window_size,
        max_forward=max_forward,
        exclusion_bars=config.exclusion_bars,
    )
    result_frame = _history_candidate_frame(
        prepared=prepared,
        symbol=symbol,
        starts=starts,
        window_size=window_size,
        target_path=target_path,
        target_metric=target_metric,
        target_features=target_features,
        forward_windows=config.forward_windows,
        algorithm=algorithm,
    )
    if result_frame.empty:
        return HistorySearchResult(
            symbol=symbol,
            as_of=as_of,
            window_size=window_size,
            current_window=current_window,
            historical_windows=[],
            results=result_frame,
        )

    scored = _score_results(result_frame, config.path_weight)
    scored = scored.sort_values(["综合相似度", "路径相似度"], ascending=False).head(config.candidate_n)
    scored = _filter_nearby_history_windows(
        scored,
        top_n=config.top_n,
        min_gap_days=config.nearby_gap_days,
    ).reset_index(drop=True)
    selected_indices = pd.to_numeric(scored["_candidate_index"], errors="coerce").dropna().astype(int).tolist()
    selected_windows = [
        prepared.iloc[starts[index] : starts[index] + window_size].reset_index(drop=True)
        for index in selected_indices
        if 0 <= int(index) < len(starts)
    ]
    scored = scored.drop(columns=["_candidate_index"])
    return HistorySearchResult(
        symbol=symbol,
        as_of=as_of,
        window_size=window_size,
        current_window=current_window,
        historical_windows=selected_windows,
        results=scored.reset_index(drop=True),
    )


def _current_history_window(
    prepared: pd.DataFrame,
    available: pd.DataFrame,
    config: HistorySearchConfig,
) -> tuple[pd.DataFrame, int, int, int]:
    if config.window_start is None:
        as_of_index = int(available.index[-1])
        current_start = as_of_index - config.window_size + 1
        current_window = prepared.iloc[current_start : as_of_index + 1].reset_index(drop=True)
        return current_window, current_start, as_of_index, config.window_size

    start = pd.Timestamp(config.window_start)
    as_of = inclusive_end_timestamp(config.as_of)
    selected = prepared.loc[prepared["date"].between(start, as_of)]
    if len(selected) < 2:
        raise ValueError("选定区间内 K 线数量不足，至少需要 2 根。")
    current_start = int(selected.index[0])
    as_of_index = int(selected.index[-1])
    current_window = selected.reset_index(drop=True)
    return current_window, current_start, as_of_index, int(len(current_window))


def _candidate_starts(
    *,
    as_of_index: int,
    current_start: int,
    window_size: int,
    max_forward: int,
    exclusion_bars: int,
) -> np.ndarray:
    latest_end = current_start - exclusion_bars - 1
    latest_start = latest_end - window_size + 1
    latest_start = min(latest_start, as_of_index - max_forward - window_size + 1)
    if latest_start < 0:
        return np.array([], dtype=int)
    return np.arange(latest_start + 1, dtype=int)


def _history_candidate_frame(
    *,
    prepared: pd.DataFrame,
    symbol: str,
    starts: np.ndarray,
    window_size: int,
    target_path: np.ndarray,
    target_metric,
    target_features: dict[str, float],
    forward_windows: tuple[int, ...],
    algorithm: str,
) -> pd.DataFrame:
    if len(starts) == 0:
        return pd.DataFrame()
    close = pd.to_numeric(prepared["close"], errors="coerce").astype(float).to_numpy()
    amount = pd.to_numeric(prepared["amount"], errors="coerce").astype(float).to_numpy()
    volume = pd.to_numeric(prepared["volume"], errors="coerce").astype(float).to_numpy()
    liquidity = np.where(np.isfinite(amount), amount, volume)
    close_windows = np.lib.stride_tricks.sliding_window_view(close, window_size)[starts]
    path_matrix = _normalized_close_paths(close_windows)
    if algorithm == BASELINE_ALGORITHM:
        path_distance = np.linalg.norm(_z_normalize_rows(path_matrix) - target_path, axis=1) / math.sqrt(window_size)
        price_path_distance = path_distance
        return_path_distance = np.full(len(starts), np.nan, dtype=float)
    else:
        distance_parts = distance_for_close_matrix(close_windows, target_metric)
        path_distance = distance_parts["路径距离"]
        price_path_distance = distance_parts["价格路径距离"]
        return_path_distance = distance_parts["收益路径距离"]
    features = _window_feature_arrays(
        close=close,
        liquidity=liquidity,
        starts=starts,
        window_size=window_size,
        path_matrix=path_matrix,
    )
    ends = starts + window_size - 1
    frame = pd.DataFrame(
        {
            "_candidate_index": np.arange(len(starts)),
            "算法": algorithm,
            "symbol": symbol,
            "窗口开始": prepared["date"].iloc[starts].to_numpy(),
            "窗口结束": prepared["date"].iloc[ends].to_numpy(),
            "K线数量": window_size,
            "路径距离": path_distance,
            "价格路径距离": price_path_distance,
            "收益路径距离": return_path_distance,
        }
    )
    for column in FEATURE_COLUMNS:
        frame[column] = features[column]
        frame[f"feature_diff::{column}"] = np.abs(features[column] - target_features[column])
    for column, values in _forward_outcome_arrays(close, starts, window_size, forward_windows).items():
        frame[column] = values
    return frame


def _normalized_close_paths(close_windows: np.ndarray) -> np.ndarray:
    first = close_windows[:, [0]]
    valid = np.isfinite(first) & (first != 0)
    return np.divide(close_windows, first, out=np.zeros_like(close_windows, dtype=float), where=valid) * 100.0


def _z_normalize_rows(values: np.ndarray) -> np.ndarray:
    means = np.nanmean(values, axis=1, keepdims=True)
    stds = np.nanstd(values, axis=1, keepdims=True)
    centered = values - means
    valid = np.isfinite(stds) & (stds != 0)
    return np.divide(centered, stds, out=centered.copy(), where=valid)


def _window_feature_arrays(
    *,
    close: np.ndarray,
    liquidity: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    path_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    returns = np.divide(
        close[1:],
        close[:-1],
        out=np.full(len(close) - 1, np.nan, dtype=float),
        where=(close[:-1] != 0) & np.isfinite(close[:-1]),
    ) - 1.0
    return_windows = np.lib.stride_tricks.sliding_window_view(returns, window_size - 1)[starts]
    liquidity_windows = np.lib.stride_tricks.sliding_window_view(liquidity, window_size)[starts]
    with np.errstate(invalid="ignore", divide="ignore"):
        interval_return = np.divide(
            path_matrix[:, -1],
            path_matrix[:, 0],
            out=np.zeros(len(starts), dtype=float),
            where=path_matrix[:, 0] != 0,
        ) - 1.0
    interval_return = np.where(np.isfinite(interval_return), interval_return, 0.0)
    volatility = _rowwise_nanstd(return_windows)
    running_max = np.maximum.accumulate(path_matrix, axis=1)
    drawdowns = np.divide(path_matrix, running_max, out=np.full_like(path_matrix, np.nan), where=running_max != 0) - 1.0
    max_drawdowns = _rowwise_nanmin(drawdowns)
    x_centered = np.arange(window_size, dtype=float) - (window_size - 1) / 2
    slope_denominator = float(np.sum(x_centered**2))
    slopes = ((path_matrix - np.nanmean(path_matrix, axis=1, keepdims=True)) @ x_centered) / slope_denominator
    total_liquidity = np.nansum(liquidity_windows, axis=1)
    has_liquidity = np.isfinite(liquidity_windows).any(axis=1)
    down_liquidity = np.nansum(np.where(return_windows < 0, liquidity_windows[:, 1:], 0.0), axis=1)
    down_share = np.divide(
        down_liquidity,
        total_liquidity,
        out=np.zeros(len(starts), dtype=float),
        where=has_liquidity & (total_liquidity != 0),
    )
    corr = _rowwise_corr(return_windows, liquidity_windows[:, 1:])
    liquidity_mean = _rowwise_nanmean(liquidity_windows)
    liquidity_scale = np.where(np.isfinite(liquidity_mean), np.log1p(liquidity_mean), 0.0)
    return {
        "区间收益": interval_return,
        "波动率": volatility,
        "最大回撤": max_drawdowns,
        "趋势斜率": np.where(np.isfinite(slopes), slopes, 0.0),
        "下跌放量占比": np.where(np.isfinite(down_share), down_share, 0.0),
        "量价相关": corr,
        "成交规模": liquidity_scale,
    }


def _forward_outcome_arrays(
    close: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    forward_windows: tuple[int, ...],
) -> dict[str, np.ndarray]:
    outcomes: dict[str, np.ndarray] = {}
    ends = starts + window_size - 1
    base_close = close[ends]
    for horizon in forward_windows:
        returns = np.full(len(starts), np.nan, dtype=float)
        max_drawdowns = np.full(len(starts), np.nan, dtype=float)
        max_favorable = np.full(len(starts), np.nan, dtype=float)
        valid = (ends + horizon < len(close)) & (base_close != 0) & np.isfinite(base_close)
        if valid.any():
            future_indices = ends[valid, None] + np.arange(1, horizon + 1)
            future = close[future_indices]
            base = base_close[valid, None]
            returns[valid] = future[:, -1] / base[:, 0] - 1.0
            future_path = np.concatenate([base, future], axis=1)
            running_max = np.maximum.accumulate(future_path, axis=1)
            drawdowns = np.divide(
                future_path,
                running_max,
                out=np.full_like(future_path, np.nan),
                where=running_max != 0,
            ) - 1.0
            max_drawdowns[valid] = _rowwise_nanmin(drawdowns)
            max_favorable[valid] = _rowwise_nanmax(future / base - 1.0)
        outcomes[f"t_plus_{horizon}_return"] = returns
        outcomes[f"t_plus_{horizon}_max_drawdown"] = max_drawdowns
        outcomes[f"t_plus_{horizon}_max_favorable"] = max_favorable
    return outcomes


def _rowwise_nanmean(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    sums = np.where(valid, values, 0.0).sum(axis=1)
    return np.divide(sums, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0)


def _rowwise_nanstd(values: np.ndarray) -> np.ndarray:
    means = _rowwise_nanmean(values)
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    centered = np.where(valid, values - means[:, None], 0.0)
    variance = np.divide(
        (centered**2).sum(axis=1),
        counts,
        out=np.zeros(values.shape[0], dtype=float),
        where=counts > 0,
    )
    return np.sqrt(variance)


def _rowwise_nanmin(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    masked = np.where(valid, values, np.inf)
    minimum = masked.min(axis=1)
    return np.where(valid.any(axis=1), minimum, 0.0)


def _rowwise_nanmax(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    masked = np.where(valid, values, -np.inf)
    maximum = masked.max(axis=1)
    return np.where(valid.any(axis=1), maximum, 0.0)


def _rowwise_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    valid = np.isfinite(left) & np.isfinite(right)
    counts = valid.sum(axis=1)
    left_mean = np.divide(
        np.where(valid, left, 0.0).sum(axis=1),
        counts,
        out=np.zeros(left.shape[0], dtype=float),
        where=counts > 0,
    )
    right_mean = np.divide(
        np.where(valid, right, 0.0).sum(axis=1),
        counts,
        out=np.zeros(right.shape[0], dtype=float),
        where=counts > 0,
    )
    left_centered = np.where(valid, left - left_mean[:, None], 0.0)
    right_centered = np.where(valid, right - right_mean[:, None], 0.0)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1))
    corr = np.divide(numerator, denominator, out=np.zeros(left.shape[0], dtype=float), where=(counts >= 2) & (denominator != 0))
    return np.where(np.isfinite(corr), corr, 0.0)


def _filter_nearby_history_windows(
    frame: pd.DataFrame,
    *,
    top_n: int,
    min_gap_days: int,
) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for _, row in frame.iterrows():
        start = pd.Timestamp(row["窗口开始"])
        end = pd.Timestamp(row["窗口结束"])
        if any(
            _window_gap_days(
                start,
                end,
                pd.Timestamp(item["窗口开始"]),
                pd.Timestamp(item["窗口结束"]),
            )
            < min_gap_days
            for item in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= top_n:
            break
    return pd.DataFrame(selected) if selected else frame.iloc[:0]


def _window_gap_days(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> int:
    if left_start <= right_end and right_start <= left_end:
        return 0
    if left_end < right_start:
        return int((right_start - left_end).days)
    return int((left_start - right_end).days)
