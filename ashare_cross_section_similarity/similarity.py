from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.data import inclusive_end_timestamp
from ashare_cross_section_similarity.features import (
    FEATURE_COLUMNS,
    z_normalize,
)
from ashare_cross_section_similarity.similarity_algorithms import (
    BASELINE_ALGORITHM,
    AlgorithmTarget,
    build_algorithm_target,
    distance_for_close_matrix,
    distance_for_window,
    ensure_algorithm_available,
)
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols

FORWARD_RETURN_WINDOWS = (3, 5, 10)


def _compat_z_normalize(values: np.ndarray) -> np.ndarray:
    return z_normalize(values)


@dataclass(frozen=True)
class CrossSectionSearchConfig:
    target_symbol: str
    universe_symbols: tuple[str, ...]
    start: str | pd.Timestamp
    end: str | pd.Timestamp
    top_n: int = 20
    min_coverage: float = 0.8
    path_weight: float = 0.7
    forward_windows: tuple[int, ...] = FORWARD_RETURN_WINDOWS
    date_tolerance_bars: int = 0
    algorithm: str = BASELINE_ALGORITHM


@dataclass(frozen=True)
class CrossSectionSearchResult:
    target_symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    window_size: int
    results: pd.DataFrame
    skipped: pd.DataFrame


@dataclass(frozen=True)
class CrossSectionTraversalConfig:
    target_symbol: str
    universe_symbols: tuple[str, ...]
    start: str | pd.Timestamp
    end: str | pd.Timestamp
    window_bars: int
    step_bars: int = 1
    top_n_per_window: int = 5
    max_windows: int = 60
    min_coverage: float = 0.8
    path_weight: float = 0.7
    forward_windows: tuple[int, ...] = FORWARD_RETURN_WINDOWS
    date_tolerance_bars: int = 0
    algorithm: str = BASELINE_ALGORITHM


@dataclass(frozen=True)
class CrossSectionTraversalResult:
    target_symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    window_bars: int
    step_bars: int
    windows: pd.DataFrame
    results: pd.DataFrame
    skipped: pd.DataFrame


@dataclass(frozen=True)
class _CandidateWindow:
    frame: pd.DataFrame
    date_offset: int
    path_distance: float | None = None
    price_path_distance: float | None = None
    return_path_distance: float | None = None


def search_cross_section(
    bars: pd.DataFrame,
    config: CrossSectionSearchConfig,
) -> CrossSectionSearchResult:
    if config.top_n < 1:
        raise ValueError("top_n 至少需要 1。")
    if not 0 < config.min_coverage <= 1:
        raise ValueError("min_coverage 必须在 0 到 1 之间。")
    if not 0 <= config.path_weight <= 1:
        raise ValueError("path_weight 必须在 0 到 1 之间。")
    if any(horizon <= 0 for horizon in config.forward_windows):
        raise ValueError("forward_windows 必须为正整数。")
    if config.date_tolerance_bars < 0:
        raise ValueError("date_tolerance_bars 不能为负数。")
    algorithm = ensure_algorithm_available(config.algorithm, mode="cross_section")
    prepared = _prepare_bars(bars)
    target_symbol = normalize_symbol(config.target_symbol)
    start = pd.Timestamp(config.start)
    end = inclusive_end_timestamp(config.end)
    bars_by_symbol = _bars_by_symbol(prepared)
    windows = _windows_by_symbol(prepared, start, end)
    target_window = windows.get(target_symbol)
    if target_window is None or target_window.empty:
        raise ValueError(f"目标标的 {target_symbol} 在所选区间没有行情数据。")

    target_length = len(target_window)
    minimum_rows = max(2, math.ceil(target_length * config.min_coverage))
    target_metric = build_algorithm_target(target_window, algorithm)
    target_features = _fast_window_features(target_window)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for symbol in unique_symbols(config.universe_symbols):
        if symbol == target_symbol:
            continue
        symbol_bars = bars_by_symbol.get(symbol)
        candidate_match = _best_candidate_window(
            symbol_bars if symbol_bars is not None else _empty_bars(),
            windows.get(symbol),
            start,
            target_length,
            minimum_rows,
            config.date_tolerance_bars,
            target_metric,
        )
        candidate = candidate_match.frame if candidate_match is not None else None
        candidate_rows = 0 if candidate is None else len(candidate)
        if candidate_rows < minimum_rows:
            skipped.append(
                {
                    "symbol": symbol,
                    "原因": f"区间数据不足：{candidate_rows} / {target_length}",
                }
            )
            continue
        distance_parts = (
            distance_for_window(candidate, target_metric)
            if candidate_match.path_distance is None
            else {
                "路径距离": candidate_match.path_distance,
                "价格路径距离": candidate_match.price_path_distance,
                "收益路径距离": candidate_match.return_path_distance,
            }
        )
        features = _fast_window_features(candidate)
        row: dict[str, object] = {
            "算法": algorithm,
            "symbol": symbol,
            "区间开始": candidate["date"].min(),
            "区间结束": candidate["date"].max(),
            "K线数量": int(len(candidate)),
            "日期偏移": int(candidate_match.date_offset),
            "覆盖率": float(len(candidate) / target_length),
            "路径距离": distance_parts["路径距离"],
            "价格路径距离": distance_parts["价格路径距离"],
            "收益路径距离": distance_parts["收益路径距离"],
        }
        for column in FEATURE_COLUMNS:
            row[column] = features[column]
            row[f"feature_diff::{column}"] = abs(features[column] - target_features[column])
        row.update(_forward_returns(bars_by_symbol[symbol], candidate["date"].max(), config.forward_windows))
        rows.append(row)

    result_frame = pd.DataFrame(rows)
    if not result_frame.empty:
        result_frame = _score_results(result_frame, config.path_weight)
        result_frame = result_frame.sort_values(
            ["综合相似度", "路径相似度"],
            ascending=False,
        ).head(config.top_n)
        result_frame = result_frame.reset_index(drop=True)
    skipped_frame = pd.DataFrame(skipped, columns=["symbol", "原因"])
    return CrossSectionSearchResult(
        target_symbol=target_symbol,
        start=start,
        end=end,
        window_size=target_length,
        results=result_frame,
        skipped=skipped_frame,
    )


def traverse_cross_section(
    bars: pd.DataFrame,
    config: CrossSectionTraversalConfig,
) -> CrossSectionTraversalResult:
    if config.window_bars < 2:
        raise ValueError("遍历窗口K线数至少需要 2。")
    if config.step_bars < 1:
        raise ValueError("遍历步长至少需要 1。")
    if config.top_n_per_window < 1:
        raise ValueError("每窗保留数量至少需要 1。")
    if config.max_windows < 1:
        raise ValueError("最多遍历窗口至少需要 1。")
    prepared = _prepare_bars(bars)
    target_symbol = normalize_symbol(config.target_symbol)
    start = pd.Timestamp(config.start)
    end = inclusive_end_timestamp(config.end)
    target_bars = _bars_by_symbol(prepared).get(target_symbol, _empty_bars())
    target_range = target_bars.loc[target_bars["date"].between(start, end)].reset_index(drop=True)
    if len(target_range) < config.window_bars:
        raise ValueError(f"目标标的 {target_symbol} 在遍历区间内不足 {config.window_bars} 根K线。")

    window_rows: list[dict[str, object]] = []
    result_frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    window_starts = range(0, len(target_range) - config.window_bars + 1, config.step_bars)
    for window_index, start_position in enumerate(window_starts, start=1):
        if window_index > config.max_windows:
            break
        target_window = target_range.iloc[start_position : start_position + config.window_bars]
        window_start = pd.Timestamp(target_window["date"].iloc[0])
        window_end = pd.Timestamp(target_window["date"].iloc[-1])
        window_rows.append(
            {
                "窗口序号": window_index,
                "目标窗口开始": window_start,
                "目标窗口结束": window_end,
                "K线数量": int(len(target_window)),
            }
        )
        result = search_cross_section(
            prepared,
            CrossSectionSearchConfig(
                target_symbol=target_symbol,
                universe_symbols=config.universe_symbols,
                start=window_start,
                end=window_end,
                top_n=config.top_n_per_window,
                min_coverage=config.min_coverage,
                path_weight=config.path_weight,
                forward_windows=config.forward_windows,
                date_tolerance_bars=config.date_tolerance_bars,
                algorithm=config.algorithm,
            ),
        )
        result_frames.append(_tag_traversal_frame(result.results, window_index, window_start, window_end))
        skipped_frames.append(_tag_traversal_frame(result.skipped, window_index, window_start, window_end))

    return CrossSectionTraversalResult(
        target_symbol=target_symbol,
        start=start,
        end=end,
        window_bars=int(config.window_bars),
        step_bars=int(config.step_bars),
        windows=pd.DataFrame(window_rows, columns=["窗口序号", "目标窗口开始", "目标窗口结束", "K线数量"]),
        results=pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame(),
        skipped=pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _tag_traversal_frame(
    frame: pd.DataFrame,
    window_index: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "目标窗口结束", window_end)
    tagged.insert(0, "目标窗口开始", window_start)
    tagged.insert(0, "窗口序号", int(window_index))
    return tagged


def _best_candidate_window(
    symbol_bars: pd.DataFrame,
    strict_window: pd.DataFrame | None,
    start: pd.Timestamp,
    target_length: int,
    minimum_rows: int,
    date_tolerance_bars: int,
    target_metric: AlgorithmTarget,
) -> _CandidateWindow | None:
    if date_tolerance_bars == 0:
        if strict_window is None or strict_window.empty:
            return None
        return _CandidateWindow(strict_window, 0)
    if symbol_bars.empty:
        return None
    date_values = symbol_bars["date"].to_numpy(dtype="datetime64[ns]", copy=False)
    anchor_position = int(np.searchsorted(date_values, start.to_datetime64(), side="left"))
    if anchor_position >= len(symbol_bars):
        return None
    best_key: tuple[float, int, int] | None = None
    best_start: int | None = None
    best_length: int | None = None
    best_distance: float | None = None
    best_price_distance: float | None = None
    best_return_distance: float | None = None
    close = symbol_bars["close"].to_numpy(dtype=float, copy=False)
    candidate_starts = np.arange(
        anchor_position - date_tolerance_bars,
        anchor_position + date_tolerance_bars + 1,
        dtype=int,
    )
    candidate_starts = candidate_starts[
        (candidate_starts >= 0)
        & (candidate_starts < len(symbol_bars))
        & ((len(symbol_bars) - candidate_starts) >= minimum_rows)
    ]
    full_starts = candidate_starts[candidate_starts + target_length <= len(symbol_bars)]
    if len(full_starts):
        close_windows = np.lib.stride_tricks.sliding_window_view(close, target_length)[full_starts]
        distance_parts = distance_for_close_matrix(close_windows, target_metric)
        distances = distance_parts["路径距离"]
        offsets = full_starts - anchor_position
        order = np.lexsort((offsets, np.abs(offsets), distances))
        best_index = int(order[0])
        best_start = int(full_starts[best_index])
        best_length = target_length
        best_distance = float(distances[best_index])
        best_price_distance = float(distance_parts["价格路径距离"][best_index])
        best_return_distance = float(distance_parts["收益路径距离"][best_index])
        best_offset = int(offsets[best_index])
        best_key = (best_distance, abs(best_offset), best_offset)

    partial_starts = candidate_starts[candidate_starts + target_length > len(symbol_bars)]
    for start_position in partial_starts:
        window_length = len(symbol_bars) - int(start_position)
        if window_length < minimum_rows:
            continue
        distance_parts = distance_for_window(
            symbol_bars.iloc[int(start_position) :].reset_index(drop=True),
            target_metric,
        )
        path_distance = float(distance_parts["路径距离"])
        date_offset = int(start_position) - anchor_position
        key = (path_distance, abs(date_offset), date_offset)
        if best_key is None or key < best_key:
            best_start = int(start_position)
            best_length = int(window_length)
            best_distance = path_distance
            best_price_distance = float(distance_parts["价格路径距离"])
            best_return_distance = float(distance_parts["收益路径距离"])
            best_key = key

    if best_start is None or best_length is None:
        return None
    date_offset = best_start - anchor_position
    return _CandidateWindow(
        symbol_bars.iloc[best_start : best_start + best_length].reset_index(drop=True),
        date_offset,
        best_distance,
        best_price_distance,
        best_return_distance,
    )


def _fast_window_features(window: pd.DataFrame) -> dict[str, float]:
    close = window["close"].to_numpy(dtype=float, copy=False)
    amount = window["amount"].to_numpy(dtype=float, copy=False)
    volume = window["volume"].to_numpy(dtype=float, copy=False)
    liquidity = np.where(np.isfinite(amount), amount, volume)
    return _window_features_from_arrays(close, liquidity)


def _window_features_from_arrays(close: np.ndarray, liquidity: np.ndarray) -> dict[str, float]:
    close = np.asarray(close, dtype=float)
    liquidity = np.asarray(liquidity, dtype=float)
    path = _normalized_close_path(close)
    returns = np.divide(
        close[1:],
        close[:-1],
        out=np.full(max(0, len(close) - 1), np.nan, dtype=float),
        where=(close[:-1] != 0) & np.isfinite(close[:-1]),
    ) - 1.0
    total_liquidity = float(np.nansum(liquidity)) if np.isfinite(liquidity).any() else 0.0
    down_liquidity = float(np.nansum(np.where(returns < 0, liquidity[1:], 0.0))) if len(returns) else 0.0
    down_share = down_liquidity / total_liquidity if total_liquidity else 0.0
    slope = 0.0
    if len(path) >= 2:
        x_centered = np.arange(len(path), dtype=float) - (len(path) - 1) / 2
        denominator = float(np.sum(x_centered**2))
        if denominator:
            slope = float(((path - np.nanmean(path)) @ x_centered) / denominator)
    liquidity_mean = float(np.nanmean(liquidity)) if np.isfinite(liquidity).any() else 0.0
    return {
        "区间收益": float(path[-1] / path[0] - 1.0) if len(path) >= 2 and path[0] else 0.0,
        "波动率": float(np.nanstd(returns)) if np.isfinite(returns).any() else 0.0,
        "最大回撤": _max_drawdown(path),
        "趋势斜率": slope if np.isfinite(slope) else 0.0,
        "下跌放量占比": down_share if np.isfinite(down_share) else 0.0,
        "量价相关": _safe_corr_arrays(returns, liquidity[1:]),
        "成交规模": float(np.log1p(liquidity_mean)) if np.isfinite(liquidity_mean) else 0.0,
    }


def _normalized_close_path(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    if len(close) == 0:
        return np.array([], dtype=float)
    first = close[0]
    if not np.isfinite(first) or first == 0:
        return np.zeros(len(close), dtype=float)
    return close / first * 100.0


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


def _max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = np.divide(values, running_max, out=np.full_like(values, np.nan), where=running_max != 0) - 1.0
    if np.isnan(drawdowns).all():
        return 0.0
    return float(np.nanmin(drawdowns))


def _safe_corr_arrays(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return 0.0
    left_values = left[valid]
    right_values = right[valid]
    left_centered = left_values - float(left_values.mean())
    right_centered = right_values - float(right_values.mean())
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    corr = float(np.sum(left_centered * right_centered) / denominator)
    return corr if np.isfinite(corr) else 0.0


def _score_results(frame: pd.DataFrame, path_weight: float) -> pd.DataFrame:
    scored = frame.copy()
    diff_columns = [f"feature_diff::{column}" for column in FEATURE_COLUMNS]
    scaled_diffs = []
    for column in diff_columns:
        values = pd.to_numeric(scored[column], errors="coerce").fillna(0.0)
        scale = float(values.median())
        if not np.isfinite(scale) or scale == 0:
            scale = float(values.mean())
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        scaled_diffs.append(values / scale)
    feature_distance = pd.concat(scaled_diffs, axis=1).mean(axis=1)
    scored["特征距离"] = feature_distance.astype(float)
    scored["路径相似度"] = np.exp(-pd.to_numeric(scored["路径距离"], errors="coerce").fillna(0.0))
    scored["特征相似度"] = np.exp(-scored["特征距离"])
    scored["综合相似度"] = (
        scored["路径相似度"].astype(float) * path_weight
        + scored["特征相似度"].astype(float) * (1.0 - path_weight)
    )
    return scored.drop(columns=diff_columns)


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return _empty_bars()
    frame = bars.copy()
    if "stock_code" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "stock_code"})
    frame["stock_code"] = frame["stock_code"].map(normalize_symbol)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "stock_code", "open", "high", "low", "close"])
    return frame.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _windows_by_symbol(
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    if bars.empty:
        return {}
    window = bars.loc[bars["date"].between(start, end)]
    if window.empty:
        return {}
    return {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in window.groupby("stock_code", sort=False)
    }


def _bars_by_symbol(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if bars.empty:
        return {}
    return {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in bars.groupby("stock_code", sort=False)
    }


def _forward_returns(
    bars: pd.DataFrame,
    window_end: object,
    forward_windows: tuple[int, ...],
) -> dict[str, float]:
    outcomes: dict[str, float] = {}
    if bars.empty:
        return {f"t_plus_{horizon}_return": float("nan") for horizon in forward_windows}
    end_ts = pd.Timestamp(window_end)
    end_positions = bars.index[bars["date"] <= end_ts]
    if len(end_positions) == 0:
        return {f"t_plus_{horizon}_return": float("nan") for horizon in forward_windows}
    end_position = int(end_positions[-1])
    base_close = float(bars.iloc[end_position]["close"])
    for horizon in forward_windows:
        target_position = end_position + horizon
        if target_position >= len(bars) or base_close == 0 or not np.isfinite(base_close):
            outcomes[f"t_plus_{horizon}_return"] = float("nan")
            continue
        target_close = float(bars.iloc[target_position]["close"])
        outcomes[f"t_plus_{horizon}_return"] = target_close / base_close - 1.0
    return outcomes


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]
    )
