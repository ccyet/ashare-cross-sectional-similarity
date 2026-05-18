from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.data import inclusive_end_timestamp
from ashare_cross_section_similarity.features import (
    FEATURE_COLUMNS,
    normalized_close_path,
    resample_path,
    window_features,
    z_normalize,
)
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols

FORWARD_RETURN_WINDOWS = (3, 5, 10)


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


@dataclass(frozen=True)
class CrossSectionSearchResult:
    target_symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    window_size: int
    results: pd.DataFrame
    skipped: pd.DataFrame


@dataclass(frozen=True)
class _CandidateWindow:
    frame: pd.DataFrame
    date_offset: int


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
    target_path = z_normalize(normalized_close_path(target_window))
    target_features = window_features(target_window)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for symbol in unique_symbols(config.universe_symbols):
        if symbol == target_symbol:
            continue
        candidate_match = _best_candidate_window(
            bars_by_symbol.get(symbol, _empty_bars()),
            windows.get(symbol),
            start,
            target_length,
            minimum_rows,
            config.date_tolerance_bars,
            target_path,
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
        candidate_path = z_normalize(resample_path(normalized_close_path(candidate), target_length))
        path_distance = float(np.linalg.norm(target_path - candidate_path) / math.sqrt(target_length))
        features = window_features(candidate)
        row: dict[str, object] = {
            "symbol": symbol,
            "区间开始": candidate["date"].min(),
            "区间结束": candidate["date"].max(),
            "K线数量": int(len(candidate)),
            "日期偏移": int(candidate_match.date_offset),
            "覆盖率": float(len(candidate) / target_length),
            "路径距离": path_distance,
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


def _best_candidate_window(
    symbol_bars: pd.DataFrame,
    strict_window: pd.DataFrame | None,
    start: pd.Timestamp,
    target_length: int,
    minimum_rows: int,
    date_tolerance_bars: int,
    target_path: np.ndarray,
) -> _CandidateWindow | None:
    if date_tolerance_bars == 0:
        if strict_window is None or strict_window.empty:
            return None
        return _CandidateWindow(strict_window, 0)
    if symbol_bars.empty:
        return None
    anchor_positions = symbol_bars.index[symbol_bars["date"] >= start]
    if len(anchor_positions) == 0:
        return None
    anchor_position = int(anchor_positions[0])
    best: _CandidateWindow | None = None
    best_key: tuple[float, int, int] | None = None
    for date_offset in range(-date_tolerance_bars, date_tolerance_bars + 1):
        start_position = anchor_position + date_offset
        if start_position < 0 or start_position >= len(symbol_bars):
            continue
        candidate = symbol_bars.iloc[start_position : start_position + target_length].reset_index(drop=True)
        if len(candidate) < minimum_rows:
            continue
        path = z_normalize(resample_path(normalized_close_path(candidate), target_length))
        path_distance = float(np.linalg.norm(target_path - path) / math.sqrt(target_length))
        key = (path_distance, abs(date_offset), date_offset)
        if best_key is None or key < best_key:
            best = _CandidateWindow(candidate, date_offset)
            best_key = key
    return best


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
