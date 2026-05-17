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


@dataclass(frozen=True)
class CrossSectionSearchConfig:
    target_symbol: str
    universe_symbols: tuple[str, ...]
    start: str | pd.Timestamp
    end: str | pd.Timestamp
    top_n: int = 20
    min_coverage: float = 0.8
    path_weight: float = 0.7


@dataclass(frozen=True)
class CrossSectionSearchResult:
    target_symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    window_size: int
    results: pd.DataFrame
    skipped: pd.DataFrame


def search_cross_section(
    bars: pd.DataFrame,
    config: CrossSectionSearchConfig,
) -> CrossSectionSearchResult:
    if not 0 < config.min_coverage <= 1:
        raise ValueError("min_coverage 必须在 0 到 1 之间。")
    if not 0 <= config.path_weight <= 1:
        raise ValueError("path_weight 必须在 0 到 1 之间。")
    prepared = _prepare_bars(bars)
    target_symbol = normalize_symbol(config.target_symbol)
    start = pd.Timestamp(config.start)
    end = inclusive_end_timestamp(config.end)
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
        candidate = windows.get(symbol)
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
            "路径距离": path_distance,
        }
        for column in FEATURE_COLUMNS:
            row[column] = features[column]
            row[f"feature_diff::{column}"] = abs(features[column] - target_features[column])
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


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]
    )
