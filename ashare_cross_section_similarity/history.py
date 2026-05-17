from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.features import (
    FEATURE_COLUMNS,
    max_drawdown,
    normalized_close_path,
    window_features,
    z_normalize,
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


@dataclass(frozen=True)
class HistorySearchResult:
    symbol: str
    as_of: pd.Timestamp
    window_size: int
    current_window: pd.DataFrame
    historical_windows: list[pd.DataFrame]
    results: pd.DataFrame


def search_history(bars: pd.DataFrame, config: HistorySearchConfig) -> HistorySearchResult:
    if config.window_size < 2:
        raise ValueError("window_size 至少需要 2。")
    if not 0 <= config.path_weight <= 1:
        raise ValueError("path_weight 必须在 0 到 1 之间。")
    prepared = _prepare_bars(bars)
    symbol = normalize_symbol(config.symbol)
    prepared = prepared.loc[prepared["stock_code"] == symbol].sort_values("date").reset_index(drop=True)
    if prepared.empty:
        raise ValueError(f"未找到 {symbol} 的本地行情。")

    as_of = pd.Timestamp(config.as_of)
    available = prepared.loc[prepared["date"] <= as_of].copy()
    if len(available) < config.window_size:
        raise ValueError("as-of 之前数据不足，无法形成当前窗口。")

    as_of_index = int(available.index[-1])
    current_start = as_of_index - config.window_size + 1
    current_window = prepared.iloc[current_start : as_of_index + 1].reset_index(drop=True)
    target_path = z_normalize(normalized_close_path(current_window))
    target_features = window_features(current_window)
    max_forward = max(config.forward_windows) if config.forward_windows else 0
    rows: list[dict[str, object]] = []
    windows: list[pd.DataFrame] = []

    latest_end = current_start - max(config.exclusion_bars, 0) - 1
    latest_start = latest_end - config.window_size + 1
    for start in range(0, max(0, latest_start + 1)):
        end = start + config.window_size - 1
        if end + max_forward > as_of_index:
            continue
        candidate = prepared.iloc[start : end + 1].reset_index(drop=True)
        candidate_path = z_normalize(normalized_close_path(candidate))
        path_distance = float(np.linalg.norm(target_path - candidate_path) / math.sqrt(config.window_size))
        features = window_features(candidate)
        row: dict[str, object] = {
            "_candidate_index": len(windows),
            "symbol": symbol,
            "窗口开始": candidate["date"].min(),
            "窗口结束": candidate["date"].max(),
            "K线数量": int(len(candidate)),
            "路径距离": path_distance,
        }
        for column in FEATURE_COLUMNS:
            row[column] = features[column]
            row[f"feature_diff::{column}"] = abs(features[column] - target_features[column])
        row.update(_forward_outcomes(prepared, end, config.forward_windows))
        rows.append(row)
        windows.append(candidate)

    result_frame = pd.DataFrame(rows)
    if result_frame.empty:
        return HistorySearchResult(
            symbol=symbol,
            as_of=as_of,
            window_size=config.window_size,
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
        windows[index]
        for index in selected_indices
        if 0 <= int(index) < len(windows)
    ]
    scored = scored.drop(columns=["_candidate_index"])
    return HistorySearchResult(
        symbol=symbol,
        as_of=as_of,
        window_size=config.window_size,
        current_window=current_window,
        historical_windows=selected_windows,
        results=scored.reset_index(drop=True),
    )


def _forward_outcomes(
    bars: pd.DataFrame,
    window_end_index: int,
    forward_windows: tuple[int, ...],
) -> dict[str, float]:
    outcomes: dict[str, float] = {}
    base_close = float(bars.iloc[window_end_index]["close"])
    for horizon in forward_windows:
        target_index = window_end_index + horizon
        if target_index >= len(bars) or base_close == 0:
            outcomes[f"t_plus_{horizon}_return"] = float("nan")
            outcomes[f"t_plus_{horizon}_max_drawdown"] = float("nan")
            outcomes[f"t_plus_{horizon}_max_favorable"] = float("nan")
            continue
        future = bars.iloc[window_end_index + 1 : target_index + 1]
        closes = pd.to_numeric(future["close"], errors="coerce").astype(float).to_numpy()
        outcomes[f"t_plus_{horizon}_return"] = float(closes[-1] / base_close - 1.0)
        outcomes[f"t_plus_{horizon}_max_drawdown"] = max_drawdown(np.r_[base_close, closes])
        outcomes[f"t_plus_{horizon}_max_favorable"] = float(np.nanmax(closes / base_close - 1.0))
    return outcomes


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
