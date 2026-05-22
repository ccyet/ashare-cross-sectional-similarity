from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.features import max_drawdown
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols


@dataclass(frozen=True)
class ReviewConfig:
    symbol: str
    start: str
    end: str
    min_swing_return: float = 0.05
    min_segment_bars: int = 3
    max_segments: int = 6


@dataclass(frozen=True)
class ReviewResult:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    window: pd.DataFrame
    overview: dict[str, float]
    segments: pd.DataFrame
    main_segments: pd.DataFrame
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EqualWeightResult:
    label: str
    frame: pd.DataFrame
    coverage: float
    warning: str = ""


def analyze_price_review(bars: pd.DataFrame, config: ReviewConfig) -> ReviewResult:
    symbol = normalize_symbol(config.symbol)
    window = _prepare_review_window(bars, symbol, config.start, config.end)
    if window.empty:
        return ReviewResult(
            symbol=symbol,
            start=pd.Timestamp(config.start),
            end=pd.Timestamp(config.end),
            window=window,
            overview=_empty_overview(),
            segments=pd.DataFrame(),
            main_segments=pd.DataFrame(),
            warnings=(f"{symbol} 在所选区间没有本地行情。",),
        )
    if len(window) < 2:
        return ReviewResult(
            symbol=symbol,
            start=pd.Timestamp(window["date"].min()),
            end=pd.Timestamp(window["date"].max()),
            window=window,
            overview=_window_overview(window),
            segments=pd.DataFrame(),
            main_segments=pd.DataFrame(),
            warnings=(f"{symbol} 所选区间 K 线不足 2 根，无法识别波段。",),
        )

    threshold = max(0.0, float(config.min_swing_return))
    min_bars = max(1, int(config.min_segment_bars))
    segments = _swing_segments(window, threshold, min_bars)
    main_segments = _select_main_segments(segments, max_segments=max(1, int(config.max_segments)))
    warnings: list[str] = []
    if segments.empty:
        warnings.append("所选区间没有达到最小波段幅度的主要上涨或回撤段。")
    return ReviewResult(
        symbol=symbol,
        start=pd.Timestamp(window["date"].min()),
        end=pd.Timestamp(window["date"].max()),
        window=window,
        overview=_window_overview(window),
        segments=segments,
        main_segments=main_segments,
        warnings=tuple(warnings),
    )


def build_comparison_stats(target_window: pd.DataFrame, comparison: pd.DataFrame, label: str) -> dict[str, object]:
    aligned = _aligned_close_frame(target_window, comparison)
    if aligned.empty or len(aligned) < 2:
        return {
            "标的": label,
            "样本数": int(len(aligned)),
            "目标收益": float("nan"),
            "对比收益": float("nan"),
            "超额收益": float("nan"),
            "相关性": float("nan"),
            "同步关系": "数据不足",
            "强弱结论": "数据不足，无法比较。",
        }

    target_return = _period_return(aligned["target"])
    comparison_return = _period_return(aligned["comparison"])
    excess = target_return - comparison_return
    target_returns = aligned["target"].pct_change()
    comparison_returns = aligned["comparison"].pct_change()
    corr = _safe_corr(target_returns, comparison_returns)
    return {
        "标的": label,
        "样本数": int(len(aligned)),
        "目标收益": target_return,
        "对比收益": comparison_return,
        "超额收益": excess,
        "相关性": corr,
        "同步关系": _sync_label(target_return, comparison_return),
        "强弱结论": _strength_label(excess),
    }


def build_equal_weight_series(
    bars: pd.DataFrame,
    symbols: list[str] | tuple[str, ...],
    *,
    label: str,
    min_coverage: float = 0.5,
) -> EqualWeightResult:
    normalized = unique_symbols(symbols)
    columns = ["date", "stock_code", "close"]
    if not normalized:
        return EqualWeightResult(label=label, frame=pd.DataFrame(columns=columns), coverage=0.0, warning="成分列表为空。")
    if bars.empty:
        return EqualWeightResult(
            label=label,
            frame=pd.DataFrame(columns=columns),
            coverage=0.0,
            warning=f"{label} 本地成分行情为空。",
        )
    frame = bars.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_symbol)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.loc[frame["stock_code"].isin(normalized)].dropna(subset=["date", "close"])
    if frame.empty:
        return EqualWeightResult(
            label=label,
            frame=pd.DataFrame(columns=columns),
            coverage=0.0,
            warning=f"{label} 没有匹配到本地成分行情。",
        )

    pivot = frame.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last").sort_index()
    coverage_by_date = pivot.notna().sum(axis=1) / max(len(normalized), 1)
    coverage = float(coverage_by_date.mean()) if not coverage_by_date.empty else 0.0
    min_coverage = min(max(float(min_coverage), 0.0), 1.0)
    valid = pivot.loc[coverage_by_date >= min_coverage]
    if valid.empty or coverage < min_coverage:
        return EqualWeightResult(
            label=label,
            frame=pd.DataFrame(columns=columns),
            coverage=coverage,
            warning=f"{label} 本地成分平均覆盖率 {_percent(coverage)}，低于阈值 {_percent(min_coverage)}。",
        )

    normalized_paths = valid.apply(_normalize_price_series, axis=0)
    equal_weight = normalized_paths.mean(axis=1, skipna=True).dropna()
    result = pd.DataFrame(
        {
            "date": equal_weight.index,
            "stock_code": label,
            "close": equal_weight.to_numpy(dtype=float),
        }
    ).reset_index(drop=True)
    warning = ""
    if coverage < 0.95:
        warning = f"{label} 本地成分平均覆盖率 {_percent(coverage)}，复盘口径为可用成分等权。"
    return EqualWeightResult(label=label, frame=result, coverage=coverage, warning=warning)


def render_review_text(result: ReviewResult, comparisons: pd.DataFrame | None = None) -> str:
    if result.window.empty:
        return "\n".join(f"- {warning}" for warning in result.warnings) or "- 没有可复盘的数据。"

    overview = result.overview
    lines = [
        (
            f"**总体复盘**：{result.symbol} 在 {_date_text(result.start)} 至 {_date_text(result.end)} "
            f"共 {int(overview['k_bars'])} 根K线，区间收益 {_percent(overview['return'])}，"
            f"最大回撤 {_percent(overview['max_drawdown'])}，最大浮盈 {_percent(overview['max_favorable'])}，"
            f"上涨K线占比 {_percent(overview['up_day_share'])}。"
        )
    ]
    if result.main_segments.empty:
        lines.append("**波段结构**：所选区间没有达到阈值的主要上涨或回撤段，走势更接近震荡或小幅单边变化。")
    else:
        lines.append("**关键波段**：")
        for _, segment in result.main_segments.sort_values("开始日期").iterrows():
            lines.append(
                "- "
                f"{_date_text(segment['开始日期'])} 至 {_date_text(segment['结束日期'])} "
                f"为{segment['方向']}段，共 {int(segment['K线数'])} 根K线，"
                f"区间收益 {_percent(segment['区间收益'])}，段内最大回撤 {_percent(segment['最大回撤'])}，"
                f"振幅 {_percent(segment['振幅'])}。"
            )

    if comparisons is not None and not comparisons.empty:
        valid = comparisons.dropna(subset=["超额收益"])
        if not valid.empty:
            strongest = valid.iloc[valid["超额收益"].astype(float).abs().argmax()]
            lines.append(
                "**相对表现**："
                f"相对 {strongest['标的']} 的超额收益为 {_percent(strongest['超额收益'])}，"
                f"相关性 {float(strongest['相关性']):.2f}，结论为{strongest['强弱结论']}。"
            )
    for warning in result.warnings:
        lines.append(f"**提示**：{warning}")
    return "\n\n".join(lines)


def _prepare_review_window(bars: pd.DataFrame, symbol: str, start: str, end: str) -> pd.DataFrame:
    columns = ["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]
    if bars.empty:
        return pd.DataFrame(columns=columns)
    frame = bars.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_symbol)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    frame = frame.loc[
        (frame["stock_code"] == symbol)
        & frame["date"].between(start_ts, end_ts)
        & frame["close"].notna()
    ]
    return frame[columns].sort_values("date").reset_index(drop=True)


def _window_overview(window: pd.DataFrame) -> dict[str, float]:
    closes = pd.to_numeric(window["close"], errors="coerce").astype(float)
    returns = closes.pct_change().dropna()
    high = pd.to_numeric(window.get("high", closes), errors="coerce").fillna(closes)
    low = pd.to_numeric(window.get("low", closes), errors="coerce").fillna(closes)
    return {
        "k_bars": float(len(window)),
        "return": _period_return(closes),
        "max_drawdown": max_drawdown(closes.to_numpy(dtype=float)),
        "max_favorable": _max_favorable_from_start(closes),
        "volatility": float(returns.std(ddof=0)) if not returns.empty else 0.0,
        "up_day_share": float((returns > 0).mean()) if not returns.empty else 0.0,
        "amplitude": _amplitude(high, low),
    }


def _empty_overview() -> dict[str, float]:
    return {
        "k_bars": 0.0,
        "return": float("nan"),
        "max_drawdown": float("nan"),
        "max_favorable": float("nan"),
        "volatility": float("nan"),
        "up_day_share": float("nan"),
        "amplitude": float("nan"),
    }


def _swing_segments(window: pd.DataFrame, threshold: float, min_bars: int) -> pd.DataFrame:
    closes = pd.to_numeric(window["close"], errors="coerce").astype(float).to_numpy()
    if len(closes) < 2:
        return pd.DataFrame()
    pivots = _zigzag_pivots(closes, threshold)
    rows: list[dict[str, object]] = []
    previous_return = 0.0
    total_abs_return = 0.0
    candidates: list[tuple[int, int, float]] = []
    for start_idx, end_idx in zip(pivots, pivots[1:]):
        if end_idx <= start_idx:
            continue
        segment_return = closes[end_idx] / closes[start_idx] - 1.0 if closes[start_idx] else 0.0
        if end_idx - start_idx + 1 < min_bars or abs(segment_return) < threshold:
            continue
        candidates.append((start_idx, end_idx, segment_return))
        total_abs_return += abs(segment_return)
    for start_idx, end_idx, segment_return in candidates:
        segment = window.iloc[start_idx : end_idx + 1].copy()
        direction = _segment_direction(segment_return, previous_return)
        previous_return = segment_return
        rows.append(_segment_row(segment, direction, segment_return, total_abs_return))
    return pd.DataFrame(rows)


def _zigzag_pivots(closes: np.ndarray, threshold: float) -> list[int]:
    if len(closes) < 2:
        return [0]
    threshold = max(0.0, float(threshold))
    pivots: list[int] = []
    trend = 0
    low_idx = high_idx = 0
    low_price = high_price = float(closes[0])
    extreme_idx = 0
    extreme_price = float(closes[0])

    for index in range(1, len(closes)):
        price = float(closes[index])
        if not math.isfinite(price) or price <= 0:
            continue
        if trend == 0:
            if price < low_price:
                low_price = price
                low_idx = index
            if price > high_price:
                high_price = price
                high_idx = index
            if low_price > 0 and price / low_price - 1.0 >= threshold:
                pivots = [low_idx]
                trend = 1
                extreme_idx = index
                extreme_price = price
            elif high_price > 0 and price / high_price - 1.0 <= -threshold:
                pivots = [high_idx]
                trend = -1
                extreme_idx = index
                extreme_price = price
            continue

        if trend > 0:
            if price >= extreme_price:
                extreme_price = price
                extreme_idx = index
            elif price / extreme_price - 1.0 <= -threshold:
                pivots.append(extreme_idx)
                trend = -1
                extreme_idx = index
                extreme_price = price
        else:
            if price <= extreme_price:
                extreme_price = price
                extreme_idx = index
            elif price / extreme_price - 1.0 >= threshold:
                pivots.append(extreme_idx)
                trend = 1
                extreme_idx = index
                extreme_price = price

    if trend == 0:
        return [0, len(closes) - 1]
    if not pivots:
        pivots = [0]
    if extreme_idx != pivots[-1]:
        pivots.append(extreme_idx)
    if len(closes) - 1 != pivots[-1]:
        final_return = closes[-1] / closes[pivots[-1]] - 1.0 if closes[pivots[-1]] else 0.0
        if abs(final_return) >= threshold:
            pivots.append(len(closes) - 1)
    return sorted(dict.fromkeys(pivots))


def _segment_row(segment: pd.DataFrame, direction: str, segment_return: float, total_abs_return: float) -> dict[str, object]:
    closes = pd.to_numeric(segment["close"], errors="coerce").astype(float)
    high = pd.to_numeric(segment.get("high", closes), errors="coerce").fillna(closes)
    low = pd.to_numeric(segment.get("low", closes), errors="coerce").fillna(closes)
    amount = pd.to_numeric(segment.get("amount"), errors="coerce")
    volume = pd.to_numeric(segment.get("volume"), errors="coerce")
    return {
        "方向": direction,
        "开始日期": pd.Timestamp(segment["date"].iloc[0]),
        "结束日期": pd.Timestamp(segment["date"].iloc[-1]),
        "K线数": int(len(segment)),
        "起点收盘": float(closes.iloc[0]),
        "终点收盘": float(closes.iloc[-1]),
        "区间收益": float(segment_return),
        "最大回撤": max_drawdown(closes.to_numpy(dtype=float)),
        "最大浮盈": _max_favorable_from_start(closes),
        "振幅": _amplitude(high, low),
        "成交额变化": _tail_head_change(amount),
        "成交量变化": _tail_head_change(volume),
        "相对贡献": abs(float(segment_return)) / total_abs_return if total_abs_return else 0.0,
    }


def _select_main_segments(segments: pd.DataFrame, max_segments: int) -> pd.DataFrame:
    if segments.empty:
        return segments
    ranked = segments.copy()
    ranked["_importance"] = (
        pd.to_numeric(ranked["区间收益"], errors="coerce").abs().fillna(0.0)
        + pd.to_numeric(ranked["最大回撤"], errors="coerce").abs().fillna(0.0)
        + pd.to_numeric(ranked["相对贡献"], errors="coerce").fillna(0.0) * 0.2
        + pd.Series(np.linspace(0.0, 0.03, len(ranked)), index=ranked.index)
    )
    selected_indexes = set(ranked.nlargest(max_segments, "_importance").index.tolist())
    selected_indexes.update(ranked.tail(min(2, len(ranked))).index.tolist())
    selected = ranked.loc[sorted(selected_indexes)]
    if len(selected) > max_segments:
        selected = selected.nlargest(max_segments, "_importance")
    result = selected.sort_values("开始日期")
    return result.drop(columns=["_importance"]).reset_index(drop=True)


def _aligned_close_frame(target: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    if target.empty or comparison.empty:
        return pd.DataFrame(columns=["date", "target", "comparison"])
    left = _close_by_date(target, "target")
    right = _close_by_date(comparison, "comparison")
    return left.merge(right, on="date", how="inner").dropna().sort_values("date").reset_index(drop=True)


def _close_by_date(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result[column] = pd.to_numeric(result["close"], errors="coerce")
    return result.dropna(subset=["date", column]).groupby("date", as_index=False)[column].last()


def _period_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2 or clean.iloc[0] == 0:
        return float("nan")
    return float(clean.iloc[-1] / clean.iloc[0] - 1.0)


def _max_favorable_from_start(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty or clean.iloc[0] == 0:
        return 0.0
    return float(clean.max() / clean.iloc[0] - 1.0)


def _amplitude(high: pd.Series, low: pd.Series) -> float:
    high_value = pd.to_numeric(high, errors="coerce").max()
    low_value = pd.to_numeric(low, errors="coerce").min()
    if pd.isna(high_value) or pd.isna(low_value) or low_value == 0:
        return float("nan")
    return float(high_value / low_value - 1.0)


def _tail_head_change(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    midpoint = max(1, len(clean) // 2)
    head = float(clean.iloc[:midpoint].mean())
    tail = float(clean.iloc[midpoint:].mean())
    if head == 0 or not math.isfinite(head):
        return float("nan")
    return tail / head - 1.0


def _normalize_price_series(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    first_valid = clean.dropna()
    if first_valid.empty or first_valid.iloc[0] == 0:
        return clean * np.nan
    return clean / first_valid.iloc[0] * 100.0


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pairs = pd.concat([left, right], axis=1).dropna()
    if len(pairs) < 2:
        return float("nan")
    corr = float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))
    return corr if math.isfinite(corr) else float("nan")


def _sync_label(target_return: float, comparison_return: float) -> str:
    if not math.isfinite(target_return) or not math.isfinite(comparison_return):
        return "数据不足"
    if target_return == 0 or comparison_return == 0:
        return "一方横盘"
    return "同向" if target_return * comparison_return > 0 else "背离"


def _strength_label(excess: float) -> str:
    if not math.isfinite(excess):
        return "数据不足，无法比较。"
    if excess >= 0.05:
        return "目标明显强于对比标的。"
    if excess >= 0.01:
        return "目标略强于对比标的。"
    if excess <= -0.05:
        return "目标明显弱于对比标的。"
    if excess <= -0.01:
        return "目标略弱于对比标的。"
    return "目标与对比标的基本同步。"


def _segment_direction(segment_return: float, previous_return: float) -> str:
    if segment_return >= 0:
        return "反弹" if previous_return < 0 else "上涨"
    return "回撤" if previous_return > 0 else "下跌"


def _percent(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(numeric) else f"{numeric:.2%}"


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")
