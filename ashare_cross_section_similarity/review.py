from __future__ import annotations

from dataclasses import dataclass
from html import escape as html_escape
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
            "波动关系": "数据不足",
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
        "波动关系": _relationship_label(corr),
        "强弱结论": _strength_label(excess),
    }


def build_video_script_profile(
    result: ReviewResult,
    ytd_bars: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    *,
    benchmark_label: str = "沪深300",
    benchmark_event_threshold: float = 0.01,
) -> dict[str, object]:
    symbol = result.symbol
    if result.window.empty:
        return _empty_video_script_profile(symbol, benchmark_label)

    available_window = _script_price_window(ytd_bars, symbol, result.end)
    ytd_start = pd.Timestamp(year=result.end.year, month=1, day=1)
    ytd_window = _script_price_window(ytd_bars, symbol, result.end, start=ytd_start)
    if ytd_window.empty:
        ytd_window = result.window.copy()
    if available_window.empty:
        available_window = result.window.copy()
    post_entry = available_window.loc[pd.to_datetime(available_window["date"], errors="coerce") >= result.start]
    if post_entry.empty:
        post_entry = result.window.copy()

    ytd_return = _period_return(ytd_window["close"])
    max_close_drawdown = max_drawdown(pd.to_numeric(post_entry["close"], errors="coerce").dropna().to_numpy(dtype=float))
    intraday_drawdown = _max_intraday_drawdown(post_entry)
    entry = _entry_challenge(available_window, result.start)
    elasticity = _index_elasticity(
        result.window,
        benchmark,
        benchmark_label=benchmark_label,
        threshold=max(0.0, float(benchmark_event_threshold)),
    )
    return {
        "代码": symbol,
        "YTD样本起点": ytd_window["date"].min() if "date" in ytd_window.columns and not ytd_window.empty else pd.NaT,
        "YTD收益": ytd_return,
        "YTD结论": _ytd_label(ytd_return),
        "买点挑战": entry["label"],
        "买点位置": entry["position"],
        "买点说明": entry["text"],
        "买入后最大收盘回撤": max_close_drawdown,
        "单日最大日内回撤": intraday_drawdown,
        **elasticity,
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

    normalized_paths = _normalize_price_frame(valid)
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


def render_video_script_text(profile: dict[str, object] | None) -> str:
    if not profile:
        return ""
    return "\n\n".join(["**视频脚本视角**：", _video_script_profile_block(profile)])


def render_multi_video_script_text(profiles: list[dict[str, object]] | tuple[dict[str, object], ...]) -> str:
    valid = [profile for profile in profiles if profile]
    if not valid:
        return ""
    lines = ["**视频脚本视角**："]
    lines.extend(_video_script_profile_block(profile) for profile in valid)
    return "\n\n".join(lines)


def rank_review_results(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    valid = [result for result in results if not result.window.empty]
    if not valid:
        return pd.DataFrame(
            columns=[
                "排名",
                "代码",
                "股票",
                "对标指数",
                "指数阶段",
                "强弱等级",
                "区间收益",
                "最大回撤",
                "上涨K占比",
                "相对超额",
                "关键转折点",
                "当前性质",
                "锐评结论",
                "明日验证",
                "排序分",
            ]
        )

    comparison_lookup = _comparison_lookup(comparisons)
    rows: list[dict[str, object]] = []
    for result in valid:
        overview = result.overview
        comparison = comparison_lookup.get(result.symbol, {})
        period_return = _numeric_value(overview.get("return"))
        drawdown = _numeric_value(overview.get("max_drawdown"))
        up_share = _numeric_value(overview.get("up_day_share"))
        excess = _numeric_value(comparison.get("超额收益"))
        nature = _lifecycle_label(overview)
        turning_point = _turning_point_label(result.window)
        score = _ranking_score(period_return, drawdown, up_share, excess, nature)
        grade = _ranking_grade(period_return, drawdown, up_share, excess, nature, score)
        rows.append(
            {
                "代码": result.symbol,
                "股票": str((stock_names or {}).get(result.symbol, "") or "").strip(),
                "对标指数": comparison.get("标的", "-") or "-",
                "指数阶段": _index_phase_label(comparison.get("对比收益")),
                "强弱等级": grade,
                "区间收益": period_return,
                "最大回撤": drawdown,
                "上涨K占比": up_share,
                "相对超额": excess,
                "关键转折点": turning_point,
                "当前性质": nature,
                "锐评结论": _ranking_critique(grade, nature, excess),
                "明日验证": _ranking_tomorrow_check(turning_point, nature, grade),
                "排序分": score,
            }
        )

    ranking = pd.DataFrame(rows).sort_values(["排序分", "相对超额", "区间收益"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "排名", np.arange(1, len(ranking) + 1))
    return ranking


def render_video_script_cards_html(profiles: list[dict[str, object]] | tuple[dict[str, object], ...]) -> str:
    valid = [profile for profile in profiles if profile]
    if not valid:
        return ""
    cards = "\n".join(_video_script_profile_card_html(profile) for profile in valid)
    return f"""
<style>
.review-script-wrap {{
  margin: 0.35rem 0 1.1rem;
}}
.review-script-title {{
  font-size: 0.95rem;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 0.65rem;
}}
.review-script-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 0.9rem;
}}
.review-script-card {{
  border: 1px solid #e5e7eb;
  border-left: 5px solid #64748b;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
  padding: 0.95rem 1rem;
}}
.review-script-card.positive {{
  border-left-color: #16a34a;
  background: linear-gradient(90deg, rgba(22, 163, 74, 0.08), #ffffff 34%);
}}
.review-script-card.negative {{
  border-left-color: #dc2626;
  background: linear-gradient(90deg, rgba(220, 38, 38, 0.08), #ffffff 34%);
}}
.review-script-card.neutral {{
  border-left-color: #2563eb;
  background: linear-gradient(90deg, rgba(37, 99, 235, 0.08), #ffffff 34%);
}}
.review-script-head {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}}
.review-script-kicker {{
  display: block;
  color: #64748b;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0;
  margin-bottom: 0.15rem;
}}
.review-script-name {{
  font-size: 1.05rem;
  line-height: 1.25;
  font-weight: 800;
  color: #111827;
  margin: 0;
}}
.review-script-code {{
  flex: 0 0 auto;
  border-radius: 999px;
  background: #f1f5f9;
  color: #334155;
  font-size: 0.76rem;
  font-weight: 700;
  padding: 0.18rem 0.48rem;
}}
.review-script-metrics {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.45rem;
  margin-bottom: 0.78rem;
}}
.review-script-metric {{
  border: 1px solid #eef2f7;
  border-radius: 6px;
  background: #f8fafc;
  padding: 0.48rem 0.55rem;
}}
.review-script-label {{
  display: block;
  color: #64748b;
  font-size: 0.72rem;
  line-height: 1.2;
  margin-bottom: 0.14rem;
}}
.review-script-value {{
  display: block;
  color: #111827;
  font-size: 0.92rem;
  font-weight: 800;
  line-height: 1.25;
}}
.review-script-value.good {{
  color: #15803d;
}}
.review-script-value.bad {{
  color: #b91c1c;
}}
.review-script-value.warn {{
  color: #b45309;
}}
.review-script-section {{
  border-top: 1px solid #eef2f7;
  padding-top: 0.62rem;
  margin-top: 0.62rem;
}}
.review-script-section-title {{
  color: #334155;
  font-size: 0.78rem;
  font-weight: 800;
  margin-bottom: 0.22rem;
}}
.review-script-body {{
  color: #1f2937;
  font-size: 0.9rem;
  line-height: 1.72;
  margin: 0;
}}
</style>
<div class="review-script-wrap" data-testid="video-script-cards">
  <div class="review-script-title">视频脚本视角</div>
  <div class="review-script-grid">
    {cards}
  </div>
</div>
""".strip()


def render_review_text(
    result: ReviewResult,
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
) -> str:
    if result.window.empty:
        return "\n".join(f"- {warning}" for warning in result.warnings) or "- 没有可复盘的数据。"

    overview = result.overview
    title = _review_display_label(result.symbol, stock_names)
    lifecycle = _lifecycle_label(overview)
    lines = [
        (
            f"**研究复盘：讲证据**\n\n"
            f"{title} 在 {_date_text(result.start)} 至 {_date_text(result.end)} 共 {int(overview['k_bars'])} 根K线。"
            f"区间收益 {_percent(overview['return'])}，最大回撤 {_percent(overview['max_drawdown'])}，"
            f"最大浮盈 {_percent(overview['max_favorable'])}，上涨K线占比 {_percent(overview['up_day_share'])}。"
        ),
        (
            f"**A股语境**：这段走势暂归为“{lifecycle}”。"
            f"{_lifecycle_conclusion(lifecycle)}"
        )
    ]
    if result.main_segments.empty:
        lines.append("**波段结构**：所选区间没有达到阈值的主要上涨或回撤段，走势更接近震荡或小幅单边变化。")
    else:
        lines.append("**关键波段证据**：")
        for _, segment in result.main_segments.sort_values("开始日期").iterrows():
            lines.append(
                "- "
                f"{_date_text(segment['开始日期'])} 至 {_date_text(segment['结束日期'])} "
                f"为{segment['方向']}段，共 {int(segment['K线数'])} 根K线，"
                f"区间收益 {_percent(segment['区间收益'])}，段内最大回撤 {_percent(segment['最大回撤'])}，"
                f"振幅 {_percent(segment['振幅'])}。"
            )

    lines.extend(_comparison_text_lines(comparisons))
    for warning in result.warnings:
        lines.append(f"**提示**：{warning}")
    return "\n\n".join(lines)


def render_multi_review_text(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
) -> str:
    valid = [result for result in results if not result.window.empty]
    if not valid:
        return "- 没有可复盘的数据。"
    returns = pd.Series([result.overview.get("return") for result in valid], dtype="float64").dropna()
    ranking = rank_review_results(valid, comparisons, stock_names=stock_names)
    best = ranking.iloc[0] if not ranking.empty else None
    worst = ranking.iloc[-1] if not ranking.empty else None
    lines = [
        (
            f"**研究复盘：讲证据**\n\n"
            f"本次共复盘 {len(valid)} 个对象，"
            f"平均区间收益 {_percent(returns.mean()) if not returns.empty else '-'}。"
            f"排序第一是 {_ranking_row_title(best)}，最后是 {_ranking_row_title(worst)}。"
        ),
        f"**市场总环境**：{_market_environment_text(comparisons)}",
        "**排序锐评**：",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            "\n".join(
                [
                    f"**第{int(row['排名'])}，{_ranking_row_title(row)}。**",
                    f"它排在这里，是因为{_ranking_reason(row)}。",
                    (
                        f"数据上看，区间收益 {_percent(row['区间收益'])}，最大回撤 {_percent(row['最大回撤'])}，"
                        f"相对超额 {_percent(row['相对超额'])}，属于{row['当前性质']}。"
                    ),
                    f"关键点在{row['关键转折点']}。明日验证：{row['明日验证']}",
                    f"一句话：{row['锐评结论']}",
                ]
            )
        )
    if comparisons is not None and not comparisons.empty:
        if "代码" not in comparisons.columns:
            lines.extend(_comparison_text_lines(comparisons))
        else:
            relation_lines = ["**对标关系**："]
            for symbol, rows in comparisons.groupby("代码", sort=False):
                valid_rows = rows.dropna(subset=["相关性"])
                if valid_rows.empty:
                    continue
                strongest = valid_rows.iloc[pd.to_numeric(valid_rows["相关性"], errors="coerce").abs().argmax()]
                relation_lines.append(
                    "- "
                    f"{_review_display_label(symbol, stock_names)} 相对 {strongest['标的']}："
                    f"{_comparison_script_label(strongest)}"
                )
            if len(relation_lines) > 1:
                lines.extend(relation_lines)
    return "\n\n".join(lines)


def _empty_video_script_profile(symbol: str, benchmark_label: str) -> dict[str, object]:
    return {
        "代码": symbol,
        "YTD样本起点": pd.NaT,
        "YTD收益": float("nan"),
        "YTD结论": "数据不足",
        "买点挑战": "数据不足",
        "买点位置": float("nan"),
        "买点说明": "所选区间没有可用K线，无法判断买点挑战性。",
        "买入后最大收盘回撤": float("nan"),
        "单日最大日内回撤": float("nan"),
        "指数": benchmark_label,
        "指数大涨日样本": 0,
        "指数大涨日标的均值": float("nan"),
        "指数大涨日指数均值": float("nan"),
        "指数大跌日样本": 0,
        "指数大跌日标的均值": float("nan"),
        "指数大跌日指数均值": float("nan"),
        "指数弹性结论": "数据不足，无法判断指数弹性。",
    }


def _script_price_window(
    bars: pd.DataFrame,
    symbol: str,
    end: pd.Timestamp,
    *,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
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
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    start_ts = pd.Timestamp.min if start is None else pd.Timestamp(start)
    frame = frame.loc[
        (frame["stock_code"] == normalize_symbol(symbol))
        & frame["date"].between(start_ts, end_ts)
        & frame["close"].notna()
    ]
    return frame[columns].sort_values("date").reset_index(drop=True)


def _entry_challenge(ytd_window: pd.DataFrame, entry_date: pd.Timestamp) -> dict[str, object]:
    if ytd_window.empty:
        return {"label": "数据不足", "position": float("nan"), "text": "没有足够的买点前序数据。"}
    frame = ytd_window.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.loc[frame["date"] <= pd.Timestamp(entry_date)].sort_values("date").tail(20)
    if frame.empty:
        return {"label": "数据不足", "position": float("nan"), "text": "没有足够的买点前序数据。"}
    entry = frame.iloc[-1]
    entry_close = float(pd.to_numeric(pd.Series([entry["close"]]), errors="coerce").iloc[0])
    entry_open = float(pd.to_numeric(pd.Series([entry.get("open", entry_close)]), errors="coerce").iloc[0])
    highs = pd.to_numeric(frame.get("high", frame["close"]), errors="coerce")
    lows = pd.to_numeric(frame.get("low", frame["close"]), errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    high_value = float(highs.max())
    low_value = float(lows.min())
    position = (entry_close - low_value) / (high_value - low_value) if high_value > low_value else float("nan")
    drawdown_from_high = entry_close / high_value - 1.0 if high_value else float("nan")
    prior_return = entry_close / float(closes.iloc[0]) - 1.0 if len(closes) >= 2 and closes.iloc[0] else 0.0
    candle_return = entry_close / entry_open - 1.0 if entry_open else 0.0

    if math.isfinite(position) and position >= 0.85 and prior_return >= 0.12:
        label = "追高区"
    elif math.isfinite(drawdown_from_high) and drawdown_from_high <= -0.12:
        label = "破位左侧" if prior_return <= -0.08 else "深回调左侧"
    elif math.isfinite(drawdown_from_high) and -0.12 < drawdown_from_high <= -0.03:
        label = "浅回调承接"
    elif candle_return > 0 and abs(candle_return) <= 0.03:
        label = "温和启动"
    else:
        label = "中性观察"

    text = (
        f"买点位于近20根区间的 {_percent(position)} 分位，"
        f"距近20根高点 {_percent(drawdown_from_high)}，"
        f"买入日收盘相对开盘 {_percent(candle_return)}。"
    )
    return {"label": label, "position": position, "text": text}


def _max_intraday_drawdown(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    high = pd.to_numeric(frame.get("high"), errors="coerce")
    low = pd.to_numeric(frame.get("low"), errors="coerce")
    values = (low / high - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.min()) if not values.empty else float("nan")


def _index_elasticity(
    target_window: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    benchmark_label: str,
    threshold: float,
) -> dict[str, object]:
    base = {
        "指数": benchmark_label,
        "指数大涨日样本": 0,
        "指数大涨日标的均值": float("nan"),
        "指数大涨日指数均值": float("nan"),
        "指数大跌日样本": 0,
        "指数大跌日标的均值": float("nan"),
        "指数大跌日指数均值": float("nan"),
        "指数弹性结论": "数据不足，无法判断指数弹性。",
    }
    if benchmark is None or benchmark.empty:
        return base
    aligned = _aligned_close_frame(target_window, benchmark)
    if len(aligned) < 3:
        return base
    returns = aligned[["target", "comparison"]].pct_change().dropna()
    if returns.empty:
        return base
    up = returns.loc[returns["comparison"] >= threshold]
    down = returns.loc[returns["comparison"] <= -threshold]
    if up.empty:
        up = returns.loc[returns["comparison"] > 0]
    if down.empty:
        down = returns.loc[returns["comparison"] < 0]
    up_target = float(up["target"].mean()) if not up.empty else float("nan")
    up_index = float(up["comparison"].mean()) if not up.empty else float("nan")
    down_target = float(down["target"].mean()) if not down.empty else float("nan")
    down_index = float(down["comparison"].mean()) if not down.empty else float("nan")
    base.update(
        {
            "指数大涨日样本": int(len(up)),
            "指数大涨日标的均值": up_target,
            "指数大涨日指数均值": up_index,
            "指数大跌日样本": int(len(down)),
            "指数大跌日标的均值": down_target,
            "指数大跌日指数均值": down_index,
            "指数弹性结论": _elasticity_label(up_target, down_target),
        }
    )
    return base


def _elasticity_label(up_target: float, down_target: float) -> str:
    up_text = "大涨日跟涨不足" if not math.isfinite(up_target) else ("大涨日弹性强" if up_target >= 0.01 else "大涨日弹性弱")
    down_text = "大跌日数据不足" if not math.isfinite(down_target) else ("大跌日抗跌" if down_target > -0.005 else "大跌日承压")
    return f"{up_text}，{down_text}。"


def _ytd_label(value: float) -> str:
    if not math.isfinite(value):
        return "数据不足"
    if value > 0:
        return "年内正收益"
    if value < 0:
        return "年内未赚钱"
    return "年内持平"


def _video_script_profile_block(profile: dict[str, object]) -> str:
    title = _video_script_title(profile)
    setup = _video_setup_sentence(profile)
    turning_point = str(profile.get("关键转折点", "") or "").strip()
    turning_line = f"**关键点**：{turning_point}。" if turning_point else ""
    return "\n\n".join(
        [line for line in [
            f"**{title}**",
            f"**定位**：{setup}",
            f"**入场**：{profile.get('买点挑战', '数据不足')}。{_video_entry_text(profile.get('买点说明'))}",
            (
                f"**压力**：入场后最大收盘回撤 {_percent(profile.get('买入后最大收盘回撤'))}，"
                f"单日日内最大回撤 {_percent(profile.get('单日最大日内回撤'))}。"
            ),
            (
                f"**指数弹性**：{profile.get('指数', '指数')}上涨日 {profile.get('指数大涨日样本', 0)} 天，"
                f"这只票平均 {_percent(profile.get('指数大涨日标的均值'))}，指数平均 {_percent(profile.get('指数大涨日指数均值'))}；"
                f"下跌日 {profile.get('指数大跌日样本', 0)} 天，"
                f"这只票平均 {_percent(profile.get('指数大跌日标的均值'))}，指数平均 {_percent(profile.get('指数大跌日指数均值'))}。"
                f"{profile.get('指数弹性结论', '')}"
            ),
            turning_line,
            f"**明日验证**：{_video_tomorrow_check(profile)}",
        ] if line]
    )


def _video_script_profile_card_html(profile: dict[str, object]) -> str:
    title = _video_script_title(profile)
    symbol = str(profile.get("代码", "") or "").strip()
    theme = _video_script_card_theme(profile.get("YTD收益"))
    ytd_class = _video_value_class(profile.get("YTD收益"), positive_is_good=True)
    drawdown_class = _video_value_class(profile.get("买入后最大收盘回撤"), positive_is_good=False)
    intraday_class = _video_value_class(profile.get("单日最大日内回撤"), positive_is_good=False)
    code_badge = ""
    if symbol and symbol != title:
        code_badge = f'<span class="review-script-code">{html_escape(symbol)}</span>'

    setup_text = _video_setup_sentence(profile)
    entry_text = f"{profile.get('买点挑战', '数据不足')}。{_video_entry_text(profile.get('买点说明'))}"
    pressure_text = (
        f"入场后最大收盘回撤 {_percent(profile.get('买入后最大收盘回撤'))}，"
        f"单日日内最大回撤 {_percent(profile.get('单日最大日内回撤'))}。"
    )
    elasticity_text = (
        f"{profile.get('指数', '指数')}上涨日 {profile.get('指数大涨日样本', 0)} 天，"
        f"这只票平均 {_percent(profile.get('指数大涨日标的均值'))}，指数平均 {_percent(profile.get('指数大涨日指数均值'))}；"
        f"下跌日 {profile.get('指数大跌日样本', 0)} 天，"
        f"这只票平均 {_percent(profile.get('指数大跌日标的均值'))}，指数平均 {_percent(profile.get('指数大跌日指数均值'))}。"
        f"{profile.get('指数弹性结论', '')}"
    )
    turning_point = str(profile.get("关键转折点", "") or "").strip()
    turning_section = _video_card_section_html("关键点", f"{turning_point}。") if turning_point else ""
    tomorrow_text = _video_tomorrow_check(profile)
    return f"""
<article class="review-script-card {theme}">
  <header class="review-script-head">
    <div>
      <span class="review-script-kicker">视频脚本</span>
      <h4 class="review-script-name">{html_escape(title)}</h4>
    </div>
    {code_badge}
  </header>
  <div class="review-script-metrics">
    {_video_metric_html("今年表现", _percent(profile.get("YTD收益")), ytd_class)}
    {_video_metric_html("入场难度", str(profile.get("买点挑战", "数据不足")))}
    {_video_metric_html("收盘回撤", _percent(profile.get("买入后最大收盘回撤")), drawdown_class)}
    {_video_metric_html("日内回撤", _percent(profile.get("单日最大日内回撤")), intraday_class)}
  </div>
  {_video_card_section_html("定位", setup_text)}
  {_video_card_section_html("入场", entry_text)}
  {_video_card_section_html("压力", pressure_text)}
  {_video_card_section_html("指数弹性", elasticity_text)}
  {turning_section}
  {_video_card_section_html("明日验证", tomorrow_text)}
</article>
""".strip()


def _video_metric_html(label: str, value: str, class_name: str = "") -> str:
    value_class = f" {class_name}" if class_name else ""
    return (
        '<div class="review-script-metric">'
        f'<span class="review-script-label">{html_escape(label)}</span>'
        f'<span class="review-script-value{value_class}">{html_escape(value)}</span>'
        "</div>"
    )


def _video_card_section_html(title: str, body: str) -> str:
    return (
        '<section class="review-script-section">'
        f'<div class="review-script-section-title">{html_escape(title)}</div>'
        f'<p class="review-script-body">{html_escape(body)}</p>'
        "</section>"
    )


def _video_script_card_theme(value: object) -> str:
    numeric = _video_numeric(value)
    if numeric > 0:
        return "positive"
    if numeric < 0:
        return "negative"
    return "neutral"


def _video_value_class(value: object, *, positive_is_good: bool) -> str:
    numeric = _video_numeric(value)
    if not math.isfinite(numeric) or numeric == 0:
        return "warn"
    good = numeric > 0 if positive_is_good else numeric > -0.05
    return "good" if good else "bad"


def _video_numeric(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _video_script_title(profile: dict[str, object]) -> str:
    symbol = str(profile.get("代码", "") or "").strip()
    name = str(profile.get("股票", "") or "").strip()
    return name or symbol


def _video_setup_sentence(profile: dict[str, object]) -> str:
    rank = _numeric_value(profile.get("排名"))
    grade = str(profile.get("强弱等级", "") or "").strip()
    nature = str(profile.get("当前性质", "") or "").strip()
    critique = str(profile.get("锐评结论", "") or "").strip()
    if math.isfinite(rank) and grade and nature:
        prefix = f"第{int(rank)}，强弱等级{grade}，当前是{nature}。"
        return f"{prefix}{critique}" if critique else prefix

    ytd = profile.get("YTD收益")
    entry_label = str(profile.get("买点挑战", "数据不足") or "数据不足")
    drawdown = _video_numeric(profile.get("买入后最大收盘回撤"))
    ytd_text = f"YTD {_percent(ytd)}，{_video_ytd_phrase(ytd)}"
    if _video_numeric(ytd) >= 0.20 and math.isfinite(drawdown) and drawdown > -0.12:
        position = "主线硬货"
        comment = "强不是因为涨得多，而是涨得多、回撤还压得住。"
    elif _video_numeric(ytd) >= 0.10 and math.isfinite(drawdown) and drawdown <= -0.15:
        position = "弹性冲浪"
        comment = "能赚钱，但路上会把人甩下车。"
    elif _video_numeric(ytd) < 0:
        position = "弱势修复"
        comment = "不是没反弹，是现在还没走出地位。"
    elif entry_label in {"追高区", "温和启动"}:
        position = "趋势观察"
        comment = "方向不差，但不要只凭热度追。"
    else:
        position = "轮动观察"
        comment = "有机会，但还要看承接和放量。"
    return f"{ytd_text}这段更像{position}。{comment}"


def _video_tomorrow_check(profile: dict[str, object]) -> str:
    ranked_check = str(profile.get("明日验证", "") or "").strip()
    if ranked_check:
        return ranked_check

    entry_label = str(profile.get("买点挑战", "") or "")
    ytd = _video_numeric(profile.get("YTD收益"))
    drawdown = _video_numeric(profile.get("买入后最大收盘回撤"))
    if entry_label == "追高区":
        return "重点看高开后能不能继续放量；如果高开低走，就是先手资金兑现。"
    if entry_label == "浅回调承接":
        return "重点看回踩是否缩量、是否守住近端低点；守得住像洗盘，守不住就是转弱。"
    if math.isfinite(ytd) and ytd < 0:
        return "重点看反弹有没有放量；没有放量，就别急着把弱修复当主升。"
    if math.isfinite(drawdown) and drawdown <= -0.15:
        return "重点看大跌日能不能收回来；能收回来是弹性，收不回来就是风险。"
    return "重点看承接。强的缩量回踩不破，才说明资金还在；高开低走就要防兑现。"


def _video_ytd_phrase(value: object) -> str:
    numeric = _video_numeric(value)
    if not math.isfinite(numeric):
        return "今年数据还不够完整。"
    if numeric > 0:
        return "今年已经跑出正收益。"
    if numeric < 0:
        return "今年还没有真正赚钱。"
    return "今年基本打平。"


def _video_entry_text(text: object) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "前序K线不够，入场位置暂时不下结论。"
    cleaned = cleaned.replace("买点位于", "买点在")
    cleaned = cleaned.replace("近20根区间", "近20根走势")
    cleaned = cleaned.replace("距近20根高点", "离近20根高点")
    cleaned = cleaned.replace("买入日收盘相对开盘", "当天收盘较开盘")
    return cleaned


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


def _normalize_price_frame(values: pd.DataFrame) -> pd.DataFrame:
    clean = values.astype(float)
    bases = clean.bfill().iloc[0].replace(0, np.nan)
    return clean.divide(bases, axis=1) * 100.0


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


def _relationship_label(corr: float) -> str:
    if not math.isfinite(corr):
        return "数据不足"
    if corr >= 0.70:
        return "同步跟随"
    if corr >= 0.35:
        return "部分跟随"
    if corr <= -0.35:
        return "反向背离"
    return "不相关"


def _comparison_text_lines(comparisons: pd.DataFrame | None) -> list[str]:
    if comparisons is None or comparisons.empty:
        return []
    valid = comparisons.dropna(subset=["超额收益"])
    if valid.empty:
        return []
    lines = ["**对标关系**："]
    for _, row in valid.iterrows():
        lines.append(
            "- "
            f"{row['标的']}：{_comparison_script_label(row)}"
        )
    return lines


def _comparison_lookup(comparisons: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if comparisons is None or comparisons.empty or "代码" not in comparisons.columns:
        return {}
    lookup: dict[str, dict[str, object]] = {}
    for symbol, rows in comparisons.groupby("代码", sort=False):
        selected = _best_comparison_row(rows)
        if selected:
            lookup[normalize_symbol(str(symbol))] = selected
    return lookup


def _best_comparison_row(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {}
    clean = rows.copy()
    clean["_excess"] = pd.to_numeric(clean.get("超额收益"), errors="coerce")
    valid = clean.dropna(subset=["_excess"])
    selected = valid.iloc[0] if not valid.empty else clean.iloc[0]
    return {key: value for key, value in selected.drop(labels=["_excess"], errors="ignore").to_dict().items()}


def _review_display_label(symbol: str, stock_names: dict[str, str] | None) -> str:
    name = str((stock_names or {}).get(symbol, "") or "").strip()
    return f"{name}（{symbol}）" if name else symbol


def _ranking_row_title(row: pd.Series | None) -> str:
    if row is None:
        return "-"
    symbol = str(row.get("代码", "") or "").strip()
    name = str(row.get("股票", "") or "").strip()
    return f"{name}（{symbol}）" if name else symbol


def _ranking_score(period_return: float, drawdown: float, up_share: float, excess: float, nature: str) -> float:
    excess_score = _clamp((excess + 0.20) / 0.50, 0.0, 1.0) * 35.0 if math.isfinite(excess) else 12.0
    drawdown_score = _clamp((drawdown + 0.35) / 0.35, 0.0, 1.0) * 25.0 if math.isfinite(drawdown) else 8.0
    return_score = _clamp((period_return + 0.10) / 0.60, 0.0, 1.0) * 25.0 if math.isfinite(period_return) else 6.0
    smooth_score = _clamp(up_share, 0.0, 1.0) * 15.0 if math.isfinite(up_share) else 5.0
    nature_adjust = {
        "主升": 8.0,
        "温和启动": 4.0,
        "弹性冲浪": 1.0,
        "震荡修复": 0.0,
        "弱势反抽": -5.0,
        "走弱": -10.0,
    }.get(nature, 0.0)
    return float(excess_score + drawdown_score + return_score + smooth_score + nature_adjust)


def _ranking_grade(period_return: float, drawdown: float, up_share: float, excess: float, nature: str, score: float) -> str:
    if nature == "走弱" or (math.isfinite(period_return) and period_return < -0.08):
        return "D"
    if math.isfinite(excess) and excess >= 0.15 and math.isfinite(drawdown) and drawdown > -0.12 and up_share >= 0.5:
        return "S"
    if score >= 70:
        return "A"
    if score >= 55:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def _ranking_reason(row: pd.Series) -> str:
    grade = str(row.get("强弱等级", "") or "")
    excess = _numeric_value(row.get("相对超额"))
    drawdown = _numeric_value(row.get("最大回撤"))
    if grade in {"S", "A"} and math.isfinite(excess) and excess > 0:
        return "相对指数有超额，回撤控制也更好"
    if math.isfinite(drawdown) and drawdown <= -0.18:
        return "弹性够大，但持有难度也大"
    if grade in {"C", "D"}:
        return "超额和趋势顺滑度还没证明强度"
    return "收益、回撤和上涨K占比的综合位置更靠前"


def _ranking_critique(grade: str, nature: str, excess: float) -> str:
    if grade == "S":
        return "这个是真强，资金真干了，不是蹭行情。"
    if grade == "A":
        return "趋势还在，等回踩比追高舒服。"
    if grade == "B":
        return "有肉，但别信仰，节奏比信念更重要。"
    if grade == "C":
        return "便宜不等于强，低位不等于启动，现在只能先看修复。"
    if nature == "弱势反抽" or (math.isfinite(excess) and excess < -0.05):
        return "看着动了，其实还没地位。"
    return "这一段大概率被薅得差不多了，先看风险。"


def _ranking_tomorrow_check(turning_point: str, nature: str, grade: str) -> str:
    if "前高" in turning_point or "箱体上沿" in turning_point:
        return "放量突破算强，冲高回落就是先手兑现。"
    if "5日/10日" in turning_point:
        return "缩量回踩不破算强，跌破短均就要降一档。"
    if "20日" in turning_point:
        return "站稳20日线才算修复延续，放量跌回去就是反抽失败。"
    if "破位" in turning_point or grade == "D" or nature == "走弱":
        return "先看能不能止跌，不能止跌就别硬说洗盘。"
    return "继续看承接和放量，强弱分界在最近的高低点。"


def _turning_point_label(window: pd.DataFrame) -> str:
    if window.empty or len(window) < 2:
        return "数据不足"
    frame = window.copy().sort_values("date")
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame.get("high", close), errors="coerce").fillna(close)
    low = pd.to_numeric(frame.get("low", close), errors="coerce").fillna(close)
    open_ = pd.to_numeric(frame.get("open", close), errors="coerce").fillna(close)
    last_close = float(close.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    last_open = float(open_.iloc[-1])
    if last_high > last_low and (last_high - last_close) / (last_high - last_low) >= 0.65 and last_close < last_open:
        return "放量长上影/高开低走"
    recent_high = float(high.tail(min(20, len(high))).max())
    ma5 = float(close.tail(min(5, len(close))).mean())
    ma10 = float(close.tail(min(10, len(close))).mean())
    ma20 = float(close.tail(min(20, len(close))).mean())
    if recent_high > 0 and last_close >= recent_high * 0.98:
        return "前高/箱体上沿"
    if last_close >= ma5 and last_close >= ma10:
        return "5日/10日线承接"
    if last_close >= ma20:
        return "20日线观察"
    return "破位观察"


def _index_phase_label(value: object) -> str:
    numeric = _numeric_value(value)
    if not math.isfinite(numeric):
        return "数据不足"
    if numeric >= 0.12:
        return "指数主升"
    if numeric >= 0.03:
        return "指数修复"
    if numeric <= -0.08:
        return "指数退潮"
    if numeric <= -0.03:
        return "高位分歧"
    return "横盘震荡"


def _market_environment_text(comparisons: pd.DataFrame | None) -> str:
    if comparisons is None or comparisons.empty or "对比收益" not in comparisons.columns:
        return "对比数据不足，先按个体超额、回撤和关键位置排序。"
    values = pd.to_numeric(comparisons["对比收益"], errors="coerce").dropna()
    if values.empty:
        return "对比数据不足，先按个体超额、回撤和关键位置排序。"
    average_return = float(values.mean())
    return f"对标组合平均收益 {_percent(average_return)}，当前更像{_index_phase_label(average_return)}。"


def _numeric_value(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if not pd.isna(numeric) else float("nan")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _lifecycle_label(overview: dict[str, float]) -> str:
    period_return = float(overview.get("return", float("nan")))
    drawdown = float(overview.get("max_drawdown", float("nan")))
    up_share = float(overview.get("up_day_share", float("nan")))
    max_favorable = float(overview.get("max_favorable", float("nan")))
    if not math.isfinite(period_return):
        return "数据不足"
    if period_return >= 0.20 and (not math.isfinite(drawdown) or drawdown > -0.12) and up_share >= 0.5:
        return "主升"
    if period_return >= 0.12 and math.isfinite(drawdown) and drawdown <= -0.15:
        return "弹性冲浪"
    if period_return >= 0.05:
        return "温和启动"
    if period_return >= -0.03 and max_favorable >= 0.08:
        return "震荡修复"
    if period_return < 0 and max_favorable >= 0.06:
        return "弱势反抽"
    if period_return < 0:
        return "走弱"
    return "横盘观察"


def _lifecycle_conclusion(label: str) -> str:
    mapping = {
        "主升": "趋势有地位，但强势阶段更要防高开兑现。",
        "弹性冲浪": "收益弹性够，但回撤也大，适合看节奏，不适合当稳定主升。",
        "温和启动": "开始有资金参与，后续要看是否放量和是否能守住短期均线。",
        "震荡修复": "有修复动作，但还没形成清晰主升，需要继续看突破。",
        "弱势反抽": "有反弹，但数据还没有证明趋势反转。",
        "走弱": "价格重心偏弱，先看止跌，不宜硬讲主线。",
        "横盘观察": "方向还不够明确，重点看放量突破或破位。",
        "数据不足": "样本不足，暂不下阶段结论。",
    }
    return mapping.get(label, "先按数据观察，不做额外推断。")


def _comparison_script_label(row: pd.Series) -> str:
    relationship = str(row.get("波动关系", "数据不足") or "数据不足")
    sync = str(row.get("同步关系", "数据不足") or "数据不足")
    corr = _decimal(row.get("相关性"))
    excess = _percent(row.get("超额收益"))
    conclusion = (
        str(row.get("强弱结论", "") or "")
        .replace("目标", "它")
        .replace("对比标的", "对比对象")
        .strip()
        .rstrip("。")
    )
    if relationship == "同步跟随" and str(excess).startswith("-"):
        role = "跟得上方向，但强度不够。"
    elif relationship == "同步跟随":
        role = "跟着指数走，而且有一定强度。"
    elif relationship == "不相关":
        role = "和对比对象不是一条节奏，更像独立逻辑。"
    elif relationship == "反向背离":
        role = "和对比对象明显背离，需要单独看驱动。"
    else:
        role = "关系不算稳定，需要继续观察。"
    suffix = f"{role}{conclusion}。" if conclusion else role
    return f"波动关系 {relationship}，同步关系 {sync}，相关性 {corr}，超额收益 {excess}。{suffix}"


def _segment_direction(segment_return: float, previous_return: float) -> str:
    if segment_return >= 0:
        return "反弹" if previous_return < 0 else "上涨"
    return "回撤" if previous_return > 0 else "下跌"


def _percent(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(numeric) else f"{numeric:.2%}"


def _decimal(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(numeric) else f"{float(numeric):.2f}"


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")
