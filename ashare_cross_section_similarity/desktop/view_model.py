from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ashare_cross_section_similarity.history import HistorySearchResult
from ashare_cross_section_similarity.review import ReviewResult, rank_review_results, render_review_text
from ashare_cross_section_similarity.similarity import CrossSectionSearchResult
from ashare_cross_section_similarity.universe import normalize_symbol


@dataclass(frozen=True)
class LineSeries:
    label: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class BarValue:
    label: str
    value: float


@dataclass(frozen=True)
class CandlestickBar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class ChartHighlight:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    color: str
    opacity: float = 0.35


@dataclass(frozen=True)
class CandlestickSeries:
    label: str
    rows: tuple[CandlestickBar, ...]
    highlights: tuple[ChartHighlight, ...] = ()


@dataclass(frozen=True)
class OverviewModule:
    key: str
    title: str
    summary: str
    chart_title: str
    action_label: str
    page_index: int


@dataclass(frozen=True)
class ReviewCritiqueCard:
    symbol: str
    title: str
    grade: str
    nature: str
    critique: str
    metrics: tuple[tuple[str, str], ...]


BASE_DATA_FIELDS = ("data_root", "timeframe", "adjust", "data_mode")
WINDOW_HIGHLIGHT_COLOR = "#2563eb"
WINDOW_HIGHLIGHT_OPACITY = 0.35
SEGMENT_HIGHLIGHT_OPACITY = 0.12
SEGMENT_HIGHLIGHT_COLORS = {
    "上涨": "#ef4444",
    "下跌": "#22c55e",
    "反弹": "#fb7185",
    "回撤": "#3b82f6",
}


def overview_modules() -> tuple[OverviewModule, ...]:
    return (
        OverviewModule(
            key="history",
            title="历史时序",
            summary="同一标的历史阶段相似搜索，适合找过往走势镜像。",
            chart_title="走势对比",
            action_label="进入历史",
            page_index=1,
        ),
        OverviewModule(
            key="cross_section",
            title="横截面",
            summary="同一时间附近跨标的搜索，适合找当前市场里的同款走势。",
            chart_title="Top 相似度",
            action_label="进入横截面",
            page_index=2,
        ),
        OverviewModule(
            key="review",
            title="走势复盘",
            summary="把单段行情压缩成收益、回撤、波段证据和锐评结论。",
            chart_title="复盘走势",
            action_label="进入复盘",
            page_index=3,
        ),
        OverviewModule(
            key="data",
            title="数据管理",
            summary="检查本地覆盖，预览和执行 K 线文件迁移。",
            chart_title="覆盖/迁移",
            action_label="进入数据",
            page_index=4,
        ),
    )


def visible_data_fields(data_mode_label: str, data_source: str) -> tuple[str, ...]:
    if data_mode_label != "数据 API":
        return BASE_DATA_FIELDS
    fields = [*BASE_DATA_FIELDS, "data_source", "api_url"]
    if data_source == "tdx":
        fields.append("tdx_path")
    return tuple(fields)


def history_line_series(result: HistorySearchResult, *, max_matches: int = 3) -> list[LineSeries]:
    series = [_close_line("当前窗口", result.current_window)]
    for index, frame in enumerate(result.historical_windows[:max_matches], start=1):
        item = _close_line(f"相似 {index}", frame)
        if item.values:
            series.append(item)
    return [item for item in series if item.values]


def history_candlestick_series(
    result: HistorySearchResult,
    bars: pd.DataFrame | None = None,
    *,
    max_matches: int = 6,
    forward_bars: int = 10,
) -> list[CandlestickSeries]:
    source = _prepared_symbol_bars(bars, result.symbol) if bars is not None else pd.DataFrame()
    windows: list[tuple[str, pd.DataFrame]] = [("当前窗口", result.current_window)]
    windows.extend((f"相似 {index}", frame) for index, frame in enumerate(result.historical_windows[:max_matches], start=1))
    series: list[CandlestickSeries] = []
    for label, window in windows:
        chart_window = _window_with_forward_bars(source, window, forward_bars) if not source.empty else window
        item = _candlestick_series(
            _dated_label(label, window),
            chart_window,
            highlights=(_window_highlight(window, "指定区间"),),
        )
        if item.rows:
            series.append(item)
    return series


def history_stat_frames(result: HistorySearchResult) -> tuple[tuple[str, pd.DataFrame], ...]:
    return (
        ("后验观察统计", _history_forward_summary(result.results)),
        ("相似度分层表现", _history_bucket_summary(result.results)),
    )


def size_spread_frame(
    bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp = "2016-01-01",
    small_symbol: str = "000852.SH",
    large_symbol: str = "000300.SH",
) -> pd.DataFrame:
    columns = ["date", "中证1000归一收益", "沪深300归一收益", "大小盘价差率"]
    if bars.empty:
        return pd.DataFrame(columns=columns)
    small = normalize_symbol(small_symbol)
    large = normalize_symbol(large_symbol)
    frame = bars.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_symbol)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.loc[
        (frame["date"] >= pd.Timestamp(start))
        & frame["stock_code"].isin([small, large])
        & frame["close"].notna()
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    pivot = frame.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last").sort_index()
    if small not in pivot.columns or large not in pivot.columns:
        return pd.DataFrame(columns=columns)
    pivot = pivot.dropna(subset=[small, large])
    if pivot.empty:
        return pd.DataFrame(columns=columns)
    base = pivot.iloc[0]
    if base[small] <= 0 or base[large] <= 0:
        return pd.DataFrame(columns=columns)
    small_return = pivot[small] / base[small] - 1.0
    large_return = pivot[large] / base[large] - 1.0
    return pd.DataFrame(
        {
            "date": pivot.index,
            "中证1000归一收益": small_return.to_numpy(dtype=float),
            "沪深300归一收益": large_return.to_numpy(dtype=float),
            "大小盘价差率": (small_return - large_return).to_numpy(dtype=float),
        }
    ).reset_index(drop=True)


def size_spread_line_series(spread: pd.DataFrame) -> list[LineSeries]:
    if spread.empty or "大小盘价差率" not in spread.columns:
        return []
    values = pd.to_numeric(spread["大小盘价差率"], errors="coerce").dropna().tolist()
    return [LineSeries(label="大小盘价差率", values=tuple(float(value) for value in values))] if values else []


def size_spread_window_stats(
    spread: pd.DataFrame,
    current_window: pd.DataFrame,
    historical_windows: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    *,
    top_n: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if spread.empty or "大小盘价差率" not in spread.columns:
        return pd.DataFrame(rows)
    spread_frame = spread.copy()
    spread_frame["date"] = pd.to_datetime(spread_frame["date"], errors="coerce")
    spread_frame["大小盘价差率"] = pd.to_numeric(spread_frame["大小盘价差率"], errors="coerce")
    spread_frame = spread_frame.dropna(subset=["date", "大小盘价差率"]).sort_values("date")
    windows: list[tuple[str, pd.DataFrame]] = [("当前窗口", current_window)]
    windows.extend((f"样本{index}", window) for index, window in enumerate(list(historical_windows)[:top_n], start=1))
    for label, window in windows:
        if window.empty or "date" not in window.columns:
            continue
        start = pd.Timestamp(window["date"].min())
        end = pd.Timestamp(window["date"].max())
        window_spread = spread_frame.loc[spread_frame["date"].between(start, end)]
        if window_spread.empty:
            continue
        start_value = float(window_spread["大小盘价差率"].iloc[0])
        end_value = float(window_spread["大小盘价差率"].iloc[-1])
        as_of_history = spread_frame.loc[spread_frame["date"] <= window_spread["date"].iloc[-1], "大小盘价差率"].dropna()
        percentile = float((as_of_history <= end_value).mean()) if not as_of_history.empty else float("nan")
        rows.append(
            {
                "窗口": label,
                "区间开始": window_spread["date"].iloc[0],
                "区间结束": window_spread["date"].iloc[-1],
                "起点价差率": start_value,
                "终点价差率": end_value,
                "区间变化": end_value - start_value,
                "区间均值": float(window_spread["大小盘价差率"].mean()),
                "终点历史分位": percentile,
            }
        )
    return pd.DataFrame(rows)


def cross_section_score_bars(result: CrossSectionSearchResult, *, limit: int = 10) -> list[BarValue]:
    frame = result.results
    if frame.empty or "symbol" not in frame.columns or "综合相似度" not in frame.columns:
        return []
    values: list[BarValue] = []
    for _, row in frame.head(limit).iterrows():
        values.append(BarValue(label=str(row["symbol"]), value=float(row["综合相似度"])))
    return values


def cross_section_stat_frames(result: CrossSectionSearchResult) -> tuple[tuple[str, pd.DataFrame], ...]:
    return (
        ("后验观察统计", _cross_section_forward_summary(result.results)),
        ("相似度分层表现", _cross_section_bucket_summary(result.results)),
        ("跳过样本", result.skipped.copy()),
    )


def cross_section_candlestick_series(
    result: CrossSectionSearchResult,
    bars: pd.DataFrame,
    *,
    max_matches: int = 6,
    forward_bars: int = 10,
    stock_names: dict[str, str] | None = None,
) -> list[CandlestickSeries]:
    symbols = [result.target_symbol]
    if not result.results.empty and "symbol" in result.results.columns:
        symbols.extend(str(symbol) for symbol in result.results["symbol"].head(max_matches).tolist())
    series: list[CandlestickSeries] = []
    for symbol in symbols:
        start, end = _cross_section_symbol_window(result, symbol)
        symbol_bars = _prepared_symbol_bars(bars, symbol)
        if symbol_bars.empty:
            continue
        window = symbol_bars.loc[symbol_bars["date"].between(start, end)].reset_index(drop=True)
        chart_window = _window_with_forward_bars(symbol_bars, window, forward_bars)
        label = _stock_label(symbol, stock_names, is_target=normalize_symbol(symbol) == result.target_symbol)
        item = _candlestick_series(
            _dated_label(label, window),
            chart_window,
            highlights=(_window_highlight(window, "指定区间"),),
        )
        if item.rows:
            series.append(item)
    return series


def review_line_series(result: ReviewResult) -> list[LineSeries]:
    series = _close_line(result.symbol, result.window)
    return [series] if series.values else []


def review_candlestick_series(results: list[ReviewResult] | tuple[ReviewResult, ...]) -> list[CandlestickSeries]:
    return [
        series
        for result in results
        if (
            series := _candlestick_series(
                result.symbol,
                result.window,
                highlights=_review_segment_highlights(result.main_segments),
            )
        ).rows
    ]


def review_critique_cards(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
    direction_by_symbol: dict[str, str] | None = None,
) -> list[ReviewCritiqueCard]:
    ranking = rank_review_results(results, comparisons, stock_names=stock_names, direction_by_symbol=direction_by_symbol)
    if ranking.empty:
        return []
    cards: list[ReviewCritiqueCard] = []
    for _, row in ranking.iterrows():
        symbol = str(row.get("代码", "") or "").strip()
        name = str(row.get("股票", "") or "").strip()
        title = f"{name}（{symbol}）" if name else symbol
        cards.append(
            ReviewCritiqueCard(
                symbol=symbol,
                title=title,
                grade=str(row.get("强弱等级", "") or "").strip(),
                nature=str(row.get("当前性质", "") or "").strip(),
                critique=str(row.get("锐评结论", "") or "").strip(),
                metrics=(
                    ("收益", _percent_text(row.get("区间收益"))),
                    ("回撤", _percent_text(row.get("最大回撤"))),
                    ("超额", _percent_text(row.get("相对超额"))),
                    ("转折", str(row.get("关键转折点", "-") or "-")),
                ),
            )
        )
    return cards


def review_script_profile_cards(
    profiles: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> list[ReviewCritiqueCard]:
    cards: list[ReviewCritiqueCard] = []
    for profile in profiles:
        symbol = str(profile.get("代码", "") or "").strip()
        name = str(profile.get("股票", "") or "").strip()
        title = f"{name}（{symbol}）" if name and symbol else name or symbol
        grade = str(profile.get("强弱等级", profile.get("标签", "")) or "").strip()
        nature = str(profile.get("当前性质", "") or "").strip()
        critique = str(profile.get("锐评结论", profile.get("结局", "")) or "").strip()
        cards.append(
            ReviewCritiqueCard(
                symbol=symbol,
                title=title,
                grade=grade,
                nature=nature or "视频脚本视角",
                critique=critique,
                metrics=(
                    ("YTD", _percent_text(profile.get("YTD收益"))),
                    ("回撤", _percent_text(profile.get("买入后最大收盘回撤"))),
                    ("弹性", _percent_text(profile.get("指数大涨日标的均值"))),
                    ("转折", str(profile.get("关键转折点", "-") or "-")),
                ),
            )
        )
    return cards


def review_script_profile_frame(
    profiles: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> pd.DataFrame:
    columns = [
        "代码",
        "股票",
        "排名",
        "标签",
        "当前性质",
        "对标指数",
        "指数阶段",
        "区间收益",
        "最大回撤",
        "相对超额",
        "关键转折点",
        "锐评结论",
        "结局",
    ]
    if not profiles:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(profiles).copy()
    if "强弱等级" in result.columns and "标签" not in result.columns:
        result["标签"] = result["强弱等级"]
    if "结局" not in result.columns:
        result["结局"] = result.apply(_script_profile_ending, axis=1)
    return result[[column for column in columns if column in result.columns]]


def review_overview_frame(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    *,
    stock_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        overview = result.overview
        rows.append(
            {
                "代码": result.symbol,
                "股票": (stock_names or {}).get(result.symbol, ""),
                "K线数": int(overview.get("k_bars", 0) or 0),
                "区间收益": overview.get("return"),
                "最大回撤": overview.get("max_drawdown"),
                "最大浮盈": overview.get("max_favorable"),
                "波动率": overview.get("volatility"),
                "上涨K线占比": overview.get("up_day_share"),
            }
        )
    return pd.DataFrame(rows)


def review_ranking_frame(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
    direction_by_symbol: dict[str, str] | None = None,
) -> pd.DataFrame:
    ranking = rank_review_results(results, comparisons, stock_names=stock_names, direction_by_symbol=direction_by_symbol)
    columns = [
        "排名",
        "代码",
        "股票",
        "所属方向",
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
    ]
    return ranking[[column for column in columns if column in ranking.columns]].copy()


def review_segments_frame(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    *,
    stock_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = result.main_segments.copy()
        if frame.empty:
            continue
        frame.insert(0, "股票", (stock_names or {}).get(result.symbol, ""))
        frame.insert(0, "代码", result.symbol)
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["代码", "股票", "方向", "开始日期", "结束日期", "K线数", "区间收益", "最大回撤", "振幅"])


def review_comparisons_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["代码", "股票", "标的", "样本数", "目标收益", "对比收益", "超额收益", "相关性", "同步关系", "波动关系", "强弱结论"]
        )
    columns = [
        "代码",
        "股票",
        "标的",
        "样本数",
        "目标收益",
        "对比收益",
        "超额收益",
        "相关性",
        "同步关系",
        "波动关系",
        "强弱结论",
    ]
    return frame[[column for column in columns if column in frame.columns]].copy()


def review_relative_line_series(
    reviews: list[ReviewResult] | tuple[ReviewResult, ...],
    comparisons: list[tuple[str, pd.DataFrame]] | tuple[tuple[str, pd.DataFrame], ...] = (),
) -> list[LineSeries]:
    series: list[LineSeries] = []
    for result in reviews:
        line = _close_line(result.symbol, result.window)
        if line.values:
            series.append(line)
    for label, frame in comparisons:
        line = _close_line(label, frame)
        if line.values:
            series.append(line)
    return series


def data_status_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "requested_start" not in result.columns:
        result["requested_start"] = result.get("start", pd.NA)
    if "requested_end" not in result.columns:
        result["requested_end"] = result.get("end", pd.NA)
    if "local_start" not in result.columns:
        result["local_start"] = result.get("start", pd.NA)
    if "local_end" not in result.columns:
        result["local_end"] = result.get("end", pd.NA)
    display_columns = [
        column
        for column in ["symbol", "status", "rows", "requested_start", "requested_end", "local_start", "local_end", "message"]
        if column in result.columns
    ]
    result = result[display_columns]
    for column in ["requested_start", "requested_end", "local_start", "local_end"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result.rename(
        columns={
            "requested_start": "请求开始",
            "requested_end": "请求结束",
            "local_start": "本地开始",
            "local_end": "本地结束",
        }
    )


def data_status_bars(frame: pd.DataFrame) -> list[BarValue]:
    if frame.empty or "status" not in frame.columns:
        return []
    counts = frame["status"].astype(str).value_counts(sort=False)
    return [BarValue(label=str(label), value=float(count)) for label, count in counts.items()]


def review_text(
    result: ReviewResult,
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
) -> str:
    return render_review_text(result, comparisons, stock_names=stock_names)


def _history_forward_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    similarity = pd.to_numeric(frame.get("综合相似度"), errors="coerce")
    for column in _forward_return_columns(frame):
        horizon = column.removeprefix("t_plus_").removesuffix("_return")
        values = pd.to_numeric(frame[column], errors="coerce")
        valid_mask = values.notna()
        valid = values.loc[valid_mask]
        if valid.empty:
            continue
        best_index = valid.idxmax()
        worst_index = valid.idxmin()
        fallback = pd.Series(index=frame.index, dtype="float64")
        drawdowns = pd.to_numeric(frame.get(f"t_plus_{horizon}_max_drawdown", fallback), errors="coerce").loc[valid_mask]
        favorable = pd.to_numeric(frame.get(f"t_plus_{horizon}_max_favorable", fallback), errors="coerce").loc[valid_mask]
        valid_similarity = similarity.loc[valid_mask]
        rows.append(
            {
                "观察窗口": f"后{horizon}根",
                "样本数": int(valid.count()),
                "平均收益": float(valid.mean()),
                "中位收益": float(valid.median()),
                "胜率": float((valid > 0).mean()),
                "平均最大回撤": float(drawdowns.mean()) if not drawdowns.dropna().empty else float("nan"),
                "平均最大浮盈": float(favorable.mean()) if not favorable.dropna().empty else float("nan"),
                "最好窗口": _history_window_label(frame, best_index),
                "最好收益": float(valid.loc[best_index]),
                "最差窗口": _history_window_label(frame, worst_index),
                "最差收益": float(valid.loc[worst_index]),
                "相似度-收益相关": _series_corr(valid_similarity, valid),
            }
        )
    return pd.DataFrame(rows)


def _history_bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    buckets = [("Top3", 3), ("Top6", 6), ("Top10", 10), ("全部", len(frame))]
    return_columns = _forward_return_columns(frame)
    for label, size in buckets:
        sample = frame.head(size)
        row: dict[str, object] = {
            "分层": label,
            "样本数": int(len(sample)),
            "平均综合相似度": float(pd.to_numeric(sample.get("综合相似度"), errors="coerce").mean()),
        }
        for column in return_columns:
            horizon = column.removeprefix("t_plus_").removesuffix("_return")
            values = pd.to_numeric(sample[column], errors="coerce").dropna()
            row[f"后{horizon}根平均收益"] = float(values.mean()) if not values.empty else float("nan")
            row[f"后{horizon}根胜率"] = float((values > 0).mean()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _cross_section_forward_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    similarity = pd.to_numeric(frame.get("综合相似度"), errors="coerce")
    for column in _forward_return_columns(frame):
        horizon = column.removeprefix("t_plus_").removesuffix("_return")
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        best_index = values.idxmax()
        worst_index = values.idxmin()
        rows.append(
            {
                "观察窗口": f"后{horizon}根",
                "样本数": int(valid.count()),
                "平均收益": float(valid.mean()),
                "中位收益": float(valid.median()),
                "胜率": float((valid > 0).mean()),
                "收益波动": float(valid.std(ddof=0)) if len(valid) else 0.0,
                "最好标的": str(frame.loc[best_index, "symbol"]) if "symbol" in frame.columns else "",
                "最好收益": float(values.loc[best_index]),
                "最差标的": str(frame.loc[worst_index, "symbol"]) if "symbol" in frame.columns else "",
                "最差收益": float(values.loc[worst_index]),
                "相似度-收益相关": _series_corr(similarity, values),
            }
        )
    return pd.DataFrame(rows)


def _cross_section_bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    buckets = [("Top3", 3), ("Top6", 6), ("Top10", 10), ("全部", len(frame))]
    return_columns = _forward_return_columns(frame)
    for label, size in buckets:
        sample = frame.head(size)
        row: dict[str, object] = {
            "分层": label,
            "样本数": int(len(sample)),
            "平均综合相似度": float(pd.to_numeric(sample.get("综合相似度"), errors="coerce").mean()),
        }
        for column in return_columns:
            horizon = column.removeprefix("t_plus_").removesuffix("_return")
            values = pd.to_numeric(sample[column], errors="coerce").dropna()
            row[f"后{horizon}根平均收益"] = float(values.mean()) if not values.empty else float("nan")
            row[f"后{horizon}根胜率"] = float((values > 0).mean()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _forward_return_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("t_plus_") and column.endswith("_return")]


def _series_corr(left: pd.Series, right: pd.Series) -> float:
    pairs = pd.concat([left, right], axis=1).dropna()
    if len(pairs) < 2:
        return float("nan")
    corr = float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))
    return corr if pd.notna(corr) else float("nan")


def _history_window_label(frame: pd.DataFrame, index: int) -> str:
    if "窗口开始" not in frame.columns or "窗口结束" not in frame.columns:
        return ""
    return f"{_date_label(frame.loc[index, '窗口开始'])} 至 {_date_label(frame.loc[index, '窗口结束'])}"


def _date_label(value: object) -> str:
    timestamp = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(timestamp) else pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _candlestick_series(
    label: str,
    frame: pd.DataFrame,
    *,
    highlights: tuple[ChartHighlight | None, ...] | list[ChartHighlight | None] = (),
) -> CandlestickSeries:
    required = {"date", "open", "high", "low", "close"}
    if frame.empty or not required.issubset(frame.columns):
        return CandlestickSeries(label=label, rows=())
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    rows = tuple(
        CandlestickBar(
            date=pd.Timestamp(row["date"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        )
        for _, row in prepared.iterrows()
    )
    return CandlestickSeries(
        label=label,
        rows=rows,
        highlights=tuple(highlight for highlight in highlights if highlight is not None),
    )


def _window_highlight(frame: pd.DataFrame, label: str) -> ChartHighlight | None:
    if frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return ChartHighlight(
        label=label,
        start=pd.Timestamp(dates.min()),
        end=pd.Timestamp(dates.max()),
        color=WINDOW_HIGHLIGHT_COLOR,
        opacity=WINDOW_HIGHLIGHT_OPACITY,
    )


def _review_segment_highlights(frame: pd.DataFrame) -> tuple[ChartHighlight, ...]:
    if frame.empty or not {"方向", "开始日期", "结束日期"}.issubset(frame.columns):
        return ()
    highlights: list[ChartHighlight] = []
    for _, row in frame.iterrows():
        direction = str(row.get("方向", "") or "").strip()
        start = pd.to_datetime(pd.Series([row.get("开始日期")]), errors="coerce").iloc[0]
        end = pd.to_datetime(pd.Series([row.get("结束日期")]), errors="coerce").iloc[0]
        if not direction or pd.isna(start) or pd.isna(end):
            continue
        highlights.append(
            ChartHighlight(
                label=direction,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                color=SEGMENT_HIGHLIGHT_COLORS.get(direction, "#64748b"),
                opacity=SEGMENT_HIGHLIGHT_OPACITY,
            )
        )
    return tuple(highlights)


def _prepared_symbol_bars(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if bars.empty or "stock_code" not in bars.columns or "date" not in bars.columns:
        return pd.DataFrame()
    normalized = normalize_symbol(symbol)
    prepared = bars.copy()
    prepared["stock_code"] = prepared["stock_code"].map(normalize_symbol)
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    return prepared.loc[prepared["stock_code"] == normalized].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _window_with_forward_bars(source: pd.DataFrame, window: pd.DataFrame, forward_bars: int) -> pd.DataFrame:
    if source.empty or window.empty or "date" not in window.columns:
        return window
    start = pd.Timestamp(window["date"].min())
    end = pd.Timestamp(window["date"].max())
    source_after_start = source.loc[source["date"] >= start].sort_values("date")
    matching = source_after_start.loc[source_after_start["date"] <= end]
    if matching.empty:
        return window
    return source_after_start.head(len(matching) + max(0, int(forward_bars))).reset_index(drop=True)


def _cross_section_symbol_window(result: CrossSectionSearchResult, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    normalized = normalize_symbol(symbol)
    if normalized == result.target_symbol:
        return pd.Timestamp(result.start), pd.Timestamp(result.end)
    if result.results.empty or "symbol" not in result.results.columns:
        return pd.Timestamp(result.start), pd.Timestamp(result.end)
    matched = result.results.loc[result.results["symbol"].map(normalize_symbol) == normalized]
    if matched.empty:
        return pd.Timestamp(result.start), pd.Timestamp(result.end)
    row = matched.iloc[0]
    return pd.Timestamp(row.get("区间开始", result.start)), pd.Timestamp(row.get("区间结束", result.end))


def _dated_label(label: str, window: pd.DataFrame) -> str:
    if window.empty or "date" not in window.columns:
        return label
    dates = pd.to_datetime(window["date"], errors="coerce").dropna()
    if dates.empty:
        return label
    return f"{label}（{dates.min():%Y-%m-%d} 至 {dates.max():%Y-%m-%d}）"


def _stock_label(symbol: str, stock_names: dict[str, str] | None = None, *, is_target: bool = False) -> str:
    normalized = normalize_symbol(symbol)
    name = (stock_names or {}).get(normalized, "").strip()
    label = f"{name}（{normalized}）" if name else normalized
    return f"{label}（目标）" if is_target else label


def _script_profile_ending(row: pd.Series) -> str:
    nature = str(row.get("当前性质", "") or "").strip() or "观察"
    turning_point = str(row.get("关键转折点", "") or "").strip() or "关键位"
    critique = str(row.get("锐评结论", "") or "").strip()
    return f"{nature}，卡在{turning_point}。{critique}".strip()


def _close_line(label: str, frame: pd.DataFrame) -> LineSeries:
    if frame.empty or "close" not in frame.columns:
        return LineSeries(label=label, values=())
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        return LineSeries(label=label, values=())
    first = float(close.iloc[0])
    if first == 0:
        values = tuple(float(value) for value in close.tolist())
    else:
        values = tuple(float(value / first * 100.0) for value in close.tolist())
    return LineSeries(label=label, values=values)


def _percent_text(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not pd.notna(number):
        return "-"
    return f"{number:.2%}"
