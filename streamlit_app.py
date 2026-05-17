from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ashare_cross_section_similarity.cli import _resolve_universe
from ashare_cross_section_similarity.data import inclusive_end_timestamp, load_local_bars
from ashare_cross_section_similarity.downloader import data_check, default_trend_repo, update_local_bars
from ashare_cross_section_similarity.features import normalized_close_path, z_normalize
from ashare_cross_section_similarity.history import HistorySearchConfig, search_history
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    CrossSectionSearchResult,
    search_cross_section,
)
from ashare_cross_section_similarity.universe import unique_symbols


PERCENT_COLUMNS = ["综合相似度", "路径相似度", "特征相似度", "区间收益", "波动率", "最大回撤"]


def main() -> None:
    st.set_page_config(page_title="A股相似阶段搜集", layout="wide")
    st.title("A股相似阶段搜集")
    st.caption("保留同一标的历史时序相似阶段搜索，并新增同一时间内的横截面相似标的搜索。")
    with st.sidebar:
        st.header("通用设置")
        trend_repo = st.text_input("原 trend-backtest 仓库", value=str(default_trend_repo()))
        data_root = st.text_input("本地行情根目录", value=str(Path(trend_repo) / "data" / "market" / "daily"))
        timeframe = st.selectbox("周期", ["1d", "30m", "15m", "5m", "1m"], index=0)
        adjust = st.text_input("复权", value="qfq")
        download_engine = st.selectbox(
            "下载引擎",
            ["trend", "openbb"],
            format_func=lambda value: "trend-backtest" if value == "trend" else "OpenBB",
        )
        provider_default = "akshare" if download_engine == "openbb" else ""
        provider = st.text_input(
            "下载源",
            value=provider_default,
            help="trend 可留空使用原配置；OpenBB 默认 akshare，需安装 openbb_akshare。",
        )

    history_tab, cross_section_tab = st.tabs(["历史时序相似", "横截面相似"])
    with history_tab:
        _render_history_tab(
            trend_repo=trend_repo,
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            provider=provider,
            download_engine=download_engine,
        )
    with cross_section_tab:
        _render_cross_section_tab(
            trend_repo=trend_repo,
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            provider=provider,
            download_engine=download_engine,
        )


def _render_history_tab(
    *,
    trend_repo: str,
    data_root: str,
    timeframe: str,
    adjust: str,
    provider: str,
    download_engine: str,
) -> None:
    st.subheader("同一标的历史时序相似")
    st.caption("选定一个标的和当前窗口结束日，系统只在这个标的自己的历史里找相似阶段。")
    col1, col2, col3, col4 = st.columns(4)
    symbol = col1.text_input("目标代码", value="399006.SZ", key="history_symbol")
    as_of = col2.text_input("当前窗口结束", value="2024-03-31", key="history_as_of")
    window_size = col3.selectbox("主走势窗口", [5, 10, 20, 60, 120], index=2, key="history_window_size")
    top_n = col4.number_input("展示数量", min_value=1, max_value=50, value=10, step=1, key="history_top_n")

    col5, col6, col7, col8 = st.columns(4)
    forward_windows = col5.text_input("后验观察窗口", value="5,20,60", key="history_forward_windows")
    candidate_n = col6.number_input("初筛候选", min_value=10, max_value=1000, value=100, step=10, key="history_candidate_n")
    exclusion_bars = col7.number_input("排除近邻K线", min_value=0, max_value=500, value=20, step=5, key="history_exclusion_bars")
    nearby_gap_days = col8.number_input("样本间隔天数", min_value=0, max_value=365, value=20, step=5, key="history_gap_days")
    path_weight = st.slider("走势权重", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="history_path_weight")

    st.markdown("**1. 数据检查**")
    bars = load_local_bars(
        data_root=Path(data_root),
        timeframe=timeframe,
        adjust=adjust,
        symbols=[symbol],
        start="1900-01-01",
        end=as_of,
    )
    if bars.empty:
        st.error("未找到该标的在 as-of 之前的本地行情。请先下载或检查代码、周期、复权目录。")
    else:
        cols = st.columns(4)
        cols[0].metric("可用K线", f"{len(bars):,}")
        cols[1].metric("最早日期", _date_text(bars["date"].min()))
        cols[2].metric("最近日期", _date_text(bars["date"].max()))
        cols[3].metric("窗口要求", f"{int(window_size)} 根")
        if len(bars) < int(window_size):
            st.warning("as-of 之前 K 线数量不足，无法形成当前窗口。")

    with st.expander("缺数据时下载或更新"):
        download_start = st.text_input("下载开始", value="2018-01-01", key="history_download_start")
        if st.button("下载或更新该标的行情", key="history_download"):
            with st.spinner("正在调用原 trend-backtest 更新行情..."):
                update_result = update_local_bars(
                    symbols=[symbol],
                    timeframe=timeframe,
                    adjust=adjust,
                    start=download_start,
                    end=as_of,
                    trend_repo=Path(trend_repo),
                    data_root=Path(data_root),
                    provider=provider,
                    download_engine=download_engine,
                )
            st.dataframe(_centered(_format_status(update_result)), use_container_width=True, hide_index=True)

    st.markdown("**2. 运行历史搜索**")
    if not st.button("运行历史时序搜索", type="primary", key="history_run"):
        st.info("确认上方有足够 K 线后，点击运行历史时序搜索。")
        return

    try:
        horizons = _parse_horizons(forward_windows)
        result = search_history(
            bars,
            HistorySearchConfig(
                symbol=symbol,
                as_of=as_of,
                window_size=int(window_size),
                forward_windows=tuple(horizons),
                candidate_n=int(candidate_n),
                top_n=int(top_n),
                exclusion_bars=int(exclusion_bars),
                nearby_gap_days=int(nearby_gap_days),
                path_weight=float(path_weight),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    st.markdown("**3. 搜索结果**")
    cols = st.columns(4)
    cols[0].metric("当前窗口", f"{_date_text(result.current_window['date'].min())} 至 {_date_text(result.current_window['date'].max())}")
    cols[1].metric("有效样本", len(result.results))
    cols[2].metric("周期", timeframe)
    cols[3].metric("目标代码", result.symbol)
    if result.results.empty:
        st.warning("没有找到可用历史样本。请缩短窗口、放宽排除近邻K线，或补充更长历史数据。")
        return

    st.dataframe(_centered(_format_history_results(result.results)), use_container_width=True, hide_index=True)
    st.plotly_chart(_history_path_chart(result.current_window, result.historical_windows), use_container_width=True)
    st.download_button(
        "下载历史搜索 CSV",
        data=result.results.to_csv(index=False).encode("utf-8-sig"),
        file_name="history_similarity.csv",
        mime="text/csv",
    )


def _render_cross_section_tab(
    *,
    trend_repo: str,
    data_root: str,
    timeframe: str,
    adjust: str,
    provider: str,
    download_engine: str,
) -> None:
    st.subheader("同一时间横截面相似")
    st.caption("选定某个标的一段区间走势，在同一段时间里从指定范围内寻找其他相似标的。")
    col1, col2, col3 = st.columns(3)
    target_symbol = col1.text_input("目标代码", value="300750.SZ", key="cross_target_symbol")
    start = col2.text_input("区间开始", value="2024-01-01", key="cross_start")
    end = col3.text_input("区间结束", value="2024-03-31", key="cross_end")
    col4, col5, col6 = st.columns(3)
    top_n = col4.number_input("展示数量", min_value=5, max_value=100, value=20, step=5, key="cross_top_n")
    min_coverage = col5.slider("最小覆盖率", min_value=0.5, max_value=1.0, value=0.8, step=0.05, key="cross_min_coverage")
    path_weight = col6.slider("走势权重", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="cross_path_weight")
    universe_symbols = st.text_area("搜索范围代码", value="", help="逗号分隔；留空时尝试读取本地目录下全部 parquet。")
    col7, col8, col9 = st.columns(3)
    universe_file = col7.text_input("搜索范围文件", value="")
    universe_index = col8.text_input("指数成分", value="", help="如 000300，需要 akshare。")
    universe_industry = col9.text_input("行业板块", value="", help="如 半导体，需要 akshare。")
    universe_concept = st.text_input("概念板块", value="", help="如 融资融券，需要 akshare。")

    try:
        universe = _cached_resolve_universe(
            data_root,
            timeframe,
            adjust,
            universe_symbols,
            universe_file,
            universe_index,
            universe_industry,
            universe_concept,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"搜索范围解析失败：{exc}")
        return
    symbols = [target_symbol, *universe]
    st.markdown("**1. 数据检查**")
    st.caption("点击后检查目标和搜索范围在所选区间内是否已有本地行情；缺数据时可直接在本页下载。")
    check_key = (tuple(symbols), data_root, timeframe, adjust, start, end)
    if st.button("检查本地数据覆盖", key="cross_check"):
        try:
            st.session_state["cross_data_check_key"] = check_key
            st.session_state["cross_data_check"] = _cached_data_check(
                tuple(symbols),
                data_root,
                timeframe,
                adjust,
                start,
                end,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"数据检查失败：{exc}")
            return
    check = st.session_state.get("cross_data_check")
    if check is not None and st.session_state.get("cross_data_check_key") == check_key:
        cols = st.columns(4)
        cols[0].metric("搜索范围", f"{len(universe):,}")
        cols[1].metric("可用标的", f"{int((check['status'] == 'available').sum()):,}")
        cols[2].metric("缺文件", f"{int((check['status'] == 'missing_file').sum()):,}")
        cols[3].metric("区间缺失", f"{int((check['status'] == 'missing_window').sum()):,}")
        st.dataframe(_centered(_format_status(check.head(200))), use_container_width=True, hide_index=True)
    else:
        st.info(f"当前搜索范围 {len(universe):,} 个标的。需要覆盖明细时点击检查。")

    st.markdown("**2. 数据抓取 / 更新**")
    st.caption("默认委托 trend-backtest；也可用 OpenBB/AKShare 直接写入本地 parquet。")
    if st.button("下载或更新当前目标与搜索范围行情", key="cross_download"):
        download_end = _forward_stats_load_end(end)
        progress_bar = st.progress(0.0, text=f"准备下载 {len(unique_symbols(symbols)):,} 个标的")
        progress_text = st.empty()

        def report_progress(completed: int, total: int, symbol: str, status: str) -> None:
            ratio = completed / total if total else 1.0
            action = "正在下载" if status == "running" else "已完成"
            progress_bar.progress(ratio, text=f"{completed}/{total} {action} {symbol}")
            progress_text.caption(f"当前标的：{symbol}；状态：{status}")

        update_result = _download_symbols_with_progress(
            symbols=symbols,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=download_end,
            trend_repo=Path(trend_repo),
            data_root=Path(data_root),
            provider=provider,
            download_engine=download_engine,
            progress_callback=report_progress,
        )
        progress_bar.progress(1.0, text="下载任务已完成")
        progress_text.caption(f"下载截止：{download_end}，用于覆盖窗口后 3/5/10 根收益统计。")
        st.cache_data.clear()
        st.session_state.pop("cross_data_check", None)
        st.session_state.pop("cross_data_check_key", None)
        st.dataframe(_centered(_format_status(update_result)), use_container_width=True, hide_index=True)

    st.markdown("**3. 运行横截面搜索**")
    if not st.button("运行横截面搜索", type="primary", key="cross_run"):
        st.info("检查数据后，缺失则先下载；数据可用后点击运行横截面搜索。")
        return

    try:
        bars = _cached_load_local_bars(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=tuple(symbols),
            start=start,
            end=_forward_stats_load_end(end),
        )
        result = search_cross_section(
            bars,
            CrossSectionSearchConfig(
                target_symbol=target_symbol,
                universe_symbols=tuple(universe),
                start=start,
                end=end,
                top_n=int(top_n),
                min_coverage=float(min_coverage),
                path_weight=float(path_weight),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    st.markdown("**4. 搜索结果**")
    st.metric("目标窗口 K 线数", result.window_size)
    st.metric("有效结果数", len(result.results))
    if result.results.empty:
        st.warning("没有找到可用结果。请检查本地数据覆盖、搜索范围和区间设置。")
        return

    st.dataframe(_centered(_format_results(result.results)), use_container_width=True, hide_index=True)
    st.plotly_chart(_score_chart(result.results), use_container_width=True)
    components.html(
        _lightweight_kline_chart_html(_lightweight_kline_series(bars, result)),
        height=820,
    )
    if not result.skipped.empty:
        with st.expander("查看跳过的标的"):
            st.dataframe(_centered(result.skipped), use_container_width=True, hide_index=True)

    st.download_button(
        "下载横截面 CSV",
        data=result.results.to_csv(index=False).encode("utf-8-sig"),
        file_name="cross_section_similarity.csv",
        mime="text/csv",
    )


class _Args:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


@st.cache_data(show_spinner=False)
def _cached_resolve_universe(
    data_root: str,
    timeframe: str,
    adjust: str,
    universe_symbols: str,
    universe_file: str,
    universe_index: str,
    universe_industry: str,
    universe_concept: str,
) -> list[str]:
    return _resolve_universe(
        _Args(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            universe_symbols=universe_symbols,
            universe_file=universe_file,
            universe_index=universe_index,
            universe_industry=universe_industry,
            universe_concept=universe_concept,
        )
    )


@st.cache_data(show_spinner=False)
def _cached_data_check(
    symbols: tuple[str, ...],
    data_root: str,
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    return data_check(
        symbols=symbols,
        data_root=Path(data_root),
        timeframe=timeframe,
        adjust=adjust,
        start=start,
        end=end,
    )


@st.cache_data(show_spinner=False)
def _cached_load_local_bars(
    *,
    data_root: str,
    timeframe: str,
    adjust: str,
    symbols: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    return load_local_bars(
        data_root=Path(data_root),
        timeframe=timeframe,
        adjust=adjust,
        symbols=symbols,
        start=start,
        end=end,
    )


def _download_symbols_with_progress(
    *,
    symbols: list[str] | tuple[str, ...],
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
    trend_repo: Path,
    data_root: Path,
    provider: str,
    download_engine: str,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> pd.DataFrame:
    normalized = unique_symbols(symbols)
    if not normalized:
        return pd.DataFrame(columns=["symbol", "status", "rows", "new_rows", "message"])
    rows: list[pd.DataFrame] = []
    total = len(normalized)
    for index, symbol in enumerate(normalized):
        if progress_callback is not None:
            progress_callback(index, total, symbol, "running")
        result = update_local_bars(
            symbols=[symbol],
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
            trend_repo=trend_repo,
            data_root=data_root,
            provider=provider,
            download_engine=download_engine,
        )
        rows.append(result)
        status = str(result["status"].iloc[0]) if not result.empty and "status" in result.columns else "unknown"
        if progress_callback is not None:
            progress_callback(index + 1, total, symbol, status)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["symbol", "status", "rows", "new_rows", "message"])


def _forward_stats_load_end(end: str | pd.Timestamp, today: pd.Timestamp | None = None) -> str:
    end_ts = pd.Timestamp(end)
    current_day = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    if end_ts >= current_day:
        return end_ts.strftime("%Y-%m-%d")
    return min(end_ts + pd.Timedelta(days=45), current_day).strftime("%Y-%m-%d")


def _format_results(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map: dict[str, str] = {}
    percent_columns = [*PERCENT_COLUMNS]
    for column in result.columns:
        if column.startswith("t_plus_") and column.endswith("_return"):
            horizon = column.removeprefix("t_plus_").removesuffix("_return")
            rename_map[column] = f"后{horizon}根收益"
            percent_columns.append(column)
    result = _format_percent_columns(result, percent_columns)
    result = result.rename(columns=rename_map)
    for column in ["区间开始", "区间结束"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result



def _format_history_results(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map: dict[str, str] = {}
    percent_columns = [*PERCENT_COLUMNS]
    for column in result.columns:
        if column.startswith("t_plus_") and column.endswith("_return"):
            horizon = column.removeprefix("t_plus_").removesuffix("_return")
            rename_map[column] = f"后{horizon}根收益"
            percent_columns.append(column)
        elif column.startswith("t_plus_") and column.endswith("_max_drawdown"):
            horizon = column.removeprefix("t_plus_").removesuffix("_max_drawdown")
            rename_map[column] = f"后{horizon}根最大回撤"
            percent_columns.append(column)
        elif column.startswith("t_plus_") and column.endswith("_max_favorable"):
            horizon = column.removeprefix("t_plus_").removesuffix("_max_favorable")
            rename_map[column] = f"后{horizon}根最大浮盈"
            percent_columns.append(column)
    result = _format_percent_columns(result, percent_columns)
    result = result.rename(columns=rename_map)
    for column in ["窗口开始", "窗口结束"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result


def _format_percent_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
    return frame


def _format_status(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["start", "end"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result


def _centered(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    return frame.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )


def _score_chart(frame: pd.DataFrame) -> go.Figure:
    top = frame.head(20)
    fig = go.Figure()
    fig.add_bar(x=top["symbol"], y=top["综合相似度"], name="综合相似度")
    fig.add_bar(x=top["symbol"], y=top["路径相似度"], name="路径相似度")
    fig.update_layout(title="Top 相似标的", yaxis_tickformat=".0%", barmode="group")
    return fig


def _history_path_chart(current_window: pd.DataFrame, historical_windows: list[pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    current_path = z_normalize(normalized_close_path(current_window))
    x_values = list(range(1, len(current_path) + 1))
    fig.add_scatter(x=x_values, y=current_path, mode="lines", name="当前窗口", line={"width": 4})
    for index, window in enumerate(historical_windows[:6], start=1):
        path = z_normalize(normalized_close_path(window))
        label = f"样本{index}: {_date_text(window['date'].min())} 至 {_date_text(window['date'].max())}"
        fig.add_scatter(x=x_values, y=path, mode="lines", name=label, opacity=0.65)
    fig.update_layout(title="当前窗口 vs 历史相似窗口", xaxis_title="窗口内第 N 根K线", yaxis_title="标准化路径")
    return fig


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("后验观察窗口必须是逗号分隔的正整数。")
    return horizons


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _lightweight_kline_series(
    bars: pd.DataFrame,
    result: CrossSectionSearchResult,
    top_n: int = 6,
) -> list[dict[str, object]]:
    start = result.start
    end = inclusive_end_timestamp(result.end)
    symbols = [result.target_symbol, *result.results["symbol"].head(top_n).tolist()]
    series: list[dict[str, object]] = []
    for symbol in symbols:
        window = bars.loc[
            (bars["stock_code"] == symbol)
            & (bars["date"] >= start)
            & (bars["date"] <= end)
        ].sort_values("date")
        if window.empty:
            continue
        label = f"{symbol}（目标）" if symbol == result.target_symbol else symbol
        series.append(
            {
                "title": label,
                "data": [
                    {
                        "time": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                    for _, row in window.iterrows()
                ],
            }
        )
    return series


def _lightweight_kline_chart_html(series: list[dict[str, object]]) -> str:
    series_json = json.dumps(series, ensure_ascii=False)
    return f"""
<div id="kline-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;width:100%;"></div>
<div style="font-size:12px;color:#666;margin-top:6px;">
  Powered by <a href="https://www.tradingview.com/" target="_blank">TradingView</a> Lightweight Charts
</div>
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<script>
const grid = document.getElementById('kline-grid');
const chartItems = {series_json};
const charts = [];
chartItems.forEach((item, index) => {{
  const panel = document.createElement('div');
  panel.style.border = '1px solid #e5e7eb';
  panel.style.borderRadius = '6px';
  panel.style.padding = '8px';
  panel.style.background = '#ffffff';
  const title = document.createElement('div');
  title.textContent = item.title;
  title.style.font = '600 13px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';
  title.style.marginBottom = '6px';
  const container = document.createElement('div');
  container.style.height = '220px';
  panel.appendChild(title);
  panel.appendChild(container);
  grid.appendChild(panel);
  const chart = LightweightCharts.createChart(container, {{
    layout: {{ background: {{ type: 'solid', color: '#ffffff' }}, textColor: '#1f2937' }},
    grid: {{ vertLines: {{ color: '#f3f4f6' }}, horzLines: {{ color: '#f3f4f6' }} }},
    rightPriceScale: {{ borderColor: '#d1d5db' }},
    timeScale: {{ borderColor: '#d1d5db' }},
    width: container.clientWidth,
    height: 220
  }});
  const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {{
    upColor: '#d62728',
    downColor: '#2ca02c',
    borderUpColor: '#d62728',
    borderDownColor: '#2ca02c',
    wickUpColor: '#d62728',
    wickDownColor: '#2ca02c'
  }});
  candles.setData(item.data);
  chart.timeScale().fitContent();
  charts.push({{ chart, container }});
}});
window.addEventListener('resize', () => {{
  charts.forEach((item) => item.chart.applyOptions({{ width: item.container.clientWidth }}));
}});
</script>
"""


if __name__ == "__main__":
    main()
