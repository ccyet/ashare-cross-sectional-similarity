from __future__ import annotations

import math
import os
from datetime import date
from html import escape
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ashare_cross_section_similarity.cli import _resolve_universe
from ashare_cross_section_similarity.data import (
    import_price_frame,
    inclusive_end_timestamp,
    load_local_bars,
    read_price_data_file,
)
from ashare_cross_section_similarity.downloader import data_check, default_trend_repo, update_local_bars
from ashare_cross_section_similarity.features import normalized_close_path, z_normalize
from ashare_cross_section_similarity.history import HistorySearchConfig, search_history
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    CrossSectionSearchResult,
    search_cross_section,
)
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols


PERCENT_COLUMNS = ["综合相似度", "路径相似度", "特征相似度", "区间收益", "波动率", "最大回撤", "下跌放量占比"]
DECIMAL_COLUMNS = ["路径距离", "趋势斜率", "量价相关", "成交规模", "特征距离"]


def main() -> None:
    st.set_page_config(page_title="A股相似阶段搜集", layout="wide")
    st.title("A股相似阶段搜集")
    st.caption("保留同一标的历史时序相似阶段搜索，并新增同一时间内的横截面相似标的搜索。")
    with st.sidebar:
        st.header("通用设置")
        trend_repo_default = os.environ.get("ASHARE_TREND_REPO", str(default_trend_repo()))
        data_root_default = os.environ.get("ASHARE_DATA_ROOT", str(Path(trend_repo_default) / "data" / "market" / "daily"))
        trend_repo = st.text_input("原 trend-backtest 仓库", value=trend_repo_default)
        data_root = st.text_input("本地行情根目录", value=data_root_default)
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
        _render_price_upload(data_root=data_root, timeframe=timeframe, adjust=adjust)

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
    normalized_target = normalize_symbol(target_symbol)
    if st.session_state.get("cross_quick_message_symbol") != normalized_target:
        st.session_state.pop("cross_quick_message", None)
        st.session_state["cross_quick_message_symbol"] = normalized_target
    start_input = {"key": "cross_start_date"}
    end_input = {"key": "cross_end_date"}
    if "cross_start_date" not in st.session_state:
        start_input["value"] = date(2024, 1, 1)
    if "cross_end_date" not in st.session_state:
        end_input["value"] = date(2024, 3, 31)
    start_date = col2.date_input("区间开始", **start_input)
    end_date = col3.date_input("区间结束", **end_input)
    quick_cols = st.columns(6)
    quick_cols[0].caption("快捷区间")
    for button_col, window_size in zip(quick_cols[1:], [5, 10, 20, 60, 120]):
        button_col.button(
            f"近{window_size}根",
            key=f"cross_quick_{window_size}",
            on_click=_set_cross_quick_window,
            args=(data_root, timeframe, adjust, target_symbol, window_size),
        )
    if st.session_state.get("cross_quick_message"):
        st.info(st.session_state["cross_quick_message"])
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
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
        normalized_targets = unique_symbols([target_symbol])
        normalized_target = normalized_targets[0] if normalized_targets else str(target_symbol).strip().upper()
        target_check = check.loc[check["symbol"] == normalized_target]
        cols = st.columns(4)
        cols[0].metric("搜索范围", f"{len(universe):,}")
        cols[1].metric("可用标的", f"{int((check['status'] == 'available').sum()):,}")
        cols[2].metric("缺文件", f"{int((check['status'] == 'missing_file').sum()):,}")
        cols[3].metric("区间缺失", f"{int((check['status'] == 'missing_window').sum()):,}")
        if not target_check.empty:
            target_row = target_check.iloc[0]
            st.info(
                f"目标标的 {normalized_target}：{target_row['status']}，"
                f"{int(target_row['rows'])} 根，"
                f"{_date_text(target_row['start'])} 至 {_date_text(target_row['end'])}"
            )
        else:
            st.warning(f"目标标的 {normalized_target} 不在本次检查结果中，请确认目标代码输入。")
        st.dataframe(_centered(_format_status(_pin_symbol_row(check, normalized_target, limit=200))), use_container_width=True, hide_index=True)
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
        normalized_targets = unique_symbols([target_symbol])
        if normalized_targets:
            normalized_target = normalized_targets[0]
            st.dataframe(_centered(_format_status(_pin_symbol_row(update_result, normalized_target))), use_container_width=True, hide_index=True)
            target_status = update_result.loc[update_result["symbol"] == normalized_target, "status"]
            if not target_status.empty and target_status.iloc[0] != "available":
                st.warning(f"目标标的 {normalized_target} 下载后仍未覆盖本地行情，请切换下载引擎或检查数据源是否支持该代码。")
        else:
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
    st.markdown("**5. 有效结果计量**")
    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, _cross_section_overview_metrics(result.results)):
        column.metric(label, value)
    summary_col, bucket_col = st.columns(2)
    with summary_col:
        st.caption("后续收益统计")
        st.dataframe(
            _centered(_format_cross_section_stats(_cross_section_forward_summary(result.results))),
            use_container_width=True,
            hide_index=True,
        )
    with bucket_col:
        st.caption("相似度分层表现")
        st.dataframe(
            _centered(_format_cross_section_stats(_cross_section_bucket_summary(result.results))),
            use_container_width=True,
            hide_index=True,
        )
    st.plotly_chart(_cross_section_price_chart(bars, result, top_n=int(top_n)), use_container_width=True)
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


def _render_price_upload(*, data_root: str, timeframe: str, adjust: str) -> None:
    with st.expander("上传自定义价格数据"):
        st.caption("支持 CSV/Parquet；必要列：date、open、high、low、close、symbol 或 stock_code。可选：volume、amount。")
        uploaded_file = st.file_uploader("价格数据文件", type=["csv", "parquet"], key="price_data_upload")
        fallback_symbol = st.text_input("默认代码", value="", help="仅当文件没有 symbol/stock_code 列时填写。")
        if not st.button("导入到本地行情目录", key="price_data_import"):
            return
        if uploaded_file is None:
            st.warning("请先选择 CSV 或 Parquet 文件。")
            return
        try:
            frame = read_price_data_file(uploaded_file, uploaded_file.name)
            result = import_price_frame(
                data_root=Path(data_root),
                timeframe=timeframe,
                adjust=adjust,
                frame=frame,
                fallback_symbol=fallback_symbol,
                source_name=uploaded_file.name,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"价格数据导入失败：{exc}")
            return
        st.cache_data.clear()
        st.dataframe(_centered(_format_import_status(result)), use_container_width=True, hide_index=True)


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


def _load_target_bars_for_quick_window(
    *,
    data_root: str,
    timeframe: str,
    adjust: str,
    target_symbol: str,
) -> pd.DataFrame:
    return load_local_bars(
        data_root=Path(data_root),
        timeframe=timeframe,
        adjust=adjust,
        symbols=[target_symbol],
        start="1900-01-01",
        end=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )


def _set_cross_quick_window(
    data_root: str,
    timeframe: str,
    adjust: str,
    target_symbol: str,
    window_size: int,
) -> None:
    local_target = _load_target_bars_for_quick_window(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        target_symbol=target_symbol,
    )
    quick_start, quick_end, quick_message = _cross_section_quick_window_feedback(
        local_target,
        window_size,
        target_symbol,
    )
    st.session_state["cross_start_date"] = quick_start
    st.session_state["cross_end_date"] = quick_end
    st.session_state["cross_quick_message"] = quick_message
    st.session_state["cross_quick_message_symbol"] = normalize_symbol(target_symbol)


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
    rows: list[dict[str, object]] = []
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
        checked = data_check(
            symbols=[symbol],
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
        )
        if checked.empty:
            checked = result
        elif not result.empty and str(checked["status"].iloc[0]) != "available":
            message = str(checked["message"].iloc[0]) if "message" in checked.columns else ""
            checked.loc[checked.index[0], "message"] = f"下载命令执行后仍缺本地 parquet；{message}".rstrip("；")
        rows.append(checked.iloc[0].to_dict())
        status = str(checked["status"].iloc[0]) if not checked.empty and "status" in checked.columns else "unknown"
        if progress_callback is not None:
            progress_callback(index + 1, total, symbol, status)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "status", "rows", "start", "end", "message"])


def _forward_stats_load_end(end: str | pd.Timestamp, today: pd.Timestamp | None = None) -> str:
    end_ts = pd.Timestamp(end)
    current_day = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    if end_ts >= current_day:
        return end_ts.strftime("%Y-%m-%d")
    return min(end_ts + pd.Timedelta(days=45), current_day).strftime("%Y-%m-%d")


def _cross_section_quick_window(
    bars: pd.DataFrame,
    window_size: int,
    today: pd.Timestamp | None = None,
) -> tuple[date, date]:
    start, end, _selected_count, _total_count = _cross_section_quick_window_selection(bars, window_size, today)
    return start, end


def _cross_section_quick_window_feedback(
    bars: pd.DataFrame,
    window_size: int,
    target_symbol: str,
    today: pd.Timestamp | None = None,
) -> tuple[date, date, str]:
    start, end, selected_count, total_count = _cross_section_quick_window_selection(bars, window_size, today)
    symbol = normalize_symbol(target_symbol)
    window_text = f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d}"
    if total_count == 0:
        return start, end, f"{symbol} 未找到本地行情，已按自然日近 {window_size} 天设置区间：{window_text}。"
    if selected_count < window_size:
        return start, end, f"{symbol} 本地仅有 {selected_count} 根K线，不足近 {window_size} 根；已使用全部可用区间：{window_text}。"
    return start, end, f"{symbol} 已选择近 {window_size} 根K线：{window_text}。"


def _cross_section_quick_window_selection(
    bars: pd.DataFrame,
    window_size: int,
    today: pd.Timestamp | None = None,
) -> tuple[date, date, int, int]:
    if window_size < 1:
        raise ValueError("window_size 至少需要 1。")
    if not bars.empty and "date" in bars.columns:
        dates = pd.to_datetime(bars["date"], errors="coerce").dropna().sort_values().drop_duplicates()
        if not dates.empty:
            selected = dates.tail(window_size)
            return selected.iloc[0].date(), selected.iloc[-1].date(), int(len(selected)), int(len(dates))
    end = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    start = end - pd.Timedelta(days=window_size - 1)
    return start.date(), end.date(), window_size, 0


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
    result = _format_decimal_columns(result, DECIMAL_COLUMNS)
    result = result.rename(columns=rename_map)
    for column in ["区间开始", "区间结束"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result


def _cross_section_overview_metrics(frame: pd.DataFrame) -> list[tuple[str, str]]:
    if frame.empty:
        return [("有效结果", "0"), ("平均相似度", "-"), ("后10根胜率", "-"), ("Top6后10根均值", "-")]
    similarity = pd.to_numeric(frame.get("综合相似度"), errors="coerce")
    returns_10 = pd.to_numeric(frame.get("t_plus_10_return"), errors="coerce")
    return [
        ("有效结果", f"{len(frame):,}"),
        ("平均相似度", _percent_text(similarity.mean())),
        ("后10根胜率", _percent_text((returns_10.dropna() > 0).mean())),
        ("Top6后10根均值", _percent_text(returns_10.head(6).mean())),
    ]


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
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(rows)
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


def _format_cross_section_stats(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    percent_columns = [
        column
        for column in result.columns
        if any(keyword in column for keyword in ("收益", "胜率", "相似度", "波动"))
        and "相关" not in column
        and column != "样本数"
    ]
    result = _format_percent_columns(result, percent_columns)
    for column in result.columns:
        if "相关" in column:
            result[column] = pd.to_numeric(result[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
    return result


def _forward_return_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("t_plus_") and column.endswith("_return")
    ]


def _series_corr(left: pd.Series, right: pd.Series) -> float:
    pairs = pd.concat([left, right], axis=1).dropna()
    if len(pairs) < 2:
        return float("nan")
    corr = float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))
    return corr if pd.notna(corr) else float("nan")


def _percent_text(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(numeric) else f"{numeric:.2%}"


def _pin_symbol_row(frame: pd.DataFrame, symbol: str, limit: int | None = None) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame.head(limit) if limit is not None else frame
    normalized_symbols = unique_symbols([symbol])
    if not normalized_symbols:
        return frame.head(limit) if limit is not None else frame
    normalized_symbol = normalized_symbols[0]
    target = frame.loc[frame["symbol"] == normalized_symbol]
    rest = frame.loc[frame["symbol"] != normalized_symbol]
    result = frame if target.empty else pd.concat([target, rest], ignore_index=True)
    return result.head(limit) if limit is not None else result


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
    result = _format_decimal_columns(result, DECIMAL_COLUMNS)
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


def _format_decimal_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
    return frame


def _format_status(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["start", "end"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result


def _format_import_status(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["start", "end"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result


def _centered(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    return frame.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )


def _cross_section_price_chart(
    bars: pd.DataFrame,
    result: CrossSectionSearchResult,
    top_n: int = 20,
) -> go.Figure:
    fig = go.Figure()
    symbols = unique_symbols([result.target_symbol, *result.results["symbol"].head(top_n).tolist()])
    start = pd.Timestamp(result.start)
    end_marker = pd.Timestamp(result.end).strftime("%Y-%m-%d")
    for symbol in symbols:
        symbol_bars = bars.loc[(bars["stock_code"] == symbol) & (bars["date"] >= start)].sort_values("date")
        if symbol_bars.empty:
            continue
        is_target = symbol == result.target_symbol
        fig.add_scatter(
            x=symbol_bars["date"],
            y=symbol_bars["close"],
            mode="lines",
            name=f"{symbol}（目标）" if is_target else symbol,
            line={"width": 4 if is_target else 1.8},
            opacity=1.0 if is_target else 0.72,
        )
    fig.add_shape(
        type="line",
        x0=end_marker,
        x1=end_marker,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#2563eb", "dash": "dot", "width": 1.5},
    )
    fig.add_annotation(
        x=end_marker,
        y=1,
        xref="x",
        yref="paper",
        text="窗口结束",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"color": "#2563eb", "size": 12},
    )
    fig.update_layout(
        title="目标与Top相似标的收盘价走势",
        xaxis_title="日期",
        yaxis_title="收盘价",
        hovermode="x unified",
    )
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
    forward_bars: int = 10,
) -> list[dict[str, object]]:
    start = result.start
    end = inclusive_end_timestamp(result.end)
    symbols = [result.target_symbol, *result.results["symbol"].head(top_n).tolist()]
    series: list[dict[str, object]] = []
    for symbol in symbols:
        symbol_bars = bars.loc[(bars["stock_code"] == symbol) & (bars["date"] >= start)].sort_values("date")
        if symbol_bars.empty:
            continue
        window = symbol_bars.loc[symbol_bars["date"] <= end]
        if window.empty:
            continue
        chart_window = symbol_bars.head(len(window) + forward_bars)
        window_end_time = pd.Timestamp(window["date"].iloc[-1]).strftime("%Y-%m-%d")
        label = f"{symbol}（目标）" if symbol == result.target_symbol else symbol
        series.append(
            {
                "title": label,
                "windowEndTime": window_end_time,
                "windowSize": int(len(window)),
                "forwardSize": int(max(0, len(chart_window) - len(window))),
                "data": [
                    {
                        "time": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                    for _, row in chart_window.iterrows()
                ],
            }
        )
    return series


def _lightweight_kline_chart_html(series: list[dict[str, object]]) -> str:
    if not series:
        return _kline_empty_message()
    panels = "\n".join(_kline_svg_panel(item) for item in series)
    return f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;width:100%;">
{panels}
</div>
"""


def _kline_empty_message() -> str:
    return """
<div style="padding:12px;border:1px solid #e5e7eb;border-radius:6px;color:#6b7280;font:14px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">
  没有可绘制的K线数据，请检查目标与相似标的在当前区间是否有本地行情。
</div>
"""


def _kline_svg_panel(item: dict[str, object]) -> str:
    rows = _valid_kline_rows(item)
    title = escape(str(item.get("title", "-")))
    window_size = int(item.get("windowSize") or 0)
    forward_size = int(item.get("forwardSize") or 0)
    if not rows:
        return f"""
<div data-kline-panel="1" style="border:1px solid #e5e7eb;border-radius:6px;padding:8px;background:#ffffff;">
  <div style="font:600 13px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin-bottom:6px;color:#111827;">{title}</div>
  <div style="height:220px;display:flex;align-items:center;justify-content:center;color:#6b7280;font:13px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">没有可绘制的K线数据</div>
</div>
"""

    width = 360.0
    height = 240.0
    left = 42.0
    right = 12.0
    top = 14.0
    bottom = 38.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    low = min(row["low"] for row in rows)
    high = max(row["high"] for row in rows)
    if high == low:
        padding = max(abs(high) * 0.01, 1.0)
        high += padding
        low -= padding

    def x_position(index: int) -> float:
        if len(rows) == 1:
            return left + plot_width / 2
        return left + plot_width * index / (len(rows) - 1)

    def y_position(price: float) -> float:
        return top + (high - price) / (high - low) * plot_height

    elements = [
        f'<rect x="0" y="0" width="{width:g}" height="{height:g}" fill="#ffffff"/>',
    ]
    for step in range(5):
        y = top + plot_height * step / 4
        elements.append(f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{width - right:.2f}" y2="{y:.2f}" stroke="#eef2f7" stroke-width="1"/>')
    elements.extend(
        [
            f'<text x="4" y="{top + 4:.2f}" fill="#6b7280" font-size="10">{high:.2f}</text>',
            f'<text x="4" y="{top + plot_height:.2f}" fill="#6b7280" font-size="10">{low:.2f}</text>',
            f'<text x="{left:.2f}" y="{height - 14:.2f}" fill="#6b7280" font-size="10">{escape(rows[0]["time"][5:])}</text>',
            f'<text x="{width - right:.2f}" y="{height - 14:.2f}" text-anchor="end" fill="#6b7280" font-size="10">{escape(rows[-1]["time"][5:])}</text>',
        ]
    )

    divider_index = _window_end_index(rows, str(item.get("windowEndTime") or ""))
    candle_width = min(10.0, max(3.0, plot_width / max(1, len(rows)) * 0.55))
    if divider_index is not None:
        divider_x = x_position(divider_index)
        if forward_size > 0:
            shade_x = min(width - right, divider_x + candle_width / 2)
            shade_width = max(0.0, width - right - shade_x)
            elements.append(
                f'<rect class="forwardShade" x="{shade_x:.2f}" y="{top:.2f}" width="{shade_width:.2f}" height="{plot_height:.2f}" fill="#2563eb" opacity="0.08"/>'
            )
        elements.append(
            f'<line class="positionWindowDivider" x1="{divider_x:.2f}" y1="{top:.2f}" x2="{divider_x:.2f}" y2="{top + plot_height:.2f}" stroke="#2563eb" stroke-width="1.5">'
            "<title>窗口结束</title></line>"
        )

    for index, row in enumerate(rows):
        x = x_position(index)
        color = "#d62728" if row["close"] >= row["open"] else "#2ca02c"
        high_y = y_position(row["high"])
        low_y = y_position(row["low"])
        open_y = y_position(row["open"])
        close_y = y_position(row["close"])
        body_top = min(open_y, close_y)
        body_height = max(abs(open_y - close_y), 1.2)
        elements.append(
            f'<line data-kline-candle="1" x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1.2"/>'
        )
        elements.append(
            f'<rect data-kline-candle="1" x="{x - candle_width / 2:.2f}" y="{body_top:.2f}" width="{candle_width:.2f}" height="{body_height:.2f}" fill="{color}" opacity="0.9"/>'
        )

    svg = "\n".join(elements)
    return f"""
<div data-kline-panel="1" style="border:1px solid #e5e7eb;border-radius:6px;padding:8px;background:#ffffff;">
  <div style="font:600 13px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin-bottom:6px;color:#111827;">{title}</div>
  <svg viewBox="0 0 360 240" role="img" aria-label="{title} K线图" style="width:100%;height:240px;display:block;">{svg}</svg>
  <div style="font:12px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#4b5563;margin-top:4px;">窗口内 {window_size} 根 | 后续 {forward_size} 根</div>
</div>
"""


def _valid_kline_rows(item: dict[str, object]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    raw_rows = item.get("data")
    if not isinstance(raw_rows, list):
        return rows
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        try:
            row = {
                "time": str(raw["time"]),
                "open": float(raw["open"]),
                "high": float(raw["high"]),
                "low": float(raw["low"]),
                "close": float(raw["close"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(float(row[column])) for column in ("open", "high", "low", "close")):
            rows.append(row)
    return rows


def _window_end_index(rows: list[dict[str, float | str]], window_end_time: str) -> int | None:
    if not window_end_time:
        return None
    for index, row in enumerate(rows):
        if row["time"] == window_end_time:
            return index
    return None


if __name__ == "__main__":
    main()
