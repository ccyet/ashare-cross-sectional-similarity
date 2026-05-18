from __future__ import annotations

import math
import os
from datetime import date
from fnmatch import fnmatch
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
    resolve_timeframe_root,
)
from ashare_cross_section_similarity.downloader import data_check, default_trend_repo, update_local_bars
from ashare_cross_section_similarity.features import normalized_close_path, z_normalize
from ashare_cross_section_similarity.history import HistorySearchConfig, HistorySearchResult, search_history
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    CrossSectionSearchResult,
    search_cross_section,
)
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols


PERCENT_COLUMNS = ["综合相似度", "路径相似度", "特征相似度", "区间收益", "波动率", "最大回撤", "下跌放量占比", "覆盖率"]
DECIMAL_COLUMNS = ["路径距离", "趋势斜率", "量价相关", "成交规模", "特征距离"]
DOWNLOAD_REQUIRED_STATUSES = {"missing_file", "missing_window", "partial_window", "read_error"}
UNIVERSE_FILE_TYPES = [("搜索范围文件", ("*.csv", "*.xlsx", "*.xls", "*.parquet")), ("所有文件", "*")]
SIZE_SPREAD_START = "2016-01-01"
SIZE_SPREAD_SMALL_SYMBOL = "000852.SH"
SIZE_SPREAD_LARGE_SYMBOL = "000300.SH"
SIZE_SPREAD_SYMBOLS = (SIZE_SPREAD_SMALL_SYMBOL, SIZE_SPREAD_LARGE_SYMBOL)


def main() -> None:
    st.set_page_config(page_title="A股相似阶段搜集", layout="wide")
    st.title("A股相似阶段搜集")
    st.caption("保留同一标的历史时序相似阶段搜索，并新增同一时间内的横截面相似标的搜索。")
    with st.sidebar:
        st.header("通用设置")
        trend_repo_default = os.environ.get("ASHARE_TREND_REPO", str(default_trend_repo()))
        trend_repo = _render_directory_picker(
            "原 trend-backtest 仓库",
            trend_repo_default,
            "trend_repo",
        )
        data_root_default = os.environ.get("ASHARE_DATA_ROOT", str(Path(trend_repo) / "data" / "market" / "daily"))
        data_root = _render_directory_picker(
            "本地行情根目录",
            data_root_default,
            "data_root",
        )
        timeframe = st.selectbox("周期", ["1d", "30m", "15m", "5m", "1m"], index=0)
        adjust = st.text_input("复权", value="qfq")
        download_engine = st.selectbox(
            "下载引擎",
            ["trend", "openbb", "tdx"],
            format_func=lambda value: {
                "trend": "trend-backtest",
                "openbb": "OpenBB",
                "tdx": "TDX 本地",
            }[value],
        )
        if download_engine == "tdx":
            provider = _render_directory_picker(
                "通达信 PYPlugins/user 目录",
                os.environ.get("TDX_TQCENTER_PATH", ""),
                "tdx_tqcenter",
            )
            st.caption("可选择通达信安装目录、PYPlugins 或 PYPlugins/user；留空则尝试系统导入路径。")
        else:
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


def _render_directory_picker(label: str, default_path: str | Path, key: str) -> str:
    state_key = f"{key}_path"
    default_key = f"{key}_default_path"
    open_key = f"{key}_browser_open"
    default_text = _path_text(default_path)
    previous_default = st.session_state.get(default_key)
    if state_key not in st.session_state or st.session_state.get(state_key) == previous_default:
        st.session_state[state_key] = default_text
    st.session_state[default_key] = default_text
    st.caption(label)
    _render_selected_path(str(st.session_state[state_key]))
    if st.button(f"选择{label}", key=f"{key}_pick"):
        st.session_state[open_key] = not bool(st.session_state.get(open_key))
    if st.session_state.get(open_key):
        with st.container(border=True):
            _render_directory_browser(label, key, state_key, st.session_state[state_key])
    return str(st.session_state[state_key])


def _render_file_picker(
    label: str,
    initial_path: str | Path,
    key: str,
    filetypes: list[tuple[str, str | tuple[str, ...]]],
) -> str:
    state_key = f"{key}_path"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""
    st.caption(label)
    _render_selected_path(str(st.session_state[state_key]) if st.session_state[state_key] else "未选择")
    open_key = f"{key}_browser_open"
    button_col, clear_col = st.columns([2, 1])
    if button_col.button(f"选择{label}", key=f"{key}_pick"):
        st.session_state[open_key] = not bool(st.session_state.get(open_key))
    if st.session_state[state_key] and clear_col.button("清除", key=f"{key}_clear"):
        st.session_state[state_key] = ""
        st.session_state[open_key] = False
    if st.session_state.get(open_key):
        with st.container(border=True):
            _render_file_browser(label, key, state_key, st.session_state[state_key] or initial_path, filetypes)
    return str(st.session_state[state_key])


def _render_directory_browser(label: str, key: str, state_key: str, current_path: str | Path) -> None:
    browser_dir_key = f"{key}_browser_dir"
    if browser_dir_key not in st.session_state:
        st.session_state[browser_dir_key] = _picker_initial_directory(current_path, Path.home())
    current_dir = Path(str(st.session_state[browser_dir_key])).expanduser()
    if not current_dir.exists() or not current_dir.is_dir():
        current_dir = Path(_picker_initial_directory(current_path, Path.home()))
        st.session_state[browser_dir_key] = str(current_dir)
    st.caption(f"当前目录：{current_dir}")
    nav_col, select_col = st.columns(2)
    if nav_col.button("上一级", key=f"{key}_parent", disabled=current_dir.parent == current_dir):
        st.session_state[browser_dir_key] = str(current_dir.parent)
        st.rerun()
    if select_col.button("选择当前目录", key=f"{key}_select_current"):
        st.session_state[state_key] = str(current_dir)
        st.session_state[f"{key}_browser_open"] = False
        st.rerun()
    entries = _directory_picker_entries(current_dir)
    if not entries:
        st.info("当前目录下没有可进入的子文件夹。")
        return
    selected = st.selectbox(
        "子文件夹",
        [str(path) for path in entries],
        key=f"{key}_dir_choice",
        format_func=lambda value: Path(value).name,
    )
    if st.button("进入子文件夹", key=f"{key}_enter_dir"):
        st.session_state[browser_dir_key] = selected
        st.rerun()


def _render_file_browser(
    label: str,
    key: str,
    state_key: str,
    current_path: str | Path,
    filetypes: list[tuple[str, str | tuple[str, ...]]],
) -> None:
    browser_dir_key = f"{key}_browser_dir"
    if browser_dir_key not in st.session_state:
        st.session_state[browser_dir_key] = _picker_initial_directory(current_path, Path.home())
    current_dir = Path(str(st.session_state[browser_dir_key])).expanduser()
    if not current_dir.exists() or not current_dir.is_dir():
        current_dir = Path(_picker_initial_directory(current_path, Path.home()))
        st.session_state[browser_dir_key] = str(current_dir)
    st.caption(f"当前目录：{current_dir}")
    if st.button("上一级", key=f"{key}_file_parent", disabled=current_dir.parent == current_dir):
        st.session_state[browser_dir_key] = str(current_dir.parent)
        st.rerun()
    directories, files = _file_picker_entries(current_dir, filetypes)
    if directories:
        selected_dir = st.selectbox(
            "子文件夹",
            [str(path) for path in directories],
            key=f"{key}_file_dir_choice",
            format_func=lambda value: Path(value).name,
        )
        if st.button("进入子文件夹", key=f"{key}_file_enter_dir"):
            st.session_state[browser_dir_key] = selected_dir
            st.rerun()
    else:
        st.info("当前目录下没有可进入的子文件夹。")
    if files:
        selected_file = st.selectbox(
            "文件",
            [str(path) for path in files],
            key=f"{key}_file_choice",
            format_func=lambda value: Path(value).name,
        )
        if st.button(f"选择{label}", key=f"{key}_select_file"):
            st.session_state[state_key] = selected_file
            st.session_state[f"{key}_browser_open"] = False
            st.rerun()
    else:
        st.info("当前目录下没有符合类型的文件。")


def _render_selected_path(path: str) -> None:
    st.markdown(
        f"<div style='font-size:12px;line-height:1.35;word-break:break-all;color:#374151;margin:-0.25rem 0 0.35rem 0;'>{escape(path)}</div>",
        unsafe_allow_html=True,
    )


def _picker_initial_directory(value: str | Path, fallback: str | Path) -> str:
    paths = [Path(str(value)).expanduser()] if str(value).strip() else []
    paths.append(Path(str(fallback)).expanduser())
    paths.append(Path.home())
    for path in paths:
        candidate = path.parent if path.exists() and path.is_file() else path
        if not candidate.exists() and path.suffix:
            candidate = path.parent
        while True:
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
    return str(Path.home())


def _directory_picker_entries(directory: str | Path) -> list[Path]:
    current_dir = Path(str(directory)).expanduser()
    try:
        entries = [path for path in current_dir.iterdir() if path.is_dir()]
    except OSError:
        return []
    return sorted(entries, key=lambda path: path.name.lower())


def _file_picker_entries(
    directory: str | Path,
    filetypes: list[tuple[str, str | tuple[str, ...]]],
) -> tuple[list[Path], list[Path]]:
    current_dir = Path(str(directory)).expanduser()
    directories = _directory_picker_entries(current_dir)
    patterns = _filetype_patterns(filetypes)
    try:
        files = [path for path in current_dir.iterdir() if path.is_file() and _path_matches_filetypes(path, patterns)]
    except OSError:
        files = []
    return directories, sorted(files, key=lambda path: path.name.lower())


def _filetype_patterns(filetypes: list[tuple[str, str | tuple[str, ...]]]) -> list[str]:
    patterns: list[str] = []
    for _label, raw_patterns in filetypes:
        if isinstance(raw_patterns, str):
            patterns.append(raw_patterns)
        else:
            patterns.extend(raw_patterns)
    return patterns or ["*"]


def _path_matches_filetypes(path: Path, patterns: list[str]) -> bool:
    return "*" in patterns or any(fnmatch(path.name, pattern) for pattern in patterns)


def _path_text(value: str | Path) -> str:
    return str(Path(str(value)).expanduser()) if str(value).strip() else ""


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
    st.caption("选定一个标的和一段自定义区间，系统只在这个标的自己的历史里找相似阶段。")
    col1, col2, col3, col4 = st.columns(4)
    symbol = col1.text_input("目标代码", value="399006.SZ", key="history_symbol")
    start_input = {"key": "history_start_date"}
    end_input = {"key": "history_end_date"}
    if "history_start_date" not in st.session_state:
        start_input["value"] = date(2024, 3, 4)
    if "history_end_date" not in st.session_state:
        end_input["value"] = date(2024, 3, 31)
    start_date = col2.date_input("区间开始", **start_input)
    end_date = col3.date_input("区间结束", **end_input)
    top_n = col4.number_input("展示数量", min_value=1, max_value=50, value=10, step=1, key="history_top_n")
    quick_cols = st.columns(7)
    quick_cols[0].caption("快捷区间")
    quick_cols[1].button(
        "最新收盘",
        key="history_quick_latest",
        on_click=_set_history_latest_end,
        args=(data_root, timeframe, adjust, symbol),
    )
    for button_col, quick_window_size in zip(quick_cols[2:], [5, 10, 20, 60, 120]):
        button_col.button(
            f"近{quick_window_size}根",
            key=f"history_quick_{quick_window_size}",
            on_click=_set_history_quick_window,
            args=(data_root, timeframe, adjust, symbol, quick_window_size),
        )
    if st.session_state.get("history_quick_message"):
        st.info(st.session_state["history_quick_message"])

    col5, col6, col7, col8 = st.columns(4)
    forward_windows = col5.text_input("后验观察窗口", value="5,20,60", key="history_forward_windows")
    candidate_n = col6.number_input("初筛候选", min_value=10, max_value=1000, value=100, step=10, key="history_candidate_n")
    exclusion_bars = col7.number_input("排除近邻K线", min_value=0, max_value=500, value=20, step=5, key="history_exclusion_bars")
    nearby_gap_days = col8.number_input("样本间隔天数", min_value=0, max_value=365, value=20, step=5, key="history_gap_days")
    path_weight = st.slider("走势权重", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="history_path_weight")
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    as_of = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    if error := _date_range_error(start, as_of):
        st.error(error)
        return

    st.markdown("**1. 数据检查**")
    normalized_symbol = normalize_symbol(symbol)
    target_fingerprint = _local_data_fingerprint(data_root, timeframe, adjust, (normalized_symbol,))
    target_check = _cached_data_check(
        (normalized_symbol,),
        data_root,
        timeframe,
        adjust,
        start,
        as_of,
        target_fingerprint,
    )
    bars = _cached_load_local_bars(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=(normalized_symbol,),
        start="1900-01-01",
        end=as_of,
        data_fingerprint=target_fingerprint,
    )
    selected_window = bars.loc[bars["date"].between(pd.Timestamp(start), inclusive_end_timestamp(as_of))] if not bars.empty else bars
    if bars.empty:
        st.error("未找到该标的在区间结束前的本地行情。请先下载或检查代码、周期、复权目录。")
    else:
        target_row = target_check.iloc[0] if not target_check.empty else None
        cols = st.columns(4)
        cols[0].metric("可用K线", f"{len(bars):,}")
        cols[1].metric("本地开始", _date_text(target_row.get("local_start") if target_row is not None else bars["date"].min()))
        cols[2].metric("本地结束", _date_text(target_row.get("local_end") if target_row is not None else bars["date"].max()))
        cols[3].metric("选定区间K线", f"{len(selected_window):,} 根")
        if len(selected_window) < 2:
            st.warning("选定区间内 K 线数量不足，至少需要 2 根。")

    with st.expander("缺数据时下载或更新"):
        download_start = st.text_input("下载开始", value="2018-01-01", key="history_download_start")
        if st.button("下载或更新该标的行情", key="history_download"):
            progress = st.progress(0.0)
            status_text = st.empty()

            def _history_download_progress(done: int, total: int, current_symbol: str, status: str) -> None:
                progress.progress(done / total if total else 1.0)
                status_text.write(f"{done}/{total} {current_symbol}：{status}")

            update_result = _download_symbols_with_progress(
                symbols=[normalized_symbol],
                timeframe=timeframe,
                adjust=adjust,
                start=download_start,
                end=as_of,
                trend_repo=Path(trend_repo),
                data_root=Path(data_root),
                provider=provider,
                download_engine=download_engine,
                progress_callback=_history_download_progress,
            )
            st.cache_data.clear()
            st.dataframe(_centered(_format_data_check_status(update_result)), use_container_width=True, hide_index=True)

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
                window_size=max(2, int(len(selected_window))),
                window_start=start,
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
    st.markdown("**4. 有效样本计量**")
    for column, (label, value) in zip(st.columns(4), _history_overview_metrics(result.results)):
        column.metric(label, value)
    summary_col, bucket_col = st.columns(2)
    with summary_col:
        st.caption("后验观察统计")
        st.dataframe(_centered(_format_history_stats(_history_forward_summary(result.results))), use_container_width=True, hide_index=True)
    with bucket_col:
        st.caption("相似度分层表现")
        st.dataframe(_centered(_format_history_stats(_history_bucket_summary(result.results))), use_container_width=True, hide_index=True)
    st.markdown("**5. 大小盘价差率**")
    st.caption("以 2016-01-01 后首个共同交易日为基准，将中证1000和沪深300分别归一化后相减。")
    size_spread_fingerprint = _local_data_fingerprint(data_root, timeframe, adjust, SIZE_SPREAD_SYMBOLS)
    size_spread_bars = _cached_load_local_bars(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=SIZE_SPREAD_SYMBOLS,
        start=SIZE_SPREAD_START,
        end=_forward_stats_load_end(as_of),
        data_fingerprint=size_spread_fingerprint,
    )
    size_spread = _size_spread_series(size_spread_bars)
    if size_spread.empty:
        st.warning("缺少 000852.SH 或 000300.SH 的本地行情，暂不能计算大小盘价差率。")
    else:
        st.plotly_chart(_size_spread_chart(size_spread, result.current_window, result.historical_windows), use_container_width=True)
        st.dataframe(
            _centered(_format_size_spread_stats(_size_spread_window_stats(size_spread, result.current_window, result.historical_windows))),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**6. K线走势核验**")
    chart_bars = _cached_load_local_bars(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=(result.symbol,),
        start="1900-01-01",
        end=_forward_stats_load_end(as_of),
        data_fingerprint=target_fingerprint,
    )
    st.plotly_chart(_history_path_chart(result.current_window, result.historical_windows), use_container_width=True)
    history_kline_series = _history_kline_series(chart_bars, result, forward_bars=max(horizons) if horizons else 0)
    components.html(
        _lightweight_kline_chart_html(history_kline_series),
        height=_kline_chart_component_height(history_kline_series),
        scrolling=False,
    )
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
    if error := _date_range_error(start, end):
        st.error(error)
        st.session_state.pop("cross_data_check", None)
        st.session_state.pop("cross_data_check_key", None)
        return
    col4, col5, col6, col10 = st.columns(4)
    top_n = col4.number_input("展示数量", min_value=5, max_value=100, value=20, step=5, key="cross_top_n")
    date_tolerance_bars = col5.number_input(
        "日期容错",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
        key="cross_date_tolerance_bars",
        help="避免精确日期带来的误判；系统不扩大目标走势，只允许候选窗口在前后 N 个交易日内平移匹配。",
    )
    min_coverage = col6.slider("最小覆盖率", min_value=0.5, max_value=1.0, value=0.8, step=0.05, key="cross_min_coverage")
    path_weight = col10.slider("走势权重", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="cross_path_weight")
    universe_symbols = st.text_area("搜索范围代码", value="", help="逗号分隔；留空时尝试读取本地目录下全部 parquet。")
    col7, col8, col9 = st.columns(3)
    with col7:
        universe_file = _render_file_picker("搜索范围文件", data_root, "cross_universe_file", UNIVERSE_FILE_TYPES)
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
    symbols_fingerprint = _local_data_fingerprint(data_root, timeframe, adjust, tuple(symbols))
    tolerance_bars = int(date_tolerance_bars)
    coverage_start = _date_tolerance_load_start(start, tolerance_bars)
    coverage_end = _cross_section_load_end(end, tolerance_bars)
    st.markdown("**1. 数据检查**")
    st.caption("点击后检查目标和搜索范围在目标区间、日期容错和后验观察范围内是否已有本地行情；缺数据时可直接在本页下载。")
    check_key = (tuple(symbols), data_root, timeframe, adjust, coverage_start, coverage_end, start, end, tolerance_bars, symbols_fingerprint)
    if st.button("检查本地数据覆盖", key="cross_check"):
        try:
            _cached_data_check.clear()
            st.session_state["cross_data_check_key"] = check_key
            st.session_state["cross_data_check"] = _cached_data_check(
                tuple(symbols),
                data_root,
                timeframe,
                adjust,
                coverage_start,
                coverage_end,
                symbols_fingerprint,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"数据检查失败：{exc}")
            return
    check = st.session_state.get("cross_data_check")
    if check is not None and st.session_state.get("cross_data_check_key") == check_key:
        normalized_targets = unique_symbols([target_symbol])
        normalized_target = normalized_targets[0] if normalized_targets else str(target_symbol).strip().upper()
        target_check = check.loc[check["symbol"] == normalized_target]
        cols = st.columns(5)
        cols[0].metric("搜索范围", f"{len(universe):,}")
        cols[1].metric("完整覆盖", f"{int((check['status'] == 'available').sum()):,}")
        cols[2].metric("覆盖不足", f"{int((check['status'] == 'partial_window').sum()):,}")
        cols[3].metric("缺文件", f"{int((check['status'] == 'missing_file').sum()):,}")
        cols[4].metric("区间无数据", f"{int((check['status'] == 'missing_window').sum()):,}")
        if not target_check.empty:
            target_row = target_check.iloc[0]
            st.info(
                f"目标标的 {normalized_target}：{target_row['status']}，"
                f"{int(target_row['rows'])} 根，"
                f"请求 {_date_text(target_row.get('requested_start'))} 至 {_date_text(target_row.get('requested_end'))}；"
                f"本地 {_date_text(target_row.get('local_start'))} 至 {_date_text(target_row.get('local_end'))}"
            )
        else:
            st.warning(f"目标标的 {normalized_target} 不在本次检查结果中，请确认目标代码输入。")
        st.dataframe(_centered(_format_data_check_status(_pin_symbol_row(check, normalized_target, limit=200))), use_container_width=True, hide_index=True)
    else:
        st.info(f"当前搜索范围 {len(universe):,} 个标的。需要覆盖明细时点击检查。")

    st.markdown("**2. 数据抓取 / 更新**")
    st.caption("先检查覆盖，只补缺文件、覆盖不足、区间无数据或读取失败的标的；按侧栏下载引擎执行。")
    if st.button("检查并下载缺失行情", key="cross_download"):
        download_end = coverage_end
        normalized_targets = unique_symbols([target_symbol])
        normalized_target = normalized_targets[0] if normalized_targets else str(target_symbol).strip().upper()
        check_for_download = check if check is not None and st.session_state.get("cross_data_check_key") == check_key else None
        if check_for_download is None:
            with st.spinner("先检查本地数据覆盖..."):
                check_for_download = _cached_data_check(
                    tuple(symbols),
                    data_root,
                    timeframe,
                    adjust,
                    coverage_start,
                    coverage_end,
                    symbols_fingerprint,
                )
            st.session_state["cross_data_check_key"] = check_key
            st.session_state["cross_data_check"] = check_for_download

        download_symbols = _symbols_requiring_download(check_for_download)
        if not download_symbols:
            st.success("所选区间本地行情已覆盖，无需下载。")
            st.dataframe(
                _centered(_format_data_check_status(_pin_symbol_row(check_for_download, normalized_target))),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(f"本次仅下载缺失标的 {len(download_symbols):,} / {len(unique_symbols(symbols)):,} 个。")
            progress_bar = st.progress(0.0, text=f"准备下载 {len(download_symbols):,} 个缺失标的")
            progress_text = st.empty()

            def report_progress(completed: int, total: int, symbol: str, status: str) -> None:
                ratio = completed / total if total else 1.0
                action = "正在下载" if status == "running" else "已完成"
                progress_bar.progress(ratio, text=f"{completed}/{total} {action} {symbol}")
                progress_text.caption(f"当前标的：{symbol}；状态：{status}")

            update_result = _download_symbols_with_progress(
                symbols=download_symbols,
                timeframe=timeframe,
                adjust=adjust,
                start=coverage_start,
                end=download_end,
                trend_repo=Path(trend_repo),
                data_root=Path(data_root),
                provider=provider,
                download_engine=download_engine,
                progress_callback=report_progress,
            )
            progress_bar.progress(1.0, text="下载任务已完成")
            progress_text.caption(f"下载区间：{coverage_start} 至 {download_end}，用于覆盖日期容错和窗口后 3/5/10 根收益统计。")
            st.cache_data.clear()
            st.session_state.pop("cross_data_check", None)
            st.session_state.pop("cross_data_check_key", None)
            st.dataframe(_centered(_format_data_check_status(_pin_symbol_row(update_result, normalized_target))), use_container_width=True, hide_index=True)
            target_status = update_result.loc[update_result["symbol"] == normalized_target, "status"]
            if not target_status.empty and target_status.iloc[0] != "available":
                st.warning(f"目标标的 {normalized_target} 下载后仍未覆盖本地行情，请切换下载引擎或检查数据源是否支持该代码。")

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
            start=coverage_start,
            end=coverage_end,
            data_fingerprint=symbols_fingerprint,
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
                date_tolerance_bars=tolerance_bars,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    st.markdown("**4. 搜索结果**")
    for column, (label, value) in zip(st.columns(2), _cross_section_result_metrics(result)):
        column.metric(label, value)
    if result.results.empty:
        st.warning("没有找到可用结果。请检查本地数据覆盖、搜索范围和区间设置。")
        return

    stock_names = _cached_stock_name_map(tuple(unique_symbols([result.target_symbol, *result.results["symbol"].astype(str).tolist()])))
    st.dataframe(_centered(_format_results(result.results, stock_names)), use_container_width=True, hide_index=True)
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
    st.plotly_chart(_cross_section_price_chart(bars, result, top_n=int(top_n), stock_names=stock_names), use_container_width=True)
    kline_series = _lightweight_kline_series(bars, result, stock_names=stock_names)
    components.html(
        _lightweight_kline_chart_html(kline_series),
        height=_kline_chart_component_height(kline_series),
        scrolling=False,
    )
    st.markdown("")
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
    data_fingerprint: tuple[tuple[str, int, int], ...],
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
    data_fingerprint: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    return load_local_bars(
        data_root=Path(data_root),
        timeframe=timeframe,
        adjust=adjust,
        symbols=symbols,
        start=start,
        end=end,
    )


def _local_data_fingerprint(
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
) -> tuple[tuple[str, int, int], ...]:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    rows: list[tuple[str, int, int]] = []
    for symbol in unique_symbols(symbols):
        path = root / f"{symbol}.parquet"
        try:
            stat = path.stat()
        except OSError:
            rows.append((symbol, -1, -1))
            continue
        rows.append((symbol, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(rows)


@st.cache_data(show_spinner=False)
def _cached_stock_name_map(symbols: tuple[str, ...]) -> dict[str, str]:
    normalized = tuple(unique_symbols(symbols))
    if not normalized:
        return {}
    try:
        import akshare as ak

        table = ak.stock_info_a_code_name()
    except Exception:  # noqa: BLE001
        return {}
    return _stock_name_map_from_table(table, normalized)


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


def _set_history_quick_window(
    data_root: str,
    timeframe: str,
    adjust: str,
    symbol: str,
    window_size: int,
) -> None:
    bars = _load_target_bars_for_quick_window(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        target_symbol=symbol,
    )
    start, end, message = _history_quick_window_feedback(bars, symbol, window_size)
    st.session_state["history_start_date"] = start
    st.session_state["history_end_date"] = end
    st.session_state["history_quick_message"] = message


def _set_history_latest_end(
    data_root: str,
    timeframe: str,
    adjust: str,
    symbol: str,
) -> None:
    bars = _load_target_bars_for_quick_window(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        target_symbol=symbol,
    )
    normalized = normalize_symbol(symbol)
    dates = pd.to_datetime(bars["date"], errors="coerce").dropna().sort_values() if not bars.empty else pd.Series(dtype="datetime64[ns]")
    if dates.empty:
        st.session_state["history_quick_message"] = f"{normalized} 未找到本地行情，无法设置最新收盘日。"
        return
    end = dates.iloc[-1].date()
    st.session_state["history_end_date"] = end
    st.session_state["history_quick_message"] = f"{normalized} 已将区间结束设为最新本地收盘日：{end:%Y-%m-%d}。"


def _history_quick_window_feedback(
    bars: pd.DataFrame,
    symbol: str,
    window_size: int,
    today: pd.Timestamp | None = None,
) -> tuple[date, date, str]:
    normalized = normalize_symbol(symbol)
    if window_size < 1:
        raise ValueError("window_size 至少需要 1。")
    if bars.empty:
        fallback = (pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()).date()
        return fallback, fallback, f"{normalized} 未找到本地行情，已按当前日期设置区间。"
    dates = pd.to_datetime(bars["date"], errors="coerce").dropna().sort_values()
    if dates.empty:
        fallback = (pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()).date()
        return fallback, fallback, f"{normalized} 未找到有效行情日期，已按当前日期设置区间。"
    selected = dates.tail(window_size)
    start = selected.iloc[0].date()
    end = selected.iloc[-1].date()
    if len(selected) < window_size:
        return start, end, f"{normalized} 本地仅有 {len(selected)} 根K线，不足近 {window_size} 根；已使用全部可用区间。"
    return start, end, f"{normalized} 已选择近 {window_size} 根K线：{start:%Y-%m-%d} 至 {end:%Y-%m-%d}。"


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
        before = data_check(
            symbols=[symbol],
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
        )
        download_start = start
        if not before.empty:
            download_start = _repair_partial_download_start(
                before.iloc[0],
                data_root=data_root,
                timeframe=timeframe,
                adjust=adjust,
                requested_start=start,
            )
        result = update_local_bars(
            symbols=[symbol],
            timeframe=timeframe,
            adjust=adjust,
            start=download_start,
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
            checked.loc[checked.index[0], "message"] = f"下载命令执行后仍未完整覆盖；{message}".rstrip("；")
        rows.append(checked.iloc[0].to_dict())
        status = str(checked["status"].iloc[0]) if not checked.empty and "status" in checked.columns else "unknown"
        if progress_callback is not None:
            progress_callback(index + 1, total, symbol, status)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "status", "rows", "start", "end", "message"])


def _symbols_requiring_download(check: pd.DataFrame) -> list[str]:
    if check.empty or not {"symbol", "status"}.issubset(check.columns):
        return []
    missing = check.loc[check["status"].astype(str).isin(DOWNLOAD_REQUIRED_STATUSES), "symbol"]
    return unique_symbols(missing.astype(str).tolist())


def _repair_partial_download_start(
    check_row: pd.Series,
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    requested_start: str | pd.Timestamp,
) -> str:
    requested_start_ts = pd.Timestamp(requested_start)
    if str(check_row.get("status", "")) != "partial_window":
        return requested_start_ts.strftime("%Y-%m-%d")
    actual_start = pd.to_datetime(check_row.get("start"), errors="coerce")
    if pd.isna(actual_start) or actual_start.normalize() <= requested_start_ts.normalize():
        return requested_start_ts.strftime("%Y-%m-%d")
    symbol = normalize_symbol(str(check_row.get("symbol", "")))
    if not symbol:
        return requested_start_ts.strftime("%Y-%m-%d")
    file_path = resolve_timeframe_root(data_root, timeframe) / adjust / f"{symbol}.parquet"
    if not file_path.exists():
        return requested_start_ts.strftime("%Y-%m-%d")
    try:
        dates = pd.to_datetime(pd.read_parquet(file_path, columns=["date"])["date"], errors="coerce").dropna()
    except Exception:  # noqa: BLE001
        return requested_start_ts.strftime("%Y-%m-%d")
    if dates.empty:
        return requested_start_ts.strftime("%Y-%m-%d")
    earliest = dates.min().normalize()
    if earliest < requested_start_ts.normalize():
        return (earliest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return requested_start_ts.strftime("%Y-%m-%d")


def _forward_stats_load_end(end: str | pd.Timestamp, today: pd.Timestamp | None = None) -> str:
    end_ts = pd.Timestamp(end)
    current_day = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    if end_ts >= current_day:
        return end_ts.strftime("%Y-%m-%d")
    return min(end_ts + pd.Timedelta(days=45), current_day).strftime("%Y-%m-%d")


def _cross_section_load_end(end: str | pd.Timestamp, date_tolerance_bars: int, today: pd.Timestamp | None = None) -> str:
    tolerant_end = pd.Timestamp(end) + pd.Timedelta(days=_date_tolerance_calendar_days(date_tolerance_bars))
    return _forward_stats_load_end(tolerant_end, today=today)


def _date_tolerance_load_start(start: str | pd.Timestamp, date_tolerance_bars: int) -> str:
    return (pd.Timestamp(start) - pd.Timedelta(days=_date_tolerance_calendar_days(date_tolerance_bars))).strftime("%Y-%m-%d")


def _date_tolerance_calendar_days(date_tolerance_bars: int) -> int:
    if date_tolerance_bars <= 0:
        return 0
    return max(date_tolerance_bars + 2, int(math.ceil(date_tolerance_bars * 2.2)))


def _cross_section_quick_window(
    bars: pd.DataFrame,
    window_size: int,
    today: pd.Timestamp | None = None,
) -> tuple[date, date]:
    start, end, _selected_count, _total_count, _is_sparse = _cross_section_quick_window_selection(
        bars,
        window_size,
        today,
    )
    return start, end


def _cross_section_quick_window_feedback(
    bars: pd.DataFrame,
    window_size: int,
    target_symbol: str,
    today: pd.Timestamp | None = None,
) -> tuple[date, date, str]:
    start, end, selected_count, total_count, is_sparse = _cross_section_quick_window_selection(bars, window_size, today)
    symbol = normalize_symbol(target_symbol)
    window_text = f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d}"
    if total_count == 0:
        return start, end, f"{symbol} 未找到本地行情，已按自然日近 {window_size} 天设置区间：{window_text}。"
    if is_sparse:
        return (
            start,
            end,
            f"{symbol} 本地数据疑似不连续，近期仅有 {selected_count} 根K线，"
            f"不足近 {window_size} 根；已使用近期可用区间：{window_text}。请先补齐行情数据。",
        )
    if selected_count < window_size:
        return start, end, f"{symbol} 本地仅有 {selected_count} 根K线，不足近 {window_size} 根；已使用全部可用区间：{window_text}。"
    return start, end, f"{symbol} 已选择近 {window_size} 根K线：{window_text}。"


def _cross_section_quick_window_selection(
    bars: pd.DataFrame,
    window_size: int,
    today: pd.Timestamp | None = None,
) -> tuple[date, date, int, int, bool]:
    if window_size < 1:
        raise ValueError("window_size 至少需要 1。")
    if not bars.empty and "date" in bars.columns:
        dates = pd.to_datetime(bars["date"], errors="coerce").dropna().sort_values().drop_duplicates()
        if not dates.empty:
            selected = dates.tail(window_size)
            if len(selected) >= window_size and _quick_window_is_sparse(selected, window_size):
                end = selected.iloc[-1]
                start_limit = end - pd.Timedelta(days=_quick_window_max_calendar_days(window_size) - 1)
                selected = dates.loc[dates >= start_limit]
                return selected.iloc[0].date(), end.date(), int(len(selected)), int(len(dates)), True
            return selected.iloc[0].date(), selected.iloc[-1].date(), int(len(selected)), int(len(dates)), False
    end = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    start = end - pd.Timedelta(days=window_size - 1)
    return start.date(), end.date(), window_size, 0, False


def _quick_window_is_sparse(selected_dates: pd.Series, window_size: int) -> bool:
    calendar_days = int((selected_dates.iloc[-1] - selected_dates.iloc[0]).days) + 1
    return calendar_days > _quick_window_max_calendar_days(window_size)


def _quick_window_max_calendar_days(window_size: int) -> int:
    return max(window_size + 2, int(math.ceil(window_size * 2.2)))


def _stock_name_map_from_table(table: pd.DataFrame, symbols: tuple[str, ...]) -> dict[str, str]:
    if table.empty:
        return {}
    code_column = next((column for column in ("code", "stock_code", "symbol", "证券代码", "代码", "股票代码") if column in table.columns), "")
    name_column = next((column for column in ("name", "stock_name", "股票名称", "证券简称", "名称", "简称") if column in table.columns), "")
    if not code_column or not name_column:
        return {}
    wanted = set(unique_symbols(symbols))
    result: dict[str, str] = {}
    for _, row in table[[code_column, name_column]].dropna(subset=[code_column]).iterrows():
        symbol = normalize_symbol(row[code_column])
        if symbol in wanted:
            result[symbol] = "" if pd.isna(row[name_column]) else str(row[name_column]).strip()
    return result


def _format_results(frame: pd.DataFrame, stock_names: dict[str, str] | None = None) -> pd.DataFrame:
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
    rename_map.update({"区间开始": "命中区间开始", "区间结束": "命中区间结束"})
    result = result.rename(columns=rename_map)
    for column in ["命中区间开始", "命中区间结束"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    if "symbol" in result.columns:
        names = {normalize_symbol(symbol): name for symbol, name in (stock_names or {}).items()}
        insert_at = result.columns.get_loc("symbol") + 1
        result.insert(insert_at, "股票", result["symbol"].map(lambda symbol: names.get(normalize_symbol(symbol), "")))
        result = result.rename(columns={"symbol": "代码"})
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


def _cross_section_result_metrics(result: CrossSectionSearchResult) -> list[tuple[str, str]]:
    return [("目标窗口 K 线数", f"{result.window_size:,}"), ("有效结果数", f"{len(result.results):,}")]


def _history_overview_metrics(frame: pd.DataFrame) -> list[tuple[str, str]]:
    if frame.empty:
        return [("有效样本", "0"), ("平均相似度", "-"), ("后验胜率", "-"), ("Top3后验均值", "-")]
    similarity = pd.to_numeric(frame.get("综合相似度"), errors="coerce")
    return_columns = _forward_return_columns(frame)
    if not return_columns:
        return [("有效样本", f"{len(frame):,}"), ("平均相似度", _percent_text(similarity.mean())), ("后验胜率", "-"), ("Top3后验均值", "-")]
    values = pd.to_numeric(frame[return_columns[0]], errors="coerce")
    horizon = return_columns[0].removeprefix("t_plus_").removesuffix("_return")
    return [
        ("有效样本", f"{len(frame):,}"),
        ("平均相似度", _percent_text(similarity.mean())),
        (f"后{horizon}根胜率", _percent_text((values.dropna() > 0).mean())),
        (f"Top3后{horizon}根均值", _percent_text(values.head(3).mean())),
    ]


def _history_forward_summary(frame: pd.DataFrame) -> pd.DataFrame:
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
        drawdowns = pd.to_numeric(frame.get(f"t_plus_{horizon}_max_drawdown"), errors="coerce")
        favorable = pd.to_numeric(frame.get(f"t_plus_{horizon}_max_favorable"), errors="coerce")
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
                "最好收益": float(values.loc[best_index]),
                "最差窗口": _history_window_label(frame, worst_index),
                "最差收益": float(values.loc[worst_index]),
                "相似度-收益相关": _series_corr(similarity, values),
            }
        )
    return pd.DataFrame(rows)


def _history_bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
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


def _history_window_label(frame: pd.DataFrame, index: int) -> str:
    if "窗口开始" not in frame.columns or "窗口结束" not in frame.columns:
        return ""
    return f"{_date_text(frame.loc[index, '窗口开始'])} 至 {_date_text(frame.loc[index, '窗口结束'])}"


def _size_spread_series(
    bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp = SIZE_SPREAD_START,
    small_symbol: str = SIZE_SPREAD_SMALL_SYMBOL,
    large_symbol: str = SIZE_SPREAD_LARGE_SYMBOL,
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


def _size_spread_window_stats(
    spread: pd.DataFrame,
    current_window: pd.DataFrame,
    historical_windows: list[pd.DataFrame],
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
    windows.extend((f"样本{index}", window) for index, window in enumerate(historical_windows[:top_n], start=1))
    for label, window in windows:
        if window.empty or "date" not in window.columns:
            continue
        start = pd.Timestamp(window["date"].min())
        end = inclusive_end_timestamp(window["date"].max())
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


def _format_history_stats(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    percent_columns = [
        column
        for column in result.columns
        if any(keyword in column for keyword in ("收益", "胜率", "相似度", "回撤", "浮盈"))
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


def _format_size_spread_stats(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = _format_percent_columns(result, ["起点价差率", "终点价差率", "区间变化", "区间均值", "终点历史分位"])
    for column in ["区间开始", "区间结束"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
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


def _format_data_check_status(frame: pd.DataFrame) -> pd.DataFrame:
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
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result.rename(
        columns={
            "requested_start": "请求开始",
            "requested_end": "请求结束",
            "local_start": "本地开始",
            "local_end": "本地结束",
        }
    )


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
    stock_names: dict[str, str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    symbols = unique_symbols([result.target_symbol, *result.results["symbol"].head(top_n).tolist()])
    end_marker = pd.Timestamp(result.end).strftime("%Y-%m-%d")
    for symbol in symbols:
        window_start, _window_end = _cross_section_symbol_window(result, symbol)
        symbol_bars = bars.loc[(bars["stock_code"] == symbol) & (bars["date"] >= window_start)].sort_values("date")
        if symbol_bars.empty:
            continue
        is_target = symbol == result.target_symbol
        fig.add_scatter(
            x=symbol_bars["date"],
            y=symbol_bars["close"],
            mode="lines",
            name=_stock_chart_label(symbol, stock_names, is_target=is_target),
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
        text="目标区间结束",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"color": "#2563eb", "size": 12},
    )
    fig.update_layout(
        title="目标区间与Top相似标的命中区间收盘价走势",
        xaxis_title="日期",
        yaxis_title="收盘价",
        hovermode="x unified",
    )
    return fig


def _size_spread_chart(
    spread: pd.DataFrame,
    current_window: pd.DataFrame,
    historical_windows: list[pd.DataFrame],
    top_n: int = 6,
) -> go.Figure:
    fig = go.Figure()
    if spread.empty:
        fig.update_layout(title="大小盘价差率")
        return fig
    spread_frame = spread.copy()
    spread_frame["date"] = pd.to_datetime(spread_frame["date"], errors="coerce")
    spread_frame["大小盘价差率"] = pd.to_numeric(spread_frame["大小盘价差率"], errors="coerce")
    spread_frame = spread_frame.dropna(subset=["date", "大小盘价差率"]).sort_values("date")
    fig.add_scatter(
        x=spread_frame["date"],
        y=spread_frame["大小盘价差率"],
        mode="lines",
        name="大小盘价差率",
        line={"color": "#dc2626", "width": 2},
    )
    fig.add_hline(y=0, line={"color": "#6b7280", "dash": "dot", "width": 1})
    _add_spread_window_vrect(fig, current_window, label="当前窗口", color="#2563eb", opacity=0.14)
    for index, window in enumerate(historical_windows[:top_n], start=1):
        _add_spread_window_vrect(fig, window, label=f"样本{index}", color="#64748b", opacity=0.07)
    fig.update_layout(
        title="大小盘价差率：中证1000 - 沪深300（2016-01-01归一）",
        xaxis_title="日期",
        yaxis_title="价差率",
        yaxis={"tickformat": ".2%"},
        hovermode="x unified",
    )
    return fig


def _add_spread_window_vrect(fig: go.Figure, window: pd.DataFrame, *, label: str, color: str, opacity: float) -> None:
    if window.empty or "date" not in window.columns:
        return
    start = pd.Timestamp(window["date"].min())
    end = pd.Timestamp(window["date"].max())
    if pd.isna(start) or pd.isna(end):
        return
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=color,
        opacity=opacity,
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
    )


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


def _history_kline_series(
    bars: pd.DataFrame,
    result: HistorySearchResult,
    top_n: int = 6,
    forward_bars: int = 10,
) -> list[dict[str, object]]:
    symbol_bars = bars.loc[bars["stock_code"] == result.symbol].sort_values("date").reset_index(drop=True)
    if symbol_bars.empty:
        return []
    windows: list[tuple[str, pd.DataFrame]] = [("当前窗口", result.current_window)]
    windows.extend((f"样本{index}", window) for index, window in enumerate(result.historical_windows[:top_n], start=1))
    series: list[dict[str, object]] = []
    for title, window in windows:
        if window.empty:
            continue
        start = pd.Timestamp(window["date"].min())
        end = inclusive_end_timestamp(window["date"].max())
        chart_source = symbol_bars.loc[symbol_bars["date"] >= start].sort_values("date")
        if chart_source.empty:
            continue
        matching_window = chart_source.loc[chart_source["date"] <= end]
        if matching_window.empty:
            continue
        chart_window = chart_source.head(len(matching_window) + forward_bars)
        series.append(
            {
                "title": _history_kline_title(title, matching_window),
                "windowEndTime": pd.Timestamp(matching_window["date"].iloc[-1]).strftime("%Y-%m-%d"),
                "windowSize": int(len(matching_window)),
                "forwardSize": int(max(0, len(chart_window) - len(matching_window))),
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


def _history_kline_title(label: str, window: pd.DataFrame) -> str:
    if window.empty or "date" not in window.columns:
        return label
    return f"{label}（{_date_text(window['date'].min())} 至 {_date_text(window['date'].max())}）"


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons or any(item <= 0 for item in horizons):
        raise ValueError("后验观察窗口必须是逗号分隔的正整数。")
    return horizons


def _date_range_error(start: str | pd.Timestamp, end: str | pd.Timestamp) -> str:
    return "区间开始不能晚于区间结束。" if pd.Timestamp(start) > pd.Timestamp(end) else ""


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _stock_chart_label(symbol: str, stock_names: dict[str, str] | None = None, *, is_target: bool = False) -> str:
    normalized = normalize_symbol(symbol)
    name = (stock_names or {}).get(normalized, "").strip()
    if not name:
        return f"{normalized}（目标）" if is_target else normalized
    return f"{name}（{normalized}，目标）" if is_target else f"{name}（{normalized}）"


def _lightweight_kline_series(
    bars: pd.DataFrame,
    result: CrossSectionSearchResult,
    top_n: int = 6,
    forward_bars: int = 10,
    stock_names: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    symbols = [result.target_symbol, *result.results["symbol"].head(top_n).tolist()]
    series: list[dict[str, object]] = []
    for symbol in symbols:
        start, end = _cross_section_symbol_window(result, symbol)
        end = inclusive_end_timestamp(end)
        symbol_bars = bars.loc[(bars["stock_code"] == symbol) & (bars["date"] >= start)].sort_values("date")
        if symbol_bars.empty:
            continue
        window = symbol_bars.loc[symbol_bars["date"] <= end]
        if window.empty:
            continue
        chart_window = symbol_bars.head(len(window) + forward_bars)
        window_end_time = pd.Timestamp(window["date"].iloc[-1]).strftime("%Y-%m-%d")
        label = _stock_chart_label(symbol, stock_names, is_target=symbol == result.target_symbol)
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
    start = row.get("区间开始", result.start)
    end = row.get("区间结束", result.end)
    return pd.Timestamp(start), pd.Timestamp(end)


def _lightweight_kline_chart_html(series: list[dict[str, object]]) -> str:
    if not series:
        return _kline_empty_message()
    panels = "\n".join(_kline_svg_panel(item) for item in series)
    return f"""
<style>
.klineGrid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  width: 100%;
  align-items: start;
}}
@media (max-width: 980px) {{
  .klineGrid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 640px) {{
  .klineGrid {{ grid-template-columns: 1fr; }}
}}
</style>
<div class="klineGrid">
{panels}
</div>
"""


def _kline_chart_component_height(series: list[dict[str, object]]) -> int:
    if not series:
        return 160
    panel_height = 390
    gap = 18
    rows = math.ceil(len(series) / 3)
    return rows * panel_height + max(0, rows - 1) * gap + 32


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
<div data-kline-panel="1" style="border:1px solid #e5e7eb;border-radius:6px;padding:10px;background:#ffffff;">
  <div style="font:600 13px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin-bottom:6px;color:#111827;">{title}</div>
  <div style="height:300px;display:flex;align-items:center;justify-content:center;color:#6b7280;font:13px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">没有可绘制的K线数据</div>
</div>
"""

    width = 420.0
    height = 300.0
    left = 48.0
    right = 14.0
    top = 16.0
    bottom = 44.0
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
            "<title>命中区间结束</title></line>"
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
<div data-kline-panel="1" style="border:1px solid #e5e7eb;border-radius:6px;padding:10px;background:#ffffff;">
  <div style="font:600 13px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin-bottom:6px;color:#111827;">{title}</div>
  <svg viewBox="0 0 420 300" role="img" aria-label="{title} K线图" style="width:100%;height:300px;display:block;">{svg}</svg>
  <div style="font:12px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#4b5563;margin-top:4px;">命中区间 {window_size} 根 | 后续 {forward_size} 根</div>
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
