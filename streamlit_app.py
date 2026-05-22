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
from ashare_cross_section_similarity.data_manager import (
    KLINE_FILE_PATTERNS,
    migrate_kline_data,
    plan_kline_migration,
)
from ashare_cross_section_similarity.downloader import (
    data_check,
    default_trend_repo,
    plan_incremental_downloads,
    update_local_bars,
)
from ashare_cross_section_similarity.features import normalized_close_path, z_normalize
from ashare_cross_section_similarity.history import HistorySearchConfig, HistorySearchResult, search_history
from ashare_cross_section_similarity.review import (
    ReviewConfig,
    ReviewResult,
    analyze_price_review,
    build_video_script_profile,
    build_comparison_stats,
    build_equal_weight_series,
    render_multi_review_text,
    render_review_text,
    render_video_script_cards_html,
)
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    CrossSectionSearchResult,
    search_cross_section,
)
from ashare_cross_section_similarity.similarity_algorithms import (
    ALGORITHM_CHOICES,
    BASELINE_ALGORITHM,
    algorithm_label,
    get_algorithm_status,
)
from ashare_cross_section_similarity.tdx_source import fetch_tdx_etf_index, fetch_tdx_kline_symbols, search_tdx_etf_index
from ashare_cross_section_similarity.universe import (
    DEFAULT_ANALYSIS_INDEX_SYMBOLS,
    fetch_concept_constituents,
    fetch_industry_constituents,
    normalize_symbol,
    symbols_with_analysis_indexes,
    unique_symbols,
)


PERCENT_COLUMNS = ["综合相似度", "路径相似度", "特征相似度", "区间收益", "波动率", "最大回撤", "下跌放量占比", "覆盖率"]
DECIMAL_COLUMNS = ["路径距离", "价格路径距离", "收益路径距离", "趋势斜率", "量价相关", "成交规模", "特征距离"]
DOWNLOAD_REQUIRED_STATUSES = {"missing_file", "missing_window", "partial_window", "read_error"}
DOWNLOAD_BATCH_SIZE = 100
DOWNLOAD_JOB_STATUSES = {"running", "paused", "completed"}
DOWNLOAD_JOB_STATUS_LABELS = {
    "running": "下载中",
    "paused": "已暂停",
    "completed": "下载完成",
}
UNIVERSE_FILE_TYPES = [("搜索范围文件", ("*.csv", "*.xlsx", "*.xls", "*.parquet")), ("所有文件", "*")]
KLINE_DATA_FILE_TYPES = [("K线数据文件", KLINE_FILE_PATTERNS), ("所有文件", "*")]
SIZE_SPREAD_START = "2016-01-01"
SCRIPT_BENCHMARK_SYMBOL = "000300.SH"
REVIEW_MAX_TARGET_SYMBOLS = 36
SIZE_SPREAD_SMALL_SYMBOL = "000852.SH"
SIZE_SPREAD_LARGE_SYMBOL = "000300.SH"
SIZE_SPREAD_SYMBOLS = (SIZE_SPREAD_SMALL_SYMBOL, SIZE_SPREAD_LARGE_SYMBOL)
DATE_INPUT_MIN = date(1990, 1, 1)
DATE_INPUT_MAX = date(2100, 12, 31)


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
        _render_data_archive_manager(data_root=data_root)

    _render_full_daily_tdx_update(trend_repo=trend_repo, data_root=data_root, adjust=adjust)
    _render_algorithm_benchmark_entry(data_root=data_root, timeframe=timeframe, adjust=adjust)

    history_tab, cross_section_tab, review_tab = st.tabs(["历史时序相似", "横截面相似", "走势复盘"])
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
    with review_tab:
        _render_review_tab(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            provider=provider,
            download_engine=download_engine,
        )


def _render_directory_picker(label: str, default_path: str | Path, key: str) -> str:
    state_key = f"{key}_path"
    default_key = f"{key}_default_path"
    error_key = f"{key}_dialog_error"
    default_text = _path_text(default_path)
    previous_default = st.session_state.get(default_key)
    if state_key not in st.session_state or st.session_state.get(state_key) == previous_default:
        st.session_state[state_key] = default_text
    st.session_state[default_key] = default_text
    st.caption(label)
    _render_selected_path(_picker_path_text(str(st.session_state[state_key])))
    button_col, clear_col = st.columns([2, 1])
    if button_col.button(f"选择{label}", key=f"{key}_pick"):
        selected_path, error = _pick_directory_with_system_dialog(label, st.session_state[state_key])
        if error is not None:
            st.session_state[error_key] = error
        elif selected_path is not None:
            st.session_state[state_key] = selected_path
            st.session_state.pop(error_key, None)
            st.rerun()
        else:
            st.session_state.pop(error_key, None)
    if st.session_state[state_key] and clear_col.button("清除", key=f"{key}_clear"):
        st.session_state[state_key] = ""
        st.session_state.pop(error_key, None)
        st.rerun()
    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
    return str(st.session_state[state_key])


def _date_input_args(
    key: str,
    default_value: date,
    *,
    session_state: object | None = None,
) -> dict[str, object]:
    state = st.session_state if session_state is None else session_state
    args: dict[str, object] = {
        "key": key,
        "min_value": DATE_INPUT_MIN,
        "max_value": DATE_INPUT_MAX,
    }
    if key not in state:
        args["value"] = default_value
    return args


def _pick_directory_with_system_dialog(
    label: str,
    current_path: str | Path,
    *,
    tk_factory: Callable[[], object] | None = None,
    askdirectory: Callable[..., str] | None = None,
) -> tuple[str | None, str | None]:
    root: object | None = None
    try:
        if tk_factory is None or askdirectory is None:
            import tkinter as tk
            from tkinter import filedialog

            tk_factory = tk.Tk
            askdirectory = filedialog.askdirectory
        root = tk_factory()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected_dir = askdirectory(
            title=f"选择{label}",
            initialdir=_picker_initial_directory(current_path, Path.home()),
            mustexist=True,
        )
        if not selected_dir:
            return None, None
        return str(Path(selected_dir).expanduser()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"无法打开系统文件夹选择器：{exc}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


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
    _render_selected_path(_picker_path_text(str(st.session_state[state_key])))
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


def _picker_path_text(path: str | Path) -> str:
    text = str(path).strip()
    return text if text else "未选择"


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


def _algorithm_option_label(name: str) -> str:
    status = get_algorithm_status(name)
    suffix = "" if status.available else "（未安装依赖）"
    return f"{algorithm_label(name)}{suffix}"


def _render_algorithm_benchmark_entry(*, data_root: str, timeframe: str, adjust: str) -> None:
    with st.expander("算法核验图集", expanded=False):
        st.caption("用固定样本对不同相似算法输出 Top 结果、运行时间和 HTML 图集；适合肉眼复核形态一致性。")
        command = (
            "python -m ashare_cross_section_similarity benchmark "
            "--cases docs/research/similarity_benchmark_cases.yaml "
            "--algorithms baseline_price_feature,return_shape,hybrid_shape_v2,dtw_optional "
            f"--data-root {data_root} --timeframe {timeframe} --adjust {adjust} "
            "--output outputs/research"
        )
        st.code(command, language="bash")


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
    start_input = _date_input_args("history_start_date", date(2024, 3, 4))
    end_input = _date_input_args("history_end_date", date(2024, 3, 31))
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
    alg_col, weight_col = st.columns([1, 2])
    algorithm = alg_col.selectbox(
        "相似算法",
        ALGORITHM_CHOICES,
        index=ALGORITHM_CHOICES.index(BASELINE_ALGORITHM),
        format_func=_algorithm_option_label,
        key="history_algorithm",
    )
    path_weight = weight_col.slider("走势权重", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="history_path_weight")
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
        if hint := _symbol_data_hint(symbol, data_root=data_root, timeframe=timeframe, adjust=adjust):
            st.warning(hint)
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
            st.session_state["history_download_job"] = _create_download_job(
                symbols=[normalized_symbol],
                timeframe=timeframe,
                adjust=adjust,
                start=download_start,
                end=as_of,
                trend_repo=Path(trend_repo),
                data_root=Path(data_root),
                provider=provider,
                download_engine=download_engine,
            )
            st.rerun()
        _render_download_job("history_download_job", target_symbol=normalized_symbol)

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
                algorithm=str(algorithm),
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
    st.caption(f"相似算法：{algorithm_label(str(algorithm))}")
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
    start_input = _date_input_args("cross_start_date", date(2024, 1, 1))
    end_input = _date_input_args("cross_end_date", date(2024, 3, 31))
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
    col4, col5, col6, col10, col11 = st.columns(5)
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
    algorithm = col11.selectbox(
        "相似算法",
        ALGORITHM_CHOICES,
        index=ALGORITHM_CHOICES.index(BASELINE_ALGORITHM),
        format_func=_algorithm_option_label,
        key="cross_algorithm",
    )
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
    normalized_targets = unique_symbols([target_symbol])
    normalized_target = normalized_targets[0] if normalized_targets else str(target_symbol).strip().upper()
    if st.button("检查并下载缺失行情", key="cross_download"):
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
            st.session_state["cross_download_job"] = _create_download_job(
                symbols=download_symbols,
                timeframe=timeframe,
                adjust=adjust,
                start=coverage_start,
                end=coverage_end,
                trend_repo=Path(trend_repo),
                data_root=Path(data_root),
                provider=provider,
                download_engine=download_engine,
            )
            st.rerun()

    update_result = _render_download_job("cross_download_job", target_symbol=normalized_target)
    job = st.session_state.get("cross_download_job")
    if job is not None and job.get("status") == "completed":
        st.session_state.pop("cross_data_check", None)
        st.session_state.pop("cross_data_check_key", None)
        if not update_result.empty and normalized_target in set(update_result.get("symbol", pd.Series(dtype=str)).astype(str)):
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
                top_n=_cross_section_search_limit(universe, int(top_n)),
                min_coverage=float(min_coverage),
                path_weight=float(path_weight),
                date_tolerance_bars=tolerance_bars,
                algorithm=str(algorithm),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    st.markdown("**4. 搜索结果**")
    for column, (label, value) in zip(st.columns(2), _cross_section_result_metrics(result)):
        column.metric(label, value)
    st.caption(f"相似算法：{algorithm_label(str(algorithm))}")
    if result.results.empty:
        st.warning("没有找到可用结果。请检查本地数据覆盖、搜索范围和区间设置。")
        return

    display_results = _display_results(result.results, int(top_n))
    name_count = max(int(top_n), 6)
    stock_names = _cached_stock_name_map(
        tuple(unique_symbols([result.target_symbol, *result.results["symbol"].head(name_count).astype(str).tolist()]))
    )
    st.caption(f"当前展示前 {len(display_results):,} / {len(result.results):,} 条；下方统计和 CSV 基于全部有效结果。")
    st.dataframe(_centered(_format_results(display_results, stock_names)), use_container_width=True, hide_index=True)
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


def _render_review_tab(*, data_root: str, timeframe: str, adjust: str, provider: str, download_engine: str) -> None:
    st.subheader("走势复盘")
    st.caption("基于本地K线识别主要上涨、回撤、下跌和反弹段，生成可复验的数据化自然语言复盘。")
    review_mode = st.radio(
        "复盘模式",
        ["单股票", "多股票"],
        horizontal=True,
        key="review_mode",
        help=f"多股票模式最多支持 {REVIEW_MAX_TARGET_SYMBOLS} 个标的，所有标的使用同一个日期区间和对标设置。",
    )
    is_multi_review = review_mode == "多股票"
    col1, col2, col3 = st.columns(3)
    if is_multi_review:
        raw_target_symbols = col1.text_area(
            f"目标代码（最多{REVIEW_MAX_TARGET_SYMBOLS}个）",
            value="601888.SH\n688603.SH\n300750.SZ",
            key="review_symbols",
            height=180,
            help="逗号、空格或换行分隔；所有标的使用统一复盘区间。",
        )
        target_symbols, target_error = _review_target_symbols(raw_target_symbols)
        target_symbol = target_symbols[0] if target_symbols else ""
    else:
        target_symbol = col1.text_input("目标代码", value="601888.SH", key="review_symbol")
        normalized_single = normalize_symbol(target_symbol)
        target_symbols = [normalized_single] if normalized_single else []
        target_error = "" if normalized_single else "请输入目标代码。"
    start_input = _date_input_args("review_start_date", date(2025, 12, 19))
    end_input = _date_input_args("review_end_date", date(2026, 5, 18))
    start_date = col2.date_input("区间开始", **start_input)
    end_date = col3.date_input("区间结束", **end_input)

    quick_cols = st.columns(4)
    quick_cols[0].caption("快捷区间")
    for button_col, window_size in zip(quick_cols[1:], [20, 60, 120]):
        button_col.button(
            f"近{window_size}根",
            key=f"review_quick_{window_size}",
            on_click=_set_review_quick_window,
            args=(data_root, timeframe, adjust, target_symbol, window_size),
        )
    if st.session_state.get("review_quick_message"):
        st.info(st.session_state["review_quick_message"])

    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    if error := _date_range_error(start, end):
        st.error(error)
        return

    param_col1, param_col2 = st.columns(2)
    min_swing_percent = param_col1.slider(
        "最小波段幅度",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        format="%d%%",
        key="review_min_swing_return",
        help="低于该涨跌幅的波动会被视为噪声，不进入主要波段。",
    )
    min_segment_bars = param_col2.number_input(
        "最小段落K线数",
        min_value=2,
        max_value=60,
        value=3,
        step=1,
        key="review_min_segment_bars",
    )

    index_enabled = st.checkbox("结合指数分析", value=True, key="review_with_index")
    index_symbols: list[str] = []
    if index_enabled:
        index_col1, index_col2 = st.columns([2, 1])
        selected_indexes = index_col1.multiselect(
            "指数代码",
            list(DEFAULT_ANALYSIS_INDEX_SYMBOLS),
            default=["000300.SH", "000852.SH", "399006.SZ"],
            key="review_index_symbols",
        )
        extra_indexes = index_col2.text_input("额外指数代码", value="", key="review_extra_indexes")
        index_symbols = unique_symbols([*selected_indexes, *_split_symbol_text(extra_indexes)])

    sector_enabled = st.checkbox("结合板块分析", value=False, key="review_with_sector")
    proxy_symbols: list[str] = []
    auto_proxy_symbols: list[str] = []
    auto_proxy_names: dict[str, str] = {}
    industry_name = ""
    concept_name = ""
    sector_min_coverage = 0.5
    if sector_enabled:
        sector_col1, sector_col2, sector_col3, sector_col4 = st.columns([1.4, 1, 1, 1])
        proxy_symbols = unique_symbols(_split_symbol_text(sector_col1.text_input("板块/ETF/指数代理代码", value="", key="review_proxy_symbols")))
        industry_name = sector_col2.text_input("行业名称", value="", key="review_industry_name")
        concept_name = sector_col3.text_input("概念名称", value="", key="review_concept_name")
        sector_min_coverage = sector_col4.slider(
            "成分覆盖率",
            min_value=0.3,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="review_sector_min_coverage",
        )
        if download_engine == "tdx":
            auto_proxy_symbols, auto_proxy_names, auto_matches, auto_warning = _review_auto_tdx_etf_proxies(
                provider,
                industry_name=industry_name,
                concept_name=concept_name,
            )
            if auto_warning:
                st.warning(auto_warning)
            elif not auto_matches.empty:
                st.caption("TDX 自动匹配 ETF（同一关键词保留成交额最大）")
                st.dataframe(_centered(_format_tdx_etf_matches(auto_matches)), use_container_width=True, hide_index=True)
            else:
                st.caption("TDX ETF 自动匹配：输入行业或概念名称后，会从本地 TDX ETF 清单中选择成交额最大的同类 ETF。")
    combined_proxy_symbols = unique_symbols([*proxy_symbols, *auto_proxy_symbols])

    if not st.button("生成走势复盘", type="primary", key="review_run"):
        st.info("设置目标、区间和对比项后，点击生成走势复盘。复盘只使用本地行情数据。")
        return

    if target_error and not target_symbols:
        st.error(target_error)
        return
    if target_error:
        st.warning(target_error)
    if is_multi_review:
        _render_multi_review_output(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            target_symbols=target_symbols,
            start=start,
            end=end,
            min_swing_percent=int(min_swing_percent),
            min_segment_bars=int(min_segment_bars),
            index_symbols=index_symbols,
            proxy_symbols=combined_proxy_symbols,
            industry_name=industry_name,
            concept_name=concept_name,
            sector_min_coverage=float(sector_min_coverage),
            extra_stock_names=auto_proxy_names,
        )
        return

    normalized_target = target_symbols[0]
    direct_symbols = unique_symbols([normalized_target, *index_symbols, *combined_proxy_symbols, SCRIPT_BENCHMARK_SYMBOL])
    data_start = _review_script_data_start(start, end)
    direct_fingerprint = _local_data_fingerprint(data_root, timeframe, adjust, tuple(direct_symbols))
    try:
        direct_bars = _cached_load_local_bars(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=tuple(direct_symbols),
            start=data_start,
            end=end,
            data_fingerprint=direct_fingerprint,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"本地行情读取失败：{exc}")
        return

    target_window = direct_bars.loc[direct_bars["stock_code"] == normalized_target]
    result = analyze_price_review(
        target_window,
        ReviewConfig(
            symbol=normalized_target,
            start=start,
            end=end,
            min_swing_return=float(min_swing_percent) / 100.0,
            min_segment_bars=int(min_segment_bars),
        ),
    )

    if result.window.empty:
        st.markdown("**1. 区间概览**")
        empty_comparison = pd.DataFrame()
        for column, (label, value) in zip(st.columns(6), _review_metric_items(result, empty_comparison, index_symbols)):
            column.metric(label, value)
        st.warning("目标标的在所选区间没有本地行情。请检查代码、周期、复权目录或先下载数据。")
        if hint := _symbol_data_hint(target_symbol, data_root=data_root, timeframe=timeframe, adjust=adjust):
            st.warning(hint)
        return

    stock_names = {**_cached_stock_name_map(tuple(direct_symbols)), **auto_proxy_names}
    comparison_frames, comparison_rows, warnings = _review_comparison_data(
        result.window,
        direct_bars,
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        index_symbols=index_symbols,
        proxy_symbols=combined_proxy_symbols,
        industry_name=industry_name,
        concept_name=concept_name,
        sector_min_coverage=float(sector_min_coverage),
        stock_names=stock_names,
    )
    comparison_frame = pd.DataFrame(comparison_rows)
    script_profile = _review_video_script_profile(
        result,
        direct_bars,
        benchmark_symbol=SCRIPT_BENCHMARK_SYMBOL,
        stock_names=stock_names,
    )
    all_warnings = [*result.warnings, *warnings]

    st.markdown("**1. 区间概览**")
    metric_values = _review_metric_items(result, comparison_frame, index_symbols)
    for column, (label, value) in zip(st.columns(len(metric_values)), metric_values):
        column.metric(label, value)

    st.markdown("**2. 复盘图表**")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(_review_kline_chart(result), use_container_width=True)
    with chart_col2:
        st.plotly_chart(_review_relative_chart(result.window, comparison_frames), use_container_width=True)

    st.markdown("**3. 自然语言复盘**")
    st.markdown(render_review_text(result, comparison_frame))
    st.markdown(render_video_script_cards_html([script_profile]), unsafe_allow_html=True)
    st.dataframe(_centered(_format_video_script_profiles([script_profile])), use_container_width=True, hide_index=True)
    for warning in all_warnings:
        st.warning(warning)

    st.markdown("**4. 波段与对比明细**")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.caption("主要波段")
        st.dataframe(_centered(_format_review_segments(result.main_segments)), use_container_width=True, hide_index=True)
    with detail_col2:
        st.caption("指数 / 板块对比")
        st.dataframe(_centered(_format_review_comparisons(comparison_frame)), use_container_width=True, hide_index=True)


def _review_shared_comparison_frames(
    direct_bars: pd.DataFrame,
    *,
    data_root: str,
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
    index_symbols: list[str],
    proxy_symbols: list[str],
    industry_name: str,
    concept_name: str,
    sector_min_coverage: float,
    stock_names: dict[str, str],
) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    comparison_frames: list[tuple[str, pd.DataFrame]] = []
    warnings: list[str] = []

    def append_direct(symbol: str, label: str) -> None:
        normalized = normalize_symbol(symbol)
        frame = direct_bars.loc[direct_bars["stock_code"] == normalized].sort_values("date")
        frame = _filter_date_range(frame, start, end)
        if frame.empty:
            warnings.append(f"{label} 缺少本地行情，未纳入对比。")
            return
        comparison_frames.append((label, frame))

    for symbol in index_symbols:
        append_direct(symbol, _stock_chart_label(symbol, stock_names))
    for symbol in proxy_symbols:
        append_direct(symbol, _stock_chart_label(symbol, stock_names))

    sector_specs = [
        ("industry", "行业板块", industry_name),
        ("concept", "概念板块", concept_name),
    ]
    for kind, label_prefix, raw_name in sector_specs:
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            constituents = _cached_review_constituents(kind, name)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{label_prefix} {name} 成分获取失败：{exc}")
            continue
        if not constituents:
            warnings.append(f"{label_prefix} {name} 没有获取到成分。")
            continue
        sector_label = f"{label_prefix}:{name}"
        sector_fingerprint = _local_data_fingerprint(data_root, timeframe, adjust, tuple(constituents))
        sector_bars = _cached_load_local_bars(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=tuple(constituents),
            start=start,
            end=end,
            data_fingerprint=sector_fingerprint,
        )
        equal_weight = build_equal_weight_series(
            sector_bars,
            constituents,
            label=sector_label,
            min_coverage=sector_min_coverage,
        )
        if equal_weight.warning:
            warnings.append(equal_weight.warning)
        if not equal_weight.frame.empty:
            comparison_frames.append((sector_label, equal_weight.frame))
    return comparison_frames, warnings


def _review_script_data_start(start: str, end: str) -> str:
    start_ts = pd.Timestamp(start)
    ytd_start = pd.Timestamp(year=pd.Timestamp(end).year, month=1, day=1)
    entry_context_start = start_ts - pd.Timedelta(days=90)
    return min(entry_context_start, ytd_start).strftime("%Y-%m-%d")


def _filter_date_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    result = frame.copy()
    dates = pd.to_datetime(result["date"], errors="coerce")
    mask = dates.between(pd.Timestamp(start), inclusive_end_timestamp(end))
    return result.loc[mask].sort_values("date").reset_index(drop=True)


def _review_video_script_profile(
    result: ReviewResult,
    direct_bars: pd.DataFrame,
    *,
    benchmark_symbol: str,
    stock_names: dict[str, str],
) -> dict[str, object]:
    symbol_bars = direct_bars.loc[direct_bars["stock_code"] == result.symbol].sort_values("date")
    normalized_benchmark = normalize_symbol(benchmark_symbol)
    benchmark = direct_bars.loc[direct_bars["stock_code"] == normalized_benchmark].sort_values("date")
    profile = build_video_script_profile(
        result,
        symbol_bars,
        benchmark if not benchmark.empty else None,
        benchmark_label=_stock_chart_label(normalized_benchmark, stock_names),
    )
    profile["股票"] = stock_names.get(result.symbol, "")
    return profile


def _review_multi_comparison_rows(
    results: list[ReviewResult],
    comparison_frames: list[tuple[str, pd.DataFrame]],
    stock_names: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        if result.window.empty:
            continue
        for label, frame in comparison_frames:
            row = build_comparison_stats(result.window, frame, label)
            row["代码"] = result.symbol
            row["股票"] = stock_names.get(result.symbol, "")
            rows.append(row)
    return rows


def _render_multi_review_output(
    *,
    data_root: str,
    timeframe: str,
    adjust: str,
    target_symbols: list[str],
    start: str,
    end: str,
    min_swing_percent: int,
    min_segment_bars: int,
    index_symbols: list[str],
    proxy_symbols: list[str],
    industry_name: str,
    concept_name: str,
    sector_min_coverage: float,
    extra_stock_names: dict[str, str] | None = None,
) -> None:
    direct_symbols = unique_symbols([*target_symbols, *index_symbols, *proxy_symbols, SCRIPT_BENCHMARK_SYMBOL])
    data_start = _review_script_data_start(start, end)
    direct_fingerprint = _local_data_fingerprint(data_root, timeframe, adjust, tuple(direct_symbols))
    try:
        direct_bars = _cached_load_local_bars(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=tuple(direct_symbols),
            start=data_start,
            end=end,
            data_fingerprint=direct_fingerprint,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"本地行情读取失败：{exc}")
        return

    stock_names = {**_cached_stock_name_map(tuple(direct_symbols)), **(extra_stock_names or {})}
    results: list[ReviewResult] = []
    all_warnings: list[str] = []
    for symbol in target_symbols:
        target_window = direct_bars.loc[direct_bars["stock_code"] == symbol]
        result = analyze_price_review(
            target_window,
            ReviewConfig(
                symbol=symbol,
                start=start,
                end=end,
                min_swing_return=float(min_swing_percent) / 100.0,
                min_segment_bars=int(min_segment_bars),
            ),
        )
        results.append(result)
        all_warnings.extend(result.warnings)
        if result.window.empty:
            if hint := _symbol_data_hint(symbol, data_root=data_root, timeframe=timeframe, adjust=adjust):
                all_warnings.append(hint)
            continue

    valid_results = [result for result in results if not result.window.empty]
    if not valid_results:
        st.warning("所有目标标的在所选区间都没有本地行情。请检查代码、周期、复权目录或先下载数据。")
        for warning in all_warnings:
            st.warning(warning)
        return

    comparison_frames, comparison_warnings = _review_shared_comparison_frames(
        direct_bars,
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        start=start,
        end=end,
        index_symbols=index_symbols,
        proxy_symbols=proxy_symbols,
        industry_name=industry_name,
        concept_name=concept_name,
        sector_min_coverage=float(sector_min_coverage),
        stock_names=stock_names,
    )
    all_warnings.extend(comparison_warnings)
    comparison_rows = _review_multi_comparison_rows(valid_results, comparison_frames, stock_names)
    comparison_frame = pd.DataFrame(comparison_rows)
    script_profiles = [
        _review_video_script_profile(
            result,
            direct_bars,
            benchmark_symbol=SCRIPT_BENCHMARK_SYMBOL,
            stock_names=stock_names,
        )
        for result in valid_results
    ]
    st.markdown("**1. 多股票区间概览**")
    st.dataframe(_centered(_format_multi_review_overview(results, stock_names)), use_container_width=True, hide_index=True)

    st.markdown("**2. 多股票 K 线复盘**")
    for row in _review_result_grid_rows(results):
        columns = st.columns(3)
        for column, result in zip(columns, row):
            with column:
                fig = _review_kline_chart(result)
                fig.update_layout(title=_stock_chart_label(result.symbol, stock_names, is_target=False), height=320)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("**3. 自然语言复盘**")
    st.markdown(render_multi_review_text(valid_results, comparison_frame))
    st.markdown(render_video_script_cards_html(script_profiles), unsafe_allow_html=True)
    st.dataframe(_centered(_format_video_script_profiles(script_profiles)), use_container_width=True, hide_index=True)
    for warning in dict.fromkeys(all_warnings):
        st.warning(warning)

    st.markdown("**4. 对比与波段明细**")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.caption("个股主要波段")
        st.dataframe(_centered(_format_multi_review_segments(results, stock_names)), use_container_width=True, hide_index=True)
    with detail_col2:
        st.caption("指数 / 板块对比")
        st.dataframe(_centered(_format_multi_review_comparisons(comparison_frame)), use_container_width=True, hide_index=True)


def _render_full_daily_tdx_update(*, trend_repo: str, data_root: str, adjust: str) -> None:
    job_state = st.session_state.get("full_daily_tdx_job")
    keep_open = isinstance(job_state, dict) and str(job_state.get("status", "")) in {"running", "paused"}
    with st.expander("TDX 全量日 K 线更新", expanded=keep_open):
        st.caption(
            "通过本机通达信更新股票、ETF、行业板块指数、概念指数和常用指数 1d 日线，"
            "写入当前本地行情根目录。标的列表和价格数据都通过 TDX 获取。"
        )
        tdx_path = _render_directory_picker(
            "通达信 PYPlugins/user 目录",
            os.environ.get("TDX_TQCENTER_PATH", ""),
            "full_daily_tdx_tqcenter",
        )
        col1, col2, col3 = st.columns(3)
        start_date = col1.date_input(
            "起始日期",
            **_date_input_args("full_daily_tdx_start", DATE_INPUT_MIN),
        )
        end_date = col2.date_input(
            "结束日期",
            **_date_input_args("full_daily_tdx_end", pd.Timestamp.today().date()),
        )
        batch_size = col3.number_input(
            "每批标的数",
            min_value=1,
            max_value=500,
            value=DOWNLOAD_BATCH_SIZE,
            step=10,
            key="full_daily_tdx_batch_size",
        )
        opt_col1, opt_col2, opt_col3 = st.columns([1.2, 1.2, 2.6])
        update_mode = opt_col1.selectbox(
            "更新方式",
            ["增量补最新", "完整覆盖"],
            key="full_daily_tdx_update_mode",
            help="增量补最新只检查每个标的本地最后一根K线，自动从后一日补到结束日期；完整覆盖会按起止区间检查历史缺口。",
        )
        incremental_latest_only = update_mode == "增量补最新"
        skip_available = opt_col2.checkbox(
            "跳过已覆盖",
            value=True,
            disabled=incremental_latest_only,
            key="full_daily_tdx_skip_available",
            help="仅完整覆盖模式生效；增量补最新会自动跳过最新K线已覆盖的标的。",
        )
        include_indexes = opt_col3.checkbox("同步常用指数", value=True, key="full_daily_tdx_include_indexes")
        extra_symbols = st.text_input(
            "额外代码",
            value="",
            key="full_daily_tdx_extra_symbols",
            help="逗号、空格或换行分隔；用于补充指数、ETF 或其他代理标的，如 399006.SZ,000300.SH。",
        )
        if include_indexes:
            st.caption("常用指数：" + "、".join(DEFAULT_ANALYSIS_INDEX_SYMBOLS))
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        if error := _date_range_error(start, end):
            st.error(error)
            return

        if st.button("通过 TDX 更新全量日 K", type="primary", key="full_daily_tdx_start_button"):
            try:
                with st.spinner("获取 TDX 标的列表并检查本地覆盖..."):
                    tdx_symbols = fetch_tdx_kline_symbols(tqcenter_path=tdx_path)
                    all_symbols = _full_daily_download_universe(
                        tdx_symbols,
                        include_indexes=bool(include_indexes),
                        extra_symbols=extra_symbols,
                    )
                    download_symbols, checked = _prepare_full_daily_download_symbols(
                        symbols=all_symbols,
                        data_root=Path(data_root),
                        adjust=adjust,
                        start=start,
                        end=end,
                        skip_available=bool(skip_available),
                        incremental_latest_only=incremental_latest_only,
                    )
                st.session_state["full_daily_tdx_summary"] = {
                    "total": len(unique_symbols(all_symbols)),
                    "tdx_total": len(unique_symbols(tdx_symbols)),
                    "index_total": len(DEFAULT_ANALYSIS_INDEX_SYMBOLS) if include_indexes else 0,
                    "download": len(download_symbols),
                    "available": _full_daily_available_count(checked, incremental_latest_only=incremental_latest_only),
                    "skip_available": bool(skip_available) and not incremental_latest_only,
                    "mode": update_mode,
                    "start": start,
                    "end": end,
                }
                if download_symbols:
                    st.session_state["full_daily_tdx_job"] = _create_download_job(
                        symbols=download_symbols,
                        timeframe="1d",
                        adjust=adjust,
                        start=start,
                        end=end,
                        trend_repo=Path(trend_repo),
                        data_root=Path(data_root),
                        provider=tdx_path,
                        download_engine="tdx",
                        batch_size=int(batch_size),
                        incremental_latest_only=incremental_latest_only,
                    )
                else:
                    st.session_state.pop("full_daily_tdx_job", None)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"TDX 全量日 K 更新任务创建失败：{exc}")

        summary = st.session_state.get("full_daily_tdx_summary")
        if isinstance(summary, dict):
            tdx_total = int(summary.get("tdx_total", summary.get("stock_total", 0)))
            st.info(
                f"下载范围 {int(summary.get('total', 0)):,} 个；"
                f"其中 TDX 标的 {tdx_total:,} 个、常用指数 {int(summary.get('index_total', 0)):,} 个；"
                f"待下载 {int(summary.get('download', 0)):,} 个；"
                f"已跳过 {int(summary.get('available', 0)):,} 个；"
                f"模式 {summary.get('mode', '完整覆盖')}；"
                f"区间 {summary.get('start')} 至 {summary.get('end')}。"
            )
            if int(summary.get("download", 0)) == 0:
                st.success("当前区间本地日线已覆盖，无需下载。")

        update_result = _render_download_job("full_daily_tdx_job")
        if not update_result.empty:
            st.download_button(
                "下载全量更新日志",
                data=update_result.to_csv(index=False).encode("utf-8-sig"),
                file_name="full_a_daily_tdx_update_log.csv",
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


def _render_data_archive_manager(*, data_root: str) -> None:
    with st.expander("管理本地K线数据"):
        st.caption("先预览迁移计划，再执行复制或移动；支持 parquet、csv、xlsx、xls 文件。")
        source_type = st.selectbox("来源类型", ["文件夹", "文件"], key="archive_source_type")
        if source_type == "文件夹":
            source_path = _render_directory_picker("来源文件夹", data_root, "archive_source_dir")
        else:
            source_path = _render_file_picker("来源文件", data_root, "archive_source_file", KLINE_DATA_FILE_TYPES)
        destination_path = _render_directory_picker("目标文件夹", data_root, "archive_destination_dir")
        mode_label = st.selectbox("迁移方式", ["复制", "移动"], key="archive_migration_mode")
        overwrite = st.checkbox("允许覆盖同名文件", value=False, key="archive_overwrite")
        mode = "move" if mode_label == "移动" else "copy"

        preview_col, execute_col = st.columns(2)
        if preview_col.button("预览迁移计划", key="archive_preview"):
            try:
                plan = plan_kline_migration(source_path, destination_path, overwrite=overwrite)
            except Exception as exc:  # noqa: BLE001
                st.error(f"无法生成迁移计划：{exc}")
                return
            st.session_state["archive_migration_plan"] = plan

        plan = st.session_state.get("archive_migration_plan")
        if isinstance(plan, pd.DataFrame):
            if plan.empty:
                st.info("来源路径下没有可迁移的 K 线数据文件。")
            else:
                ready_count = int((plan["status"] == "ready").sum())
                exists_count = int((plan["status"] == "exists").sum())
                st.info(f"计划文件 {len(plan):,} 个；可执行 {ready_count:,} 个；同名已存在 {exists_count:,} 个。")
                st.dataframe(_centered(_format_migration_status(plan)), use_container_width=True, hide_index=True)

        if not execute_col.button("执行迁移", key="archive_execute"):
            return
        try:
            result = migrate_kline_data(source_path, destination_path, mode=mode, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            st.error(f"K 线数据迁移失败：{exc}")
            return
        st.cache_data.clear()
        st.session_state["archive_migration_plan"] = result
        failed_count = int((result["status"] == "failed").sum()) if not result.empty else 0
        if failed_count:
            st.error(f"迁移完成，但有 {failed_count:,} 个文件失败。")
        else:
            st.success("K 线数据迁移完成。")
        st.dataframe(_centered(_format_migration_status(result)), use_container_width=True, hide_index=True)


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


def _symbol_data_hint(symbol: str, *, data_root: str | Path, timeframe: str, adjust: str) -> str:
    normalized = normalize_symbol(symbol)
    if not normalized or "." not in normalized:
        return ""
    code = normalized.split(".", 1)[0]
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    alternatives = [
        candidate
        for suffix in ("SH", "SZ", "BJ")
        if (candidate := f"{code}.{suffix}") != normalized and (root / f"{candidate}.parquet").exists()
    ]
    if not alternatives:
        return ""
    return (
        f"当前输入会解析为 {normalized}，但本地存在 {', '.join(alternatives)}。"
        "如要查看指数或指定市场标的，请输入完整后缀代码。"
    )


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


@st.cache_data(show_spinner=False)
def _cached_review_constituents(kind: str, name: str) -> list[str]:
    text = str(name or "").strip()
    if not text:
        return []
    if kind == "industry":
        return fetch_industry_constituents(text)
    if kind == "concept":
        return fetch_concept_constituents(text)
    raise ValueError(f"未知板块类型：{kind}")


@st.cache_data(show_spinner=False)
def _cached_tdx_etf_index(provider: str) -> pd.DataFrame:
    return fetch_tdx_etf_index(tqcenter_path=provider)


def _review_auto_tdx_etf_proxies(
    provider: str,
    *,
    industry_name: str,
    concept_name: str,
) -> tuple[list[str], dict[str, str], pd.DataFrame, str]:
    queries = _tdx_etf_queries(industry_name, concept_name)
    if not queries:
        return [], {}, pd.DataFrame(columns=["query", "symbol", "name", "amount", "category"]), ""
    try:
        index = _cached_tdx_etf_index(provider)
    except Exception as exc:  # noqa: BLE001
        return [], {}, pd.DataFrame(columns=["query", "symbol", "name", "amount", "category"]), f"TDX ETF 索引读取失败：{exc}"
    matches = search_tdx_etf_index(index, queries)
    if matches.empty:
        return [], {}, matches, f"TDX ETF 索引没有匹配到：{', '.join(queries)}。"
    symbols = unique_symbols(matches["symbol"].dropna().astype(str).tolist())
    names = {
        normalize_symbol(row["symbol"]): str(row["name"]).strip()
        for _, row in matches.dropna(subset=["symbol"]).iterrows()
        if str(row.get("name", "")).strip()
    }
    return symbols, names, matches, ""


def _tdx_etf_queries(industry_name: str, concept_name: str) -> list[str]:
    return [text for text in dict.fromkeys([str(industry_name or "").strip(), str(concept_name or "").strip()]) if text]


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


def _set_review_quick_window(
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
    start, end, message = _review_quick_window_feedback(bars, symbol, window_size)
    st.session_state["review_start_date"] = start
    st.session_state["review_end_date"] = end
    st.session_state["review_quick_message"] = message


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


def _review_quick_window_feedback(
    bars: pd.DataFrame,
    symbol: str,
    window_size: int,
    today: pd.Timestamp | None = None,
) -> tuple[date, date, str]:
    start, end, selected_count, total_count, is_sparse = _cross_section_quick_window_selection(bars, window_size, today)
    normalized = normalize_symbol(symbol)
    window_text = f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d}"
    if total_count == 0:
        return start, end, f"{normalized} 未找到本地行情，已按自然日近 {window_size} 天设置复盘区间：{window_text}。"
    if is_sparse:
        return (
            start,
            end,
            f"{normalized} 本地数据疑似不连续，近期仅有 {selected_count} 根K线，"
            f"不足近 {window_size} 根；已使用近期可用复盘区间：{window_text}。",
        )
    if selected_count < window_size:
        return start, end, f"{normalized} 本地仅有 {selected_count} 根K线，不足近 {window_size} 根；已使用全部可用区间：{window_text}。"
    return start, end, f"{normalized} 已选择近 {window_size} 根K线：{window_text}。"


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
    incremental_latest_only: bool = False,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> pd.DataFrame:
    normalized = unique_symbols(symbols)
    if not normalized:
        return pd.DataFrame(columns=["symbol", "status", "rows", "new_rows", "message"])
    rows: list[dict[str, object]] = []
    total = len(normalized)
    before = (
        plan_incremental_downloads(
            symbols=normalized,
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
        )
        if incremental_latest_only
        else data_check(
            symbols=normalized,
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
        )
    )
    before_rows = _frame_rows_by_symbol(before)
    download_groups: dict[str, list[str]] = {}
    for symbol in normalized:
        check_row = before_rows.get(symbol)
        download_start = start
        if incremental_latest_only:
            if check_row is None or not bool(check_row.get("download_required", False)):
                continue
            download_start = str(check_row.get("download_start") or start)
        elif check_row is not None:
            download_start = _repair_partial_download_start(
                check_row,
                data_root=data_root,
                timeframe=timeframe,
                adjust=adjust,
                requested_start=start,
            )
        download_groups.setdefault(download_start, []).append(symbol)

    completed = 0
    for download_start, group_symbols in download_groups.items():
        for batch_symbols in _batched_symbols(group_symbols, DOWNLOAD_BATCH_SIZE):
            if progress_callback is not None:
                progress_callback(completed, total, _progress_symbol_label(batch_symbols), "running")
            result = update_local_bars(
                symbols=batch_symbols,
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
                symbols=batch_symbols,
                data_root=data_root,
                timeframe=timeframe,
                adjust=adjust,
                start=download_start if incremental_latest_only else start,
                end=end,
            )
            for row in _merge_download_check_rows(batch_symbols, result, checked):
                rows.append(row)
                completed += 1
                status = str(row.get("status", "unknown"))
                if progress_callback is not None:
                    progress_callback(completed, total, str(row.get("symbol", "")), status)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "status", "rows", "start", "end", "message"])


def _create_download_job(
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
    batch_size: int = DOWNLOAD_BATCH_SIZE,
    incremental_latest_only: bool = False,
) -> dict[str, object]:
    normalized = unique_symbols(symbols)
    return {
        "symbols": normalized,
        "timeframe": timeframe,
        "adjust": adjust,
        "start": start,
        "end": end,
        "trend_repo": trend_repo,
        "data_root": data_root,
        "provider": provider,
        "download_engine": download_engine,
        "batch_size": batch_size,
        "incremental_latest_only": incremental_latest_only,
        "cursor": 0,
        "rows": [],
        "status": "running" if normalized else "completed",
    }


def _set_download_job_status(job: dict[str, object], status: str) -> None:
    if status not in DOWNLOAD_JOB_STATUSES:
        raise ValueError(f"未知下载任务状态：{status}")
    job["status"] = status


def _run_download_job_step(
    job: dict[str, object],
    *,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> pd.DataFrame:
    if job.get("status") != "running":
        return pd.DataFrame()
    symbols = list(job.get("symbols", []))
    total = len(symbols)
    cursor = int(job.get("cursor", 0))
    batch_size = int(job.get("batch_size", DOWNLOAD_BATCH_SIZE))
    if batch_size < 1:
        raise ValueError("batch_size 至少需要 1。")
    batch_symbols = symbols[cursor : cursor + batch_size]
    if not batch_symbols:
        job["status"] = "completed"
        return pd.DataFrame()

    def report_progress(completed: int, _batch_total: int, symbol: str, status: str) -> None:
        if progress_callback is not None:
            progress_callback(cursor + completed, total, symbol, status)

    result = _download_symbols_with_progress(
        symbols=batch_symbols,
        timeframe=str(job["timeframe"]),
        adjust=str(job["adjust"]),
        start=str(job["start"]),
        end=str(job["end"]),
        trend_repo=Path(job["trend_repo"]),
        data_root=Path(job["data_root"]),
        provider=str(job["provider"]),
        download_engine=str(job["download_engine"]),
        incremental_latest_only=bool(job.get("incremental_latest_only", False)),
        progress_callback=report_progress,
    )
    rows = list(job.get("rows", []))
    rows.extend(result.to_dict("records"))
    job["rows"] = rows
    job["cursor"] = min(cursor + len(batch_symbols), len(symbols))
    if int(job["cursor"]) >= len(symbols):
        job["status"] = "completed"
    return result


def _download_job_result_frame(job: dict[str, object]) -> pd.DataFrame:
    rows = list(job.get("rows", []))
    if not rows:
        return pd.DataFrame(columns=["symbol", "status", "rows", "start", "end", "message"])
    return pd.DataFrame(rows)


def _download_job_summary(job: dict[str, object]) -> dict[str, object]:
    total = len(list(job.get("symbols", [])))
    completed = min(int(job.get("cursor", 0)), total)
    remaining = max(total - completed, 0)
    result = _download_job_result_frame(job)
    statuses = result["status"].astype(str) if "status" in result.columns else pd.Series(dtype=str)
    failed = int((statuses == "failed").sum())
    uncovered = int(statuses.isin(DOWNLOAD_REQUIRED_STATUSES).sum())
    status = str(job.get("status", ""))
    batch_size = max(1, int(job.get("batch_size", DOWNLOAD_BATCH_SIZE)))
    if remaining <= 0:
        batch_label = "无剩余批次"
    else:
        start_index = completed + 1
        end_index = min(completed + batch_size, total)
        prefix = "当前批" if status == "running" else "下一批"
        batch_label = f"{prefix}：第 {start_index}-{end_index} / {total} 个"
    return {
        "total": total,
        "completed": completed,
        "remaining": remaining,
        "failed": failed,
        "uncovered": uncovered,
        "status_label": DOWNLOAD_JOB_STATUS_LABELS.get(status, status or "未开始"),
        "batch_label": batch_label,
    }


def _download_progress_text(completed: int, total: int, symbol: str, row_status: str) -> str:
    if row_status == "running":
        index = min(completed + 1, total) if total else completed
        return f"正在下载第 {index}/{total} 个：{symbol}"
    if row_status == "failed":
        return f"第 {completed}/{total} 个失败：{symbol}"
    return f"已完成第 {completed}/{total} 个：{symbol}"


def _render_download_job(job_key: str, *, target_symbol: str = "") -> pd.DataFrame:
    job = st.session_state.get(job_key)
    if not isinstance(job, dict):
        return pd.DataFrame()
    status = str(job.get("status", ""))
    summary = _download_job_summary(job)
    summary_cols = st.columns(5)
    summary_cols[0].metric("任务状态", str(summary["status_label"]))
    summary_cols[1].metric("总标的", f"{int(summary['total']):,}")
    summary_cols[2].metric("已完成", f"{int(summary['completed']):,}")
    summary_cols[3].metric("失败", f"{int(summary['failed']):,}")
    summary_cols[4].metric("剩余", f"{int(summary['remaining']):,}")
    st.caption(str(summary["batch_label"]))

    button_cols = st.columns([1, 1, 3])
    if status == "running":
        if button_cols[0].button("暂停下载", key=f"{job_key}_pause"):
            _set_download_job_status(job, "paused")
            st.rerun()
    elif status == "paused":
        if button_cols[0].button("继续下载", key=f"{job_key}_resume"):
            _set_download_job_status(job, "running")
            st.rerun()
    if button_cols[1].button("清除任务", key=f"{job_key}_clear"):
        st.session_state.pop(job_key, None)
        st.rerun()

    total = int(summary["total"])
    cursor = int(summary["completed"])
    ratio = cursor / total if total else 1.0
    progress_bar = st.progress(ratio, text=f"{cursor}/{total} {summary['status_label']}")
    progress_text = st.empty()

    if status == "running":
        def report_progress(completed: int, total_count: int, symbol: str, row_status: str) -> None:
            step_ratio = completed / total_count if total_count else 1.0
            progress_bar.progress(step_ratio, text=_download_progress_text(completed, total_count, symbol, row_status))
            progress_text.caption(f"当前状态：{_download_progress_text(completed, total_count, symbol, row_status)}")

        with st.spinner("正在下载下一批行情..."):
            _run_download_job_step(job, progress_callback=report_progress)
        st.cache_data.clear()
        summary = _download_job_summary(job)
        total = int(summary["total"])
        cursor = int(summary["completed"])
        progress_bar.progress(cursor / total if total else 1.0, text=f"{cursor}/{total} {summary['status_label']}")
        if job.get("status") in {"running", "completed"}:
            st.rerun()

    result = _download_job_result_frame(job)
    if not result.empty:
        display = _pin_symbol_row(result, target_symbol) if target_symbol else result
        st.dataframe(_centered(_format_data_check_status(display)), use_container_width=True, hide_index=True)
    if job.get("status") == "paused":
        st.info("下载已暂停。当前批次已结束，点击继续下载会从下一批接着跑。")
    elif job.get("status") == "completed":
        final_summary = _download_job_summary(job)
        if int(final_summary["failed"]) > 0 or int(final_summary["uncovered"]) > 0:
            st.warning(
                f"下载任务已结束：失败 {int(final_summary['failed']):,} 个，"
                f"仍未覆盖 {int(final_summary['uncovered']):,} 个。请在上方表格查看原因。"
            )
        else:
            st.success("下载任务已完成，所选标的已覆盖。")
    return result


def _batched_symbols(symbols: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError("batch_size 至少需要 1。")
    return [symbols[index : index + batch_size] for index in range(0, len(symbols), batch_size)]


def _frame_rows_by_symbol(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "symbol" not in frame.columns:
        return {}
    return {
        normalize_symbol(str(row["symbol"])): row
        for _, row in frame.iterrows()
    }


def _progress_symbol_label(symbols: list[str]) -> str:
    if len(symbols) == 1:
        return symbols[0]
    return f"{symbols[0]} 等 {len(symbols)} 个"


def _merge_download_check_rows(
    symbols: list[str],
    download_result: pd.DataFrame,
    checked: pd.DataFrame,
) -> list[dict[str, object]]:
    download_rows = _frame_rows_by_symbol(download_result)
    checked_rows = _frame_rows_by_symbol(checked)
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        result_row = download_rows.get(symbol)
        check_row = checked_rows.get(symbol)
        if check_row is None:
            row = result_row.to_dict() if result_row is not None else {"symbol": symbol, "status": "unknown"}
            rows.append(row)
            continue
        row = check_row.to_dict()
        if result_row is not None and str(result_row.get("status", "")) == "failed":
            row = result_row.to_dict()
        elif result_row is not None and str(row.get("status", "")) != "available":
            message = str(row.get("message", ""))
            row["message"] = f"下载命令执行后仍未完整覆盖；{message}".rstrip("；")
        rows.append(row)
    return rows


def _symbols_requiring_download(check: pd.DataFrame) -> list[str]:
    if check.empty or not {"symbol", "status"}.issubset(check.columns):
        return []
    missing = check.loc[check["status"].astype(str).isin(DOWNLOAD_REQUIRED_STATUSES), "symbol"]
    return unique_symbols(missing.astype(str).tolist())


def _prepare_full_daily_download_symbols(
    *,
    symbols: list[str] | tuple[str, ...],
    data_root: str | Path,
    adjust: str,
    start: str,
    end: str,
    skip_available: bool,
    incremental_latest_only: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    normalized = unique_symbols(symbols)
    if incremental_latest_only:
        checked = plan_incremental_downloads(
            symbols=normalized,
            data_root=Path(data_root),
            timeframe="1d",
            adjust=adjust,
            start=start,
            end=end,
        )
        if checked.empty or "download_required" not in checked.columns:
            return [], checked
        required = checked.loc[checked["download_required"].fillna(False), "symbol"]
        return unique_symbols(required.astype(str).tolist()), checked
    if not skip_available:
        return normalized, pd.DataFrame()
    checked = data_check(
        symbols=normalized,
        data_root=Path(data_root),
        timeframe="1d",
        adjust=adjust,
        start=start,
        end=end,
    )
    return _symbols_requiring_download(checked), checked


def _full_daily_available_count(checked: pd.DataFrame, *, incremental_latest_only: bool) -> int:
    if checked.empty:
        return 0
    if incremental_latest_only and "download_required" in checked.columns:
        return int((~checked["download_required"].fillna(False).astype(bool)).sum())
    if "status" not in checked.columns:
        return 0
    return int((checked["status"] == "available").sum())


def _full_daily_download_universe(
    stock_symbols: list[str] | tuple[str, ...],
    *,
    include_indexes: bool,
    extra_symbols: str,
) -> list[str]:
    return symbols_with_analysis_indexes(
        stock_symbols,
        include_indexes=include_indexes,
        extra_symbols=_split_symbol_text(extra_symbols),
    )


def _split_symbol_text(value: str) -> list[str]:
    text = str(value or "")
    for separator in ("，", "、", ";", "；", "\n", "\t", " "):
        text = text.replace(separator, ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _review_target_symbols(value: str, max_symbols: int = REVIEW_MAX_TARGET_SYMBOLS) -> tuple[list[str], str]:
    symbols = unique_symbols(_split_symbol_text(value))
    if not symbols:
        return [], "请至少输入 1 个目标代码。"
    if len(symbols) > max_symbols:
        return symbols[:max_symbols], f"最多支持 {max_symbols} 个目标代码，已保留前 {max_symbols} 个。"
    return symbols, ""


def _review_result_grid_rows(results: list[ReviewResult], columns: int = 3) -> list[list[ReviewResult]]:
    column_count = max(1, int(columns))
    return [results[index : index + column_count] for index in range(0, len(results), column_count)]


def _review_comparison_data(
    target_window: pd.DataFrame,
    direct_bars: pd.DataFrame,
    *,
    data_root: str,
    timeframe: str,
    adjust: str,
    index_symbols: list[str],
    proxy_symbols: list[str],
    industry_name: str,
    concept_name: str,
    sector_min_coverage: float,
    stock_names: dict[str, str],
) -> tuple[list[tuple[str, pd.DataFrame]], list[dict[str, object]], list[str]]:
    comparison_frames, warnings = _review_shared_comparison_frames(
        direct_bars,
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        start=str(target_window["date"].min()) if not target_window.empty else "1900-01-01",
        end=str(target_window["date"].max()) if not target_window.empty else pd.Timestamp.today().strftime("%Y-%m-%d"),
        index_symbols=index_symbols,
        proxy_symbols=proxy_symbols,
        industry_name=industry_name,
        concept_name=concept_name,
        sector_min_coverage=sector_min_coverage,
        stock_names=stock_names,
    )
    comparison_rows = [
        build_comparison_stats(target_window, frame, label)
        for label, frame in comparison_frames
    ]
    return comparison_frames, comparison_rows, warnings


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


def _review_metric_items(
    result: ReviewResult,
    comparison_frame: pd.DataFrame,
    index_symbols: list[str],
) -> list[tuple[str, str]]:
    overview = result.overview
    index_excess = _review_index_excess_text(comparison_frame, index_symbols)
    return [
        ("区间收益", _percent_text(overview.get("return"))),
        ("最大回撤", _percent_text(overview.get("max_drawdown"))),
        ("最大浮盈", _percent_text(overview.get("max_favorable"))),
        ("波动率", _percent_text(overview.get("volatility"))),
        ("上涨K线占比", _percent_text(overview.get("up_day_share"))),
        ("相对指数超额", index_excess),
    ]


def _review_index_excess_text(comparison_frame: pd.DataFrame, index_symbols: list[str]) -> str:
    if comparison_frame.empty or "标的" not in comparison_frame.columns:
        return "-"
    labels = {normalize_symbol(symbol) for symbol in index_symbols}
    values: list[float] = []
    for _, row in comparison_frame.iterrows():
        label = str(row.get("标的", ""))
        if not any(symbol in label for symbol in labels):
            continue
        value = pd.to_numeric(pd.Series([row.get("超额收益")]), errors="coerce").iloc[0]
        if pd.notna(value):
            values.append(float(value))
    return _percent_text(pd.Series(values).mean()) if values else "-"


def _format_multi_review_overview(results: list[ReviewResult], stock_names: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        overview = result.overview
        rows.append(
            {
                "代码": result.symbol,
                "股票": stock_names.get(result.symbol, ""),
                "K线数": int(overview.get("k_bars", 0) or 0),
                "区间收益": overview.get("return"),
                "最大回撤": overview.get("max_drawdown"),
                "最大浮盈": overview.get("max_favorable"),
                "波动率": overview.get("volatility"),
                "上涨K线占比": overview.get("up_day_share"),
            }
        )
    frame = pd.DataFrame(rows)
    return _format_percent_columns(frame, ["区间收益", "最大回撤", "最大浮盈", "波动率", "上涨K线占比"])


def _format_multi_review_segments(results: list[ReviewResult], stock_names: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        if result.main_segments.empty:
            continue
        frame = result.main_segments.copy()
        frame.insert(0, "股票", stock_names.get(result.symbol, ""))
        frame.insert(0, "代码", result.symbol)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["代码", "股票", "方向", "开始日期", "结束日期", "K线数", "区间收益", "最大回撤", "振幅"])
    result = pd.concat(frames, ignore_index=True)
    for column in ["开始日期", "结束日期"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    result = _format_percent_columns(
        result,
        ["区间收益", "最大回撤", "最大浮盈", "振幅", "成交额变化", "成交量变化", "相对贡献"],
    )
    result = _format_decimal_columns(result, ["起点收盘", "终点收盘"])
    columns = [
        "代码",
        "股票",
        "方向",
        "开始日期",
        "结束日期",
        "K线数",
        "区间收益",
        "最大回撤",
        "最大浮盈",
        "振幅",
        "成交额变化",
        "成交量变化",
        "相对贡献",
    ]
    return result[[column for column in columns if column in result.columns]]


def _format_multi_review_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["代码", "股票", "标的", "样本数", "目标收益", "对比收益", "超额收益", "相关性", "同步关系", "波动关系", "强弱结论"]
        )
    result = _format_review_comparisons(frame)
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
    return result[[column for column in columns if column in result.columns]]


def _format_video_script_profiles(profiles: list[dict[str, object]] | tuple[dict[str, object], ...]) -> pd.DataFrame:
    if not profiles:
        return pd.DataFrame(
            columns=[
                "代码",
                "股票",
                "YTD样本起点",
                "YTD收益",
                "YTD结论",
                "买点挑战",
                "买点位置",
                "买入后最大收盘回撤",
                "单日最大日内回撤",
                "指数",
                "指数大涨日样本",
                "指数大涨日标的均值",
                "指数大跌日样本",
                "指数大跌日标的均值",
                "指数弹性结论",
            ]
        )
    result = pd.DataFrame(profiles).copy()
    if "YTD样本起点" in result.columns:
        result["YTD样本起点"] = pd.to_datetime(result["YTD样本起点"], errors="coerce").dt.strftime("%Y-%m-%d")
    result = _format_percent_columns(
        result,
        [
            "YTD收益",
            "买点位置",
            "买入后最大收盘回撤",
            "单日最大日内回撤",
            "指数大涨日标的均值",
            "指数大涨日指数均值",
            "指数大跌日标的均值",
            "指数大跌日指数均值",
        ],
    )
    columns = [
        "代码",
        "股票",
        "YTD样本起点",
        "YTD收益",
        "YTD结论",
        "买点挑战",
        "买点位置",
        "买入后最大收盘回撤",
        "单日最大日内回撤",
        "指数",
        "指数大涨日样本",
        "指数大涨日标的均值",
        "指数大涨日指数均值",
        "指数大跌日样本",
        "指数大跌日标的均值",
        "指数大跌日指数均值",
        "指数弹性结论",
    ]
    return result[[column for column in columns if column in result.columns]]


def _format_tdx_etf_matches(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["关键词", "ETF代码", "ETF名称", "成交额", "同类键"])
    result = frame.copy()
    result = result.rename(
        columns={
            "query": "关键词",
            "symbol": "ETF代码",
            "name": "ETF名称",
            "amount": "成交额",
            "category": "同类键",
        }
    )
    if "成交额" in result.columns:
        result["成交额"] = pd.to_numeric(result["成交额"], errors="coerce").map(
            lambda value: "-" if pd.isna(value) else f"{float(value):,.0f}"
        )
    columns = ["关键词", "ETF代码", "ETF名称", "成交额", "同类键"]
    return result[[column for column in columns if column in result.columns]]


def _format_review_segments(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["方向", "开始日期", "结束日期", "K线数", "区间收益", "最大回撤", "振幅"])
    result = frame.copy()
    for column in ["开始日期", "结束日期"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    result = _format_percent_columns(
        result,
        ["区间收益", "最大回撤", "最大浮盈", "振幅", "成交额变化", "成交量变化", "相对贡献"],
    )
    result = _format_decimal_columns(result, ["起点收盘", "终点收盘"])
    columns = [
        "方向",
        "开始日期",
        "结束日期",
        "K线数",
        "区间收益",
        "最大回撤",
        "最大浮盈",
        "振幅",
        "成交额变化",
        "成交量变化",
        "相对贡献",
    ]
    return result[[column for column in columns if column in result.columns]]


def _format_review_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["标的", "样本数", "目标收益", "对比收益", "超额收益", "相关性", "同步关系", "波动关系", "强弱结论"])
    result = frame.copy()
    result = _format_percent_columns(result, ["目标收益", "对比收益", "超额收益"])
    result = _format_decimal_columns(result, ["相关性"])
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


def _cross_section_search_limit(universe_symbols: list[str] | tuple[str, ...], display_n: int) -> int:
    return max(int(display_n), len(unique_symbols(universe_symbols)))


def _display_results(frame: pd.DataFrame, display_n: int) -> pd.DataFrame:
    return frame.head(max(1, int(display_n))).copy()


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


def _format_migration_status(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "size_bytes" in result.columns:
        result["size_bytes"] = pd.to_numeric(result["size_bytes"], errors="coerce").map(_format_file_size)
    return result.rename(
        columns={
            "source": "来源",
            "destination": "目标",
            "status": "状态",
            "size_bytes": "大小",
            "message": "说明",
        }
    )


def _format_file_size(value: object) -> str:
    if pd.isna(value):
        return ""
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def _centered(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    return frame.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )


def _review_kline_chart(result: ReviewResult) -> go.Figure:
    fig = go.Figure()
    window = result.window.sort_values("date")
    if window.empty:
        fig.update_layout(title="目标K线与主要波段")
        return fig
    x_values = pd.to_datetime(window["date"]).dt.strftime("%Y-%m-%d")
    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            name=result.symbol,
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    for _, segment in result.main_segments.iterrows():
        direction = str(segment.get("方向", ""))
        color = "#16a34a" if direction in {"上涨", "反弹"} else "#dc2626"
        fig.add_vrect(
            x0=pd.Timestamp(segment["开始日期"]).strftime("%Y-%m-%d"),
            x1=pd.Timestamp(segment["结束日期"]).strftime("%Y-%m-%d"),
            fillcolor=color,
            opacity=0.11,
            line_width=0,
            annotation_text=direction,
            annotation_position="top left",
        )
    fig.update_layout(
        title="目标K线与主要波段",
        xaxis_title="日期",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        xaxis_type="category",
        hovermode="x unified",
    )
    return fig


def _review_relative_chart(target_window: pd.DataFrame, comparisons: list[tuple[str, pd.DataFrame]]) -> go.Figure:
    fig = go.Figure()
    if target_window.empty:
        fig.update_layout(title="目标 / 指数 / 板块归一化走势")
        return fig
    target_series = _normalized_chart_frame(target_window)
    fig.add_scatter(
        x=target_series["date"],
        y=target_series["close"],
        mode="lines",
        name="目标",
        line={"width": 4, "color": "#2563eb"},
    )
    palette = ["#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#64748b"]
    for index, (label, frame) in enumerate(comparisons):
        series = _normalized_chart_frame(frame)
        if series.empty:
            continue
        fig.add_scatter(
            x=series["date"],
            y=series["close"],
            mode="lines",
            name=label,
            line={"width": 2, "color": palette[index % len(palette)]},
            opacity=0.78,
        )
    fig.update_layout(
        title="目标 / 指数 / 板块归一化走势",
        xaxis_title="日期",
        yaxis_title="起点=100",
        hovermode="x unified",
    )
    return fig


def _normalized_chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "close"])
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"]).sort_values("date")
    if result.empty or result["close"].iloc[0] == 0:
        return pd.DataFrame(columns=["date", "close"])
    result["close"] = result["close"] / result["close"].iloc[0] * 100.0
    return result[["date", "close"]]


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
