from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ashare_cross_section_similarity.cli import _resolve_universe
from ashare_cross_section_similarity.data import load_local_bars
from ashare_cross_section_similarity.downloader import data_check, default_trend_repo, update_local_bars
from ashare_cross_section_similarity.features import normalized_close_path, z_normalize
from ashare_cross_section_similarity.history import HistorySearchConfig, search_history
from ashare_cross_section_similarity.similarity import CrossSectionSearchConfig, search_cross_section


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
        provider = st.text_input("下载源", value="", help="留空使用原 trend-backtest 配置；也可填 akshare 或 tdx。")

    history_tab, cross_section_tab = st.tabs(["历史时序相似", "横截面相似"])
    with history_tab:
        _render_history_tab(
            trend_repo=trend_repo,
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            provider=provider,
        )
    with cross_section_tab:
        _render_cross_section_tab(
            trend_repo=trend_repo,
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            provider=provider,
        )


def _render_history_tab(
    *,
    trend_repo: str,
    data_root: str,
    timeframe: str,
    adjust: str,
    provider: str,
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
                    provider=provider,
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

    args = _Args(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        target_symbol=target_symbol,
        start=start,
        end=end,
        universe_symbols=universe_symbols,
        universe_file=universe_file,
        universe_index=universe_index,
        universe_industry=universe_industry,
        universe_concept=universe_concept,
    )
    try:
        universe = _resolve_universe(args)
    except Exception as exc:  # noqa: BLE001
        st.error(f"搜索范围解析失败：{exc}")
        return
    symbols = [target_symbol, *universe]
    st.markdown("**1. 数据检查**")
    try:
        check = data_check(
            symbols=symbols,
            data_root=Path(data_root),
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"数据检查失败：{exc}")
        return
    cols = st.columns(4)
    cols[0].metric("搜索范围", f"{len(universe):,}")
    cols[1].metric("可用标的", f"{int((check['status'] == 'available').sum()):,}")
    cols[2].metric("缺文件", f"{int((check['status'] == 'missing_file').sum()):,}")
    cols[3].metric("区间缺失", f"{int((check['status'] == 'missing_window').sum()):,}")
    st.dataframe(_centered(_format_status(check.head(200))), use_container_width=True, hide_index=True)

    with st.expander("缺数据时下载或更新"):
        if st.button("下载或更新当前目标与搜索范围行情", key="cross_download"):
            with st.spinner("正在调用原 trend-backtest 更新行情..."):
                update_result = update_local_bars(
                    symbols=symbols,
                    timeframe=timeframe,
                    adjust=adjust,
                    start=start,
                    end=end,
                    trend_repo=Path(trend_repo),
                    provider=provider,
                )
            st.dataframe(_centered(_format_status(update_result)), use_container_width=True, hide_index=True)

    st.markdown("**2. 运行横截面搜索**")
    if not st.button("运行横截面搜索", type="primary", key="cross_run"):
        st.info("检查数据后，缺失则先下载；数据可用后点击运行横截面搜索。")
        return

    try:
        bars = load_local_bars(
            data_root=Path(data_root),
            timeframe=timeframe,
            adjust=adjust,
            symbols=symbols,
            start=start,
            end=end,
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

    st.markdown("**3. 搜索结果**")
    st.metric("目标窗口 K 线数", result.window_size)
    st.metric("有效结果数", len(result.results))
    if result.results.empty:
        st.warning("没有找到可用结果。请检查本地数据覆盖、搜索范围和区间设置。")
        return

    st.dataframe(_centered(_format_results(result.results)), use_container_width=True, hide_index=True)
    st.plotly_chart(_score_chart(result.results), use_container_width=True)
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


def _format_results(frame: pd.DataFrame) -> pd.DataFrame:
    return _format_percent_columns(frame.copy(), PERCENT_COLUMNS)


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


if __name__ == "__main__":
    main()
