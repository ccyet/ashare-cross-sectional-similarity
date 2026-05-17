from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ashare_cross_section_similarity.cli import _resolve_universe
from ashare_cross_section_similarity.data import load_local_bars
from ashare_cross_section_similarity.downloader import data_check, default_trend_repo, update_local_bars
from ashare_cross_section_similarity.similarity import CrossSectionSearchConfig, search_cross_section


def main() -> None:
    st.set_page_config(page_title="A股横截面相似搜集", layout="wide")
    st.title("A股横截面相似搜集")
    st.caption("选定某个个股/板块代理的一段走势，在同一时间窗口内搜索其他个股或板块代理的相似走势。")
    with st.sidebar:
        st.header("运行设置")
        trend_repo = st.text_input("原 trend-backtest 仓库", value=str(default_trend_repo()))
        data_root = st.text_input("本地行情根目录", value=str(Path(trend_repo) / "data" / "market" / "daily"))
        timeframe = st.selectbox("周期", ["1d", "30m", "15m", "5m", "1m"], index=0)
        adjust = st.text_input("复权", value="qfq")
        provider = st.text_input("下载源", value="", help="留空使用原 trend-backtest 配置；也可填 akshare 或 tdx。")
        target_symbol = st.text_input("目标代码", value="300750.SZ")
        start = st.text_input("区间开始", value="2024-01-01")
        end = st.text_input("区间结束", value="2024-03-31")
        universe_symbols = st.text_area("搜索范围代码", value="", help="逗号分隔；留空时尝试读取本地目录下全部 parquet。")
        universe_file = st.text_input("搜索范围文件", value="")
        universe_index = st.text_input("指数成分", value="", help="如 000300，需要 akshare。")
        universe_industry = st.text_input("行业板块", value="", help="如 半导体，需要 akshare。")
        universe_concept = st.text_input("概念板块", value="", help="如 融资融券，需要 akshare。")
        top_n = st.number_input("展示数量", min_value=5, max_value=100, value=20, step=5)
        min_coverage = st.slider("最小覆盖率", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
        path_weight = st.slider("走势权重", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

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
    st.caption("先确认目标和搜索范围在所选区间内是否已有本地行情；缺数据时可直接在本页下载。")
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
    st.dataframe(check.head(200), use_container_width=True, hide_index=True)

    st.markdown("**2. 数据抓取 / 更新**")
    st.caption("日线和近端分钟线使用 AkShare 写入本地 parquet；30m 长历史建议提前准备本地数据。")
    if st.button("下载或更新当前目标与搜索范围行情"):
        with st.spinner("正在抓取行情并写入本地 parquet..."):
            update_result = update_local_bars(
                symbols=symbols,
                timeframe=timeframe,
                adjust=adjust,
                start=start,
                end=end,
                trend_repo=Path(trend_repo),
                provider=provider,
            )
        st.dataframe(update_result, use_container_width=True, hide_index=True)

    st.markdown("**3. 横截面相似搜索**")
    if not st.button("运行横截面搜索", type="primary"):
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

    st.markdown("**4. 搜索结果**")
    st.metric("目标窗口 K 线数", result.window_size)
    st.metric("有效结果数", len(result.results))
    if result.results.empty:
        st.warning("没有找到可用结果。请检查本地数据覆盖、搜索范围和区间设置。")
        return

    st.dataframe(_format_results(result.results), use_container_width=True, hide_index=True)
    st.plotly_chart(_score_chart(result.results), use_container_width=True)
    if not result.skipped.empty:
        with st.expander("查看跳过的标的"):
            st.dataframe(result.skipped, use_container_width=True, hide_index=True)

    st.download_button(
        "下载 CSV",
        data=result.results.to_csv(index=False).encode("utf-8-sig"),
        file_name="cross_section_similarity.csv",
        mime="text/csv",
    )


class _Args:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def _format_results(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["综合相似度", "路径相似度", "特征相似度", "区间收益", "波动率", "最大回撤"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").map(lambda value: f"{value:.2%}")
    return result


def _score_chart(frame: pd.DataFrame) -> go.Figure:
    top = frame.head(20)
    fig = go.Figure()
    fig.add_bar(x=top["symbol"], y=top["综合相似度"], name="综合相似度")
    fig.add_bar(x=top["symbol"], y=top["路径相似度"], name="路径相似度")
    fig.update_layout(title="Top 相似标的", yaxis_tickformat=".0%", barmode="group")
    return fig


if __name__ == "__main__":
    main()
