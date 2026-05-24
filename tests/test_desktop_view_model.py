from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.desktop.view_model import (
    CandlestickSeries,
    ChartHighlight,
    cross_section_candlestick_series,
    cross_section_stat_frames,
    cross_section_score_bars,
    data_status_bars,
    data_status_frame,
    history_candlestick_series,
    history_line_series,
    history_stat_frames,
    overview_modules,
    review_script_profile_cards,
    review_script_profile_frame,
    review_candlestick_series,
    review_comparisons_frame,
    review_critique_cards,
    review_overview_frame,
    review_ranking_frame,
    review_segments_frame,
    review_line_series,
    review_text,
    size_spread_frame,
    size_spread_line_series,
    size_spread_window_stats,
    visible_data_fields,
)
from ashare_cross_section_similarity.history import HistorySearchConfig, search_history
from ashare_cross_section_similarity.review import ReviewConfig, analyze_price_review, build_comparison_stats, build_video_script_profile
from ashare_cross_section_similarity.similarity import CrossSectionSearchConfig, search_cross_section


def _bars(symbol: str, closes: list[float], *, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "stock_code": symbol,
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [1000 + index for index in range(len(closes))],
            "amount": [10_000 + index * 100 for index in range(len(closes))],
        }
    )


def test_history_line_series_includes_current_and_matched_windows() -> None:
    result = search_history(
        _bars("000001.SZ", [10, 11, 12, 11, 13, 20, 19, 18, 17, 16, 30, 33, 36, 33, 39]),
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(1,),
            top_n=1,
            exclusion_bars=0,
            nearby_gap_days=0,
        ),
    )

    series = history_line_series(result)

    assert [item.label for item in series] == ["当前窗口", "相似 1"]
    assert series[0].values[0] == 100.0
    assert series[1].values[0] == 100.0


def test_history_candlestick_series_includes_current_and_matched_windows() -> None:
    bars = _bars("000001.SZ", [10, 11, 12, 11, 13, 20, 19, 18, 17, 16, 30, 33, 36, 33, 39, 40])
    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(1,),
            top_n=1,
            exclusion_bars=0,
            nearby_gap_days=0,
        ),
    )

    series = history_candlestick_series(result, bars, forward_bars=1)

    assert [item.label.split("（")[0] for item in series] == ["当前窗口", "相似 1"]
    assert series[0].rows[-1].date == pd.Timestamp("2024-01-16")
    assert isinstance(series[0].highlights[0], ChartHighlight)
    assert series[0].highlights[0].label == "指定区间"
    assert series[0].highlights[0].start == pd.Timestamp("2024-01-11")
    assert series[0].highlights[0].end == pd.Timestamp("2024-01-15")
    assert 0.25 <= series[0].highlights[0].opacity <= 0.5


def test_history_stat_frames_expose_forward_and_bucket_tables() -> None:
    result = search_history(
        _bars("000001.SZ", [10, 11, 12, 11, 13, 20, 19, 18, 17, 16, 30, 33, 36, 33, 39, 40]),
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(1,),
            top_n=2,
            exclusion_bars=0,
            nearby_gap_days=0,
        ),
    )

    sections = dict(history_stat_frames(result))

    assert list(sections) == ["后验观察统计", "相似度分层表现"]
    assert sections["后验观察统计"].iloc[0]["观察窗口"] == "后1根"
    assert "胜率" in sections["后验观察统计"].columns
    assert sections["相似度分层表现"].iloc[0]["分层"] == "Top3"


def test_size_spread_helpers_expose_chart_series_and_window_stats() -> None:
    spread = size_spread_frame(
        pd.concat(
            [
                _bars("000852.SH", [1000, 1100, 1050, 1200]),
                _bars("000300.SH", [1000, 1010, 1100, 1150]),
            ],
            ignore_index=True,
        ),
        start="2024-01-01",
    )
    current = _bars("399006.SZ", [10, 11, 12]).iloc[1:3]
    historical = _bars("399006.SZ", [10, 11, 12, 13]).iloc[2:4]

    series = size_spread_line_series(spread)
    stats = size_spread_window_stats(spread, current, [historical])

    assert spread["大小盘价差率"].round(4).tolist() == [0.0, 0.09, -0.05, 0.05]
    assert series[0].label == "大小盘价差率"
    assert tuple(round(value, 4) for value in series[0].values) == (0.0, 0.09, -0.05, 0.05)
    assert stats.loc[0, "窗口"] == "当前窗口"
    assert round(stats.loc[0, "区间变化"], 4) == -0.14


def test_cross_section_score_bars_uses_top_similarity_scores() -> None:
    bars = pd.concat(
        [
            _bars("300750.SZ", [10, 11, 12, 13, 14]),
            _bars("000001.SZ", [20, 22, 24, 26, 28]),
            _bars("600519.SH", [30, 29, 28, 27, 26]),
        ],
        ignore_index=True,
    )
    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="300750.SZ",
            universe_symbols=("000001.SZ", "600519.SH"),
            start="2024-01-01",
            end="2024-01-05",
            top_n=2,
        ),
    )

    score_bars = cross_section_score_bars(result)

    assert score_bars[0].label == "000001.SZ"
    assert score_bars[0].value > score_bars[1].value


def test_cross_section_candlestick_series_uses_target_and_top_matches() -> None:
    bars = pd.concat(
        [
            _bars("300750.SZ", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            _bars("000001.SZ", [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]),
            _bars("600519.SH", [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19]),
        ],
        ignore_index=True,
    )
    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="300750.SZ",
            universe_symbols=("000001.SZ", "600519.SH"),
            start="2024-01-01",
            end="2024-01-05",
            top_n=2,
        ),
    )

    series = cross_section_candlestick_series(result, bars, max_matches=1, forward_bars=1)

    assert series[0].label.startswith("300750.SZ（目标）")
    assert series[1].label.startswith("000001.SZ")
    assert series[0].rows[-1].date == pd.Timestamp("2024-01-06")
    assert series[0].highlights[0].label == "指定区间"
    assert series[0].highlights[0].start == pd.Timestamp("2024-01-01")
    assert series[0].highlights[0].end == pd.Timestamp("2024-01-05")


def test_cross_section_stat_frames_expose_forward_bucket_and_skipped_tables() -> None:
    bars = pd.concat(
        [
            _bars("300750.SZ", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            _bars("000001.SZ", [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]),
            _bars("600519.SH", [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19]),
        ],
        ignore_index=True,
    )
    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="300750.SZ",
            universe_symbols=("000001.SZ", "600519.SH", "000002.SZ"),
            start="2024-01-01",
            end="2024-01-05",
            top_n=2,
        ),
    )

    sections = dict(cross_section_stat_frames(result))

    assert list(sections) == ["后验观察统计", "相似度分层表现", "跳过样本"]
    assert sections["后验观察统计"].iloc[0]["观察窗口"].startswith("后")
    assert sections["相似度分层表现"].iloc[0]["分层"] == "Top3"
    assert "symbol" in sections["跳过样本"].columns


def test_review_line_series_and_text_include_review_and_critique() -> None:
    result = analyze_price_review(
        _bars("300750.SZ", [10, 11, 12, 13, 12, 14]),
        ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-06"),
    )

    series = review_line_series(result)
    text = review_text(result)

    assert series[0].label == "300750.SZ"
    assert series[0].values[0] == 100.0
    assert "研究端排序复盘" in text
    assert "逐个锐评" in text


def test_review_candlestick_series_preserves_ohlc_rows() -> None:
    result = analyze_price_review(
        _bars("300750.SZ", [10, 11, 12, 13, 12, 14]),
        ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-06"),
    )

    series = review_candlestick_series([result])

    assert isinstance(series[0], CandlestickSeries)
    assert series[0].label == "300750.SZ"
    assert series[0].rows[0].open == 10
    assert series[0].rows[0].high == 10.2
    assert series[0].rows[0].low == 9.8
    assert series[0].rows[0].close == 10


def test_review_candlestick_series_highlights_main_segments_by_direction() -> None:
    result = analyze_price_review(
        _bars("300750.SZ", [10, 11, 12, 13, 12, 11, 10, 11, 12, 13]),
        ReviewConfig(
            symbol="300750.SZ",
            start="2024-01-01",
            end="2024-01-10",
            min_swing_return=0.10,
            min_segment_bars=2,
        ),
    )

    series = review_candlestick_series([result])
    highlights = series[0].highlights

    assert [item.label for item in highlights] == ["上涨", "回撤", "反弹"]
    assert highlights[0].color == "#ef4444"
    assert highlights[2].color == "#fb7185"
    assert "#d97706" not in {item.color for item in highlights}
    assert all(0.12 <= item.opacity <= 0.25 for item in highlights)


def test_review_critique_cards_expose_grade_and_short_copy() -> None:
    results = [
        analyze_price_review(
            _bars("300750.SZ", [10, 11, 12, 13, 12, 14]),
            ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-06"),
        ),
        analyze_price_review(
            _bars("600519.SH", [20, 19, 18, 17, 16, 15]),
            ReviewConfig(symbol="600519.SH", start="2024-01-01", end="2024-01-06"),
        ),
    ]

    cards = review_critique_cards(results)

    assert [card.symbol for card in cards] == ["300750.SZ", "600519.SH"]
    assert all(card.grade for card in cards)
    assert all(card.critique for card in cards)
    assert all(card.metrics for card in cards)


def test_review_detail_frames_expose_overview_rankings_segments_and_comparisons() -> None:
    target = analyze_price_review(
        _bars("300750.SZ", [10, 11, 12, 13, 12, 14]),
        ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-06"),
    )
    weak = analyze_price_review(
        _bars("600519.SH", [20, 19, 18, 17, 16, 15]),
        ReviewConfig(symbol="600519.SH", start="2024-01-01", end="2024-01-06"),
    )
    comparison = pd.DataFrame(
        [
            {
                **build_comparison_stats(target.window, _bars("000300.SH", [9, 9.5, 9.8, 10, 10.2, 10.5]), "000300.SH"),
                "代码": target.symbol,
            }
        ]
    )

    overview = review_overview_frame([target, weak])
    rankings = review_ranking_frame([target, weak], comparison)
    segments = review_segments_frame([target, weak])
    comparisons = review_comparisons_frame(comparison)

    assert overview.loc[0, "代码"] == "300750.SZ"
    assert {"区间收益", "最大回撤", "上涨K线占比"}.issubset(overview.columns)
    assert rankings.loc[0, "排名"] == 1
    assert "锐评结论" in rankings.columns
    assert {"代码", "方向", "区间收益"}.issubset(segments.columns)
    assert comparisons.loc[0, "标的"] == "000300.SH"


def test_review_script_profile_helpers_expose_card_and_table_content() -> None:
    result = analyze_price_review(
        _bars("300750.SZ", [10, 11, 12, 13, 12, 14]),
        ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-06"),
    )
    profile = build_video_script_profile(
        result,
        _bars("300750.SZ", [9, 10, 11, 12, 13, 12, 14], start="2023-12-31"),
        _bars("000300.SH", [10, 10.2, 10.1, 10.4, 10.3, 10.5]),
        benchmark_label="沪深300",
    )
    profile.update(
        {
            "股票": "宁德时代",
            "强弱等级": "夯爆了",
            "当前性质": "趋势启动",
            "锐评结论": "弹性足，交易难度低。",
            "关键转折点": "20日线观察",
        }
    )

    cards = review_script_profile_cards([profile])
    frame = review_script_profile_frame([profile])

    assert cards[0].title == "宁德时代（300750.SZ）"
    assert cards[0].grade == "夯爆了"
    assert cards[0].nature == "趋势启动"
    assert "弹性足" in cards[0].critique
    assert frame.loc[0, "代码"] == "300750.SZ"
    assert "结局" in frame.columns


def test_data_status_helpers_feed_flat_status_chart_and_table() -> None:
    raw = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "status": "available",
                "rows": 20,
                "requested_start": "2024-01-01",
                "requested_end": "2024-01-20",
                "local_start": "2024-01-01",
                "local_end": "2024-01-20",
                "message": "ok",
            },
            {
                "symbol": "600519.SH",
                "status": "missing_file",
                "rows": 0,
                "requested_start": "2024-01-01",
                "requested_end": "2024-01-20",
                "message": "missing",
            },
        ]
    )

    frame = data_status_frame(raw)
    bars = data_status_bars(raw)

    assert list(frame.columns) == ["symbol", "status", "rows", "请求开始", "请求结束", "本地开始", "本地结束", "message"]
    assert [(item.label, item.value) for item in bars] == [("available", 1), ("missing_file", 1)]


def test_overview_modules_expose_extensible_research_entries() -> None:
    modules = overview_modules()

    assert [item.key for item in modules] == ["history", "cross_section", "review", "data"]
    assert [item.title for item in modules] == ["历史时序", "横截面", "走势复盘", "数据管理"]
    assert [item.chart_title for item in modules] == ["走势对比", "Top 相似度", "复盘走势", "覆盖/迁移"]
    assert all(item.action_label.startswith("进入") for item in modules)


def test_visible_data_fields_keep_local_mode_simple_and_reveal_api_options() -> None:
    assert visible_data_fields("本地 parquet", "local") == (
        "data_root",
        "timeframe",
        "adjust",
        "data_mode",
    )
    assert visible_data_fields("数据 API", "akshare") == (
        "data_root",
        "timeframe",
        "adjust",
        "data_mode",
        "data_source",
        "api_url",
    )
    assert visible_data_fields("数据 API", "tdx") == (
        "data_root",
        "timeframe",
        "adjust",
        "data_mode",
        "data_source",
        "api_url",
        "tdx_path",
    )
