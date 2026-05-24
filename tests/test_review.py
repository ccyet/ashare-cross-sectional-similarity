from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.review import (
    ReviewConfig,
    analyze_price_review,
    build_video_script_profile,
    build_comparison_stats,
    build_equal_weight_series,
    rank_review_results,
    render_multi_video_script_text,
    render_multi_review_text,
    render_review_text,
    render_video_script_text,
    render_video_script_cards_html,
)


def _bars(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "stock_code": [symbol] * len(closes),
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": list(range(100, 100 + len(closes))),
            "amount": list(range(1000, 1000 + len(closes))),
        }
    )


def test_analyze_price_review_finds_up_drawdown_and_rebound_segments() -> None:
    bars = _bars("000001.SZ", [10, 11, 12, 13, 12, 11, 10, 11, 12, 13])

    result = analyze_price_review(
        bars,
        ReviewConfig(
            symbol="000001.SZ",
            start="2024-01-01",
            end="2024-01-10",
            min_swing_return=0.10,
            min_segment_bars=2,
        ),
    )

    assert round(result.overview["return"], 4) == 0.3
    assert result.segments["方向"].tolist() == ["上涨", "回撤", "反弹"]
    assert result.segments["区间收益"].round(4).tolist() == [0.3, -0.2308, 0.3]
    assert result.main_segments["方向"].tolist() == ["上涨", "回撤", "反弹"]


def test_analyze_price_review_filters_small_swings() -> None:
    bars = _bars("000001.SZ", [10.0, 10.2, 10.1, 10.3])

    result = analyze_price_review(
        bars,
        ReviewConfig(
            symbol="000001.SZ",
            start="2024-01-01",
            end="2024-01-04",
            min_swing_return=0.05,
            min_segment_bars=2,
        ),
    )

    assert result.segments.empty
    assert "没有达到最小波段幅度" in result.warnings[0]


def test_build_comparison_stats_reports_excess_return_and_strength() -> None:
    target = _bars("000001.SZ", [10, 12, 11])
    index = _bars("000300.SH", [10, 10.5, 10.2])

    stats = build_comparison_stats(target, index, "沪深300")

    assert round(float(stats["目标收益"]), 4) == 0.1
    assert round(float(stats["对比收益"]), 4) == 0.02
    assert round(float(stats["超额收益"]), 4) == 0.08
    assert stats["同步关系"] == "同向"
    assert stats["波动关系"] == "同步跟随"
    assert stats["强弱结论"] == "目标明显强于对比对象。"


def test_build_comparison_stats_reports_inverse_relationship() -> None:
    target = _bars("000001.SZ", [10, 11, 10, 11, 10, 11])
    index = _bars("000300.SH", [20, 18, 20, 18, 20, 18])

    stats = build_comparison_stats(target, index, "沪深300")

    assert stats["波动关系"] == "反向背离"


def test_build_equal_weight_series_requires_enough_local_coverage() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12]),
            _bars("000002.SZ", [20, 22, 21]),
        ],
        ignore_index=True,
    )

    result = build_equal_weight_series(
        bars,
        ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        label="行业:测试",
        min_coverage=0.75,
    )

    assert result.frame.empty
    assert round(result.coverage, 4) == 0.5
    assert "低于阈值" in result.warning


def test_build_equal_weight_series_returns_normalized_sector_path() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12]),
            _bars("000002.SZ", [20, 22, 21]),
        ],
        ignore_index=True,
    )

    result = build_equal_weight_series(
        bars,
        ["000001.SZ", "000002.SZ"],
        label="行业:测试",
        min_coverage=0.5,
    )

    assert result.warning == ""
    assert result.frame["close"].round(2).tolist() == [100.0, 110.0, 112.5]


def test_build_equal_weight_series_normalizes_without_dataframe_apply(monkeypatch) -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12]),
            _bars("000002.SZ", [20, 22, 21]),
        ],
        ignore_index=True,
    )

    def fail_apply(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("DataFrame.apply should not be needed for equal-weight normalization")

    monkeypatch.setattr(pd.DataFrame, "apply", fail_apply)

    result = build_equal_weight_series(
        bars,
        ["000001.SZ", "000002.SZ"],
        label="行业:测试",
        min_coverage=0.5,
    )

    assert result.frame["close"].round(2).tolist() == [100.0, 110.0, 112.5]


def test_render_review_text_uses_data_only_language() -> None:
    bars = _bars("000001.SZ", [10, 11, 12, 13, 12, 11, 10, 11, 12, 13])
    result = analyze_price_review(
        bars,
        ReviewConfig(
            symbol="000001.SZ",
            start="2024-01-01",
            end="2024-01-10",
            min_swing_return=0.10,
            min_segment_bars=2,
        ),
    )

    text = render_review_text(result)

    assert "研究复盘：讲证据" in text
    assert "区间收益 30.00%" in text
    assert "趋势背景" in text
    assert "关键波段证据" in text
    assert "新闻" not in text
    assert "基本面" not in text


def test_render_review_text_lists_structured_benchmark_relationship() -> None:
    bars = _bars("000001.SZ", [10, 11, 12, 13])
    result = analyze_price_review(
        bars,
        ReviewConfig(symbol="000001.SZ", start="2024-01-01", end="2024-01-04"),
    )
    comparisons = pd.DataFrame([build_comparison_stats(bars, _bars("000300.SH", [20, 22, 24, 26]), "沪深300")])

    text = render_review_text(result, comparisons)

    assert "对标关系" in text
    assert "沪深300" in text
    assert "同步跟随" in text
    assert "趋势背景" in text
    assert "相对强度" in text
    assert "交易难度" in text


def test_render_multi_review_text_summarizes_symbols_and_benchmark_relationships() -> None:
    first = analyze_price_review(
        _bars("000001.SZ", [10, 11, 12]),
        ReviewConfig(symbol="000001.SZ", start="2024-01-01", end="2024-01-03"),
    )
    second = analyze_price_review(
        _bars("600519.SH", [20, 19, 21]),
        ReviewConfig(symbol="600519.SH", start="2024-01-01", end="2024-01-03"),
    )
    comparisons = pd.DataFrame(
        [
            {"代码": "000001.SZ", **build_comparison_stats(first.window, _bars("000300.SH", [10, 10.5, 11]), "沪深300")},
            {"代码": "600519.SH", **build_comparison_stats(second.window, _bars("000300.SH", [10, 10.2, 10.4]), "沪深300")},
        ]
    )

    text = render_multi_review_text([first, second], comparisons)

    assert "研究复盘：讲证据" in text
    assert "000001.SZ" in text
    assert "600519.SH" in text
    assert "排序锐评" in text
    assert "对标关系" in text


def test_render_multi_review_text_accepts_comparison_table_without_symbol_column() -> None:
    result = analyze_price_review(
        _bars("000001.SZ", [10, 11, 12]),
        ReviewConfig(symbol="000001.SZ", start="2024-01-01", end="2024-01-03"),
    )
    comparisons = pd.DataFrame([build_comparison_stats(result.window, _bars("000300.SH", [10, 10.5, 11]), "沪深300")])

    text = render_multi_review_text([result], comparisons)

    assert "研究复盘：讲证据" in text
    assert "沪深300" in text


def test_rank_review_results_orders_by_strength_not_input_order() -> None:
    high_return_high_drawdown = analyze_price_review(
        _bars("000001.SZ", [10, 16, 10.5, 17], start="2026-01-01"),
        ReviewConfig(symbol="000001.SZ", start="2026-01-01", end="2026-01-04"),
    )
    steadier_outperformer = analyze_price_review(
        _bars("600519.SH", [10, 11, 11.6, 12.2], start="2026-01-01"),
        ReviewConfig(symbol="600519.SH", start="2026-01-01", end="2026-01-04"),
    )
    comparisons = pd.DataFrame(
        [
            {
                "代码": "000001.SZ",
                **build_comparison_stats(
                    high_return_high_drawdown.window,
                    _bars("000300.SH", [10, 13, 15, 18], start="2026-01-01"),
                    "沪深300",
                ),
            },
            {
                "代码": "600519.SH",
                **build_comparison_stats(
                    steadier_outperformer.window,
                    _bars("000300.SH", [10, 10.2, 10.3, 10.4], start="2026-01-01"),
                    "沪深300",
                ),
            },
        ]
    )

    ranking = rank_review_results(
        [high_return_high_drawdown, steadier_outperformer],
        comparisons,
        stock_names={"000001.SZ": "高波动", "600519.SH": "稳趋势"},
    )

    assert ranking["代码"].tolist() == ["600519.SH", "000001.SZ"]
    assert ranking["排名"].tolist() == [1, 2]
    assert ranking.loc[0, "股票"] == "稳趋势"
    assert ranking.loc[0, "强弱等级"] in {"S", "A"}
    assert "交易难度" in ranking.columns
    assert "锐评档位" in ranking.columns


def test_render_multi_review_text_uses_ranked_critic_order() -> None:
    high_return_high_drawdown = analyze_price_review(
        _bars("000001.SZ", [10, 16, 10.5, 17], start="2026-01-01"),
        ReviewConfig(symbol="000001.SZ", start="2026-01-01", end="2026-01-04"),
    )
    steadier_outperformer = analyze_price_review(
        _bars("600519.SH", [10, 11, 11.6, 12.2], start="2026-01-01"),
        ReviewConfig(symbol="600519.SH", start="2026-01-01", end="2026-01-04"),
    )
    comparisons = pd.DataFrame(
        [
            {
                "代码": "000001.SZ",
                **build_comparison_stats(
                    high_return_high_drawdown.window,
                    _bars("000300.SH", [10, 13, 15, 18], start="2026-01-01"),
                    "沪深300",
                ),
            },
            {
                "代码": "600519.SH",
                **build_comparison_stats(
                    steadier_outperformer.window,
                    _bars("000300.SH", [10, 10.2, 10.3, 10.4], start="2026-01-01"),
                    "沪深300",
                ),
            },
        ]
    )

    text = render_multi_review_text(
        [high_return_high_drawdown, steadier_outperformer],
        comparisons,
        stock_names={"000001.SZ": "高波动", "600519.SH": "稳趋势"},
    )

    assert "市场总环境" in text
    assert "排序锐评" in text
    assert "关键转折点复盘" in text
    assert "明日验证" in text
    assert "第1，" in text and "稳趋势" in text
    assert "第2，" in text and "高波动" in text
    assert text.index("稳趋势") < text.index("高波动")
    assert "结局" in text
    assert "标的" not in text


def test_build_video_script_profile_measures_ytd_entry_exit_and_index_elasticity() -> None:
    ytd_bars = _bars(
        "000001.SZ",
        [10, 11, 12, 11, 13, 12, 15, 14],
        start="2026-01-01",
    )
    window = ytd_bars.iloc[3:].copy()
    result = analyze_price_review(
        window,
        ReviewConfig(symbol="000001.SZ", start="2026-01-04", end="2026-01-08"),
    )
    benchmark = _bars(
        "000300.SH",
        [10.0, 10.2, 10.0, 10.2, 10.5, 10.2, 10.4, 10.1],
        start="2026-01-01",
    )

    profile = build_video_script_profile(result, ytd_bars, benchmark, benchmark_label="沪深300")

    assert profile["代码"] == "000001.SZ"
    assert round(float(profile["YTD收益"]), 4) == 0.4
    assert profile["YTD结论"] == "年内正收益"
    assert profile["结构位置"] in {"浅回调承接", "破位修复", "深回调修复", "温和启动", "高位强势"}
    assert round(float(profile["区间最大收盘回撤"]), 4) == -0.0769
    assert round(float(profile["单日最大日内回撤"]), 4) == -0.0392
    assert int(profile["指数大涨日样本"]) >= 1
    assert int(profile["指数大跌日样本"]) >= 1
    assert "大涨日" in str(profile["指数弹性结论"])


def test_build_video_script_profile_uses_pre_entry_context_for_cross_year_range() -> None:
    bars = _bars(
        "000001.SZ",
        [10, 10.2, 10.4, 10.1, 10.3, 10.5, 10.8, 11.0],
        start="2025-12-25",
    )
    result = analyze_price_review(
        bars,
        ReviewConfig(symbol="000001.SZ", start="2025-12-29", end="2026-01-01"),
    )

    profile = build_video_script_profile(result, bars)

    assert profile["结构位置"] != "数据不足"
    assert "近20根" in str(profile["结构说明"])


def test_render_video_script_text_includes_script_structure() -> None:
    ytd_bars = _bars("000001.SZ", [10, 11, 12, 13], start="2026-01-01")
    result = analyze_price_review(
        ytd_bars,
        ReviewConfig(symbol="000001.SZ", start="2026-01-01", end="2026-01-04"),
    )
    profile = build_video_script_profile(result, ytd_bars, _bars("000300.SH", [10, 10.2, 10.1, 10.4], start="2026-01-01"))
    profile["股票"] = "测试股"

    text = render_video_script_text(profile)

    assert "视频脚本视角" in text
    assert "YTD" in text
    assert "结局" in text
    assert any(word in text for word in ["夯爆了", "人上人", "立棍单打", "刷子", "混子", "NPC", "拉完了"])
    assert "入场" not in text
    assert "买点" not in text
    assert "明日验证" not in text
    assert "标的" not in text
    assert "\n\n- " not in text
    assert "测试股" in text
    assert "000001.SZ｜测试股" not in text


def test_render_video_script_cards_html_uses_separate_highlight_cards() -> None:
    ytd_bars = _bars("000001.SZ", [10, 11, 12, 13], start="2026-01-01")
    result = analyze_price_review(
        ytd_bars,
        ReviewConfig(symbol="000001.SZ", start="2026-01-01", end="2026-01-04"),
    )
    profile = build_video_script_profile(result, ytd_bars, _bars("000300.SH", [10, 10.2, 10.1, 10.4], start="2026-01-01"))
    profile["股票"] = "测试股"

    html = render_video_script_cards_html([profile])

    assert 'data-testid="video-script-cards"' in html
    assert 'class="review-script-card positive"' in html
    assert "测试股" in html
    assert "000001.SZ" in html
    assert "今年表现" in html
    assert "锐评档位" in html
    assert "定性" in html
    assert "数据" in html
    assert "结局" in html
    assert any(word in html for word in ["夯爆了", "人上人", "立棍单打", "刷子", "混子", "NPC", "拉完了"])
    assert "入场" not in html
    assert "买点" not in html
    assert "明日验证" not in html
    assert "<li" not in html
    assert "标的" not in html


def test_render_multi_video_script_text_lists_each_symbol() -> None:
    first = analyze_price_review(
        _bars("000001.SZ", [10, 11, 12], start="2026-01-01"),
        ReviewConfig(symbol="000001.SZ", start="2026-01-01", end="2026-01-03"),
    )
    second = analyze_price_review(
        _bars("600519.SH", [20, 19, 18], start="2026-01-01"),
        ReviewConfig(symbol="600519.SH", start="2026-01-01", end="2026-01-03"),
    )
    benchmark = _bars("000300.SH", [10, 10.2, 10.0], start="2026-01-01")
    profiles = [
        build_video_script_profile(first, first.window, benchmark),
        build_video_script_profile(second, second.window, benchmark),
    ]

    text = render_multi_video_script_text(profiles)

    assert "视频脚本视角" in text
    assert "000001.SZ" in text
    assert "600519.SH" in text
    assert "结局" in text
    assert "入场" not in text
    assert "买点" not in text
    assert "标的" not in text
    assert "\n\n- " not in text
