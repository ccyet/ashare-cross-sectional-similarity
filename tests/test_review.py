from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.review import (
    ReviewConfig,
    analyze_price_review,
    build_comparison_stats,
    build_equal_weight_series,
    render_review_text,
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
    assert stats["强弱结论"] == "目标明显强于对比标的。"


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

    assert "总体复盘" in text
    assert "区间收益 30.00%" in text
    assert "关键波段" in text
    assert "新闻" not in text
    assert "基本面" not in text
