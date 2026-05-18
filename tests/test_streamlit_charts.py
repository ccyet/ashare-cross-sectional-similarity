from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from ashare_cross_section_similarity.similarity import CrossSectionSearchResult
from streamlit_app import (
    CUSTOM_DIRECTORY_OPTION,
    _cross_section_bucket_summary,
    _cross_section_forward_summary,
    _cross_section_overview_metrics,
    _cross_section_result_metrics,
    _cross_section_price_chart,
    _cross_section_quick_window_feedback,
    _cross_section_quick_window,
    _download_symbols_with_progress,
    _format_cross_section_stats,
    _format_results,
    _forward_stats_load_end,
    _directory_choice_value,
    _directory_options,
    _kline_chart_component_height,
    _lightweight_kline_chart_html,
    _lightweight_kline_series,
    _pin_symbol_row,
    _stock_name_map_from_table,
    _symbols_requiring_download,
)


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "stock_code": [symbol] * len(closes),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1] * len(closes),
            "amount": [1] * len(closes),
        }
    )


def test_directory_options_keep_default_first_and_custom_last(tmp_path: Path) -> None:
    default_path = tmp_path / "trend-backtest"
    default_path.mkdir()
    other_path = tmp_path / "other-trend"
    other_path.mkdir()

    options = _directory_options(default_path, [other_path, default_path, tmp_path / "missing"])

    assert options == [str(default_path), str(other_path), CUSTOM_DIRECTORY_OPTION]


def test_directory_choice_value_uses_custom_fallback_when_blank() -> None:
    assert _directory_choice_value("/tmp/trend", "", "/tmp/default") == "/tmp/trend"
    assert _directory_choice_value(CUSTOM_DIRECTORY_OPTION, "/tmp/custom", "/tmp/default") == "/tmp/custom"
    assert _directory_choice_value(CUSTOM_DIRECTORY_OPTION, "", "/tmp/default") == "/tmp/default"


def test_cross_section_result_metrics_are_one_row_pair() -> None:
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-03"),
        window_size=3,
        results=pd.DataFrame({"symbol": ["000002.SZ", "000003.SZ"]}),
        skipped=pd.DataFrame(),
    )

    assert _cross_section_result_metrics(result) == [("目标窗口 K 线数", "3"), ("有效结果数", "2")]


def test_lightweight_kline_series_uses_top_six_ohlc_points() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", list(range(10, 25))),
            *[_bars(f"00000{index}.SZ", list(range(20 + index, 35 + index))) for index in range(2, 9)],
        ],
        ignore_index=True,
    )
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-03"),
        window_size=3,
        results=pd.DataFrame({"symbol": [f"00000{index}.SZ" for index in range(2, 9)]}),
        skipped=pd.DataFrame(),
    )

    series = _lightweight_kline_series(bars, result)

    assert series[0]["title"] == "000001.SZ（目标）"
    assert series[0]["data"] == [
        {"time": "2024-01-01", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
        {"time": "2024-01-02", "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0},
        {"time": "2024-01-03", "open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0},
        {"time": "2024-01-04", "open": 13.0, "high": 13.0, "low": 13.0, "close": 13.0},
        {"time": "2024-01-05", "open": 14.0, "high": 14.0, "low": 14.0, "close": 14.0},
        {"time": "2024-01-06", "open": 15.0, "high": 15.0, "low": 15.0, "close": 15.0},
        {"time": "2024-01-07", "open": 16.0, "high": 16.0, "low": 16.0, "close": 16.0},
        {"time": "2024-01-08", "open": 17.0, "high": 17.0, "low": 17.0, "close": 17.0},
        {"time": "2024-01-09", "open": 18.0, "high": 18.0, "low": 18.0, "close": 18.0},
        {"time": "2024-01-10", "open": 19.0, "high": 19.0, "low": 19.0, "close": 19.0},
        {"time": "2024-01-11", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
        {"time": "2024-01-12", "open": 21.0, "high": 21.0, "low": 21.0, "close": 21.0},
        {"time": "2024-01-13", "open": 22.0, "high": 22.0, "low": 22.0, "close": 22.0},
    ]
    assert series[1]["title"] == "000002.SZ"
    assert series[1]["data"][1] == {"time": "2024-01-02", "open": 23.0, "high": 23.0, "low": 23.0, "close": 23.0}
    assert series[0]["windowEndTime"] == "2024-01-03"
    assert series[0]["windowSize"] == 3
    assert series[0]["forwardSize"] == 10
    assert [item["title"] for item in series] == [
        "000001.SZ（目标）",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
        "000006.SZ",
        "000007.SZ",
    ]


def test_cross_section_price_chart_uses_close_line_traces() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 13]),
            _bars("000002.SZ", [20, 21, 22, 23]),
            _bars("000003.SZ", [30, 31, 32, 33]),
        ],
        ignore_index=True,
    )
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-03"),
        window_size=3,
        results=pd.DataFrame(
            {
                "symbol": ["000002.SZ", "000003.SZ"],
                "综合相似度": [0.9, 0.8],
                "路径相似度": [0.91, 0.81],
            }
        ),
        skipped=pd.DataFrame(),
    )

    fig = _cross_section_price_chart(bars, result)

    assert all(trace.type == "scatter" for trace in fig.data)
    assert all(trace.mode == "lines" for trace in fig.data)
    assert [trace.name for trace in fig.data] == ["000001.SZ（目标）", "000002.SZ", "000003.SZ"]
    assert list(fig.data[0].y) == [10, 11, 12, 13]
    assert not any(trace.type == "bar" for trace in fig.data)
    assert fig.layout.yaxis.title.text == "收盘价"
    assert fig.layout.shapes


def test_cross_section_price_chart_uses_stock_name_labels() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 13]),
            _bars("000002.SZ", [20, 21, 22, 23]),
            _bars("000003.SZ", [30, 31, 32, 33]),
        ],
        ignore_index=True,
    )
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-03"),
        window_size=3,
        results=pd.DataFrame({"symbol": ["000002.SZ", "000003.SZ"]}),
        skipped=pd.DataFrame(),
    )

    fig = _cross_section_price_chart(
        bars,
        result,
        stock_names={"000001.SZ": "平安银行", "000002.SZ": "万科A"},
    )

    assert [trace.name for trace in fig.data] == ["平安银行（000001.SZ，目标）", "万科A（000002.SZ）", "000003.SZ"]


def test_lightweight_kline_series_uses_stock_name_titles() -> None:
    bars = pd.concat([_bars("000001.SZ", [10, 11, 12, 13]), _bars("000002.SZ", [20, 21, 22, 23])], ignore_index=True)
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-03"),
        window_size=3,
        results=pd.DataFrame({"symbol": ["000002.SZ"]}),
        skipped=pd.DataFrame(),
    )

    series = _lightweight_kline_series(
        bars,
        result,
        stock_names={"000001.SZ": "平安银行", "000002.SZ": "万科A"},
    )

    assert [item["title"] for item in series] == ["平安银行（000001.SZ，目标）", "万科A（000002.SZ）"]


def test_lightweight_kline_chart_marks_window_and_forward_area() -> None:
    html = _lightweight_kline_chart_html(
        [
            {
                "title": "000001.SZ（目标）",
                "windowEndTime": "2024-01-03",
                "windowSize": 3,
                "forwardSize": 10,
                "data": [
                    {"time": "2024-01-01", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
                    {"time": "2024-01-02", "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0},
                    {"time": "2024-01-03", "open": 3.0, "high": 3.0, "low": 3.0, "close": 3.0},
                    {"time": "2024-01-04", "open": 4.0, "high": 4.0, "low": 4.0, "close": 4.0},
                ],
            }
        ]
    )

    assert "窗口结束" in html
    assert "forwardShade" in html
    assert "positionWindowDivider" in html


def test_lightweight_kline_chart_renders_inline_svg_without_external_dependency() -> None:
    html = _lightweight_kline_chart_html(
        [
            {
                "title": "000001.SZ（目标）",
                "windowEndTime": "2024-01-03",
                "windowSize": 3,
                "forwardSize": 1,
                "data": [
                    {"time": "2024-01-01", "open": 1.0, "high": 3.0, "low": 0.5, "close": 2.0},
                    {"time": "2024-01-02", "open": 2.0, "high": 2.5, "low": 1.5, "close": 1.8},
                    {"time": "2024-01-03", "open": 1.8, "high": 2.2, "low": 1.7, "close": 2.1},
                    {"time": "2024-01-04", "open": 2.1, "high": 2.4, "low": 2.0, "close": 2.3},
                ],
            }
        ]
    )

    assert "<svg" in html
    assert "data-kline-panel" in html
    assert "data-kline-candle" in html
    assert "lightweight-charts" not in html


def test_lightweight_kline_chart_empty_series_has_visible_message() -> None:
    html = _lightweight_kline_chart_html([])

    assert "没有可绘制的K线数据" in html
    assert "Powered by" not in html


def test_kline_chart_component_height_scales_with_panel_count() -> None:
    assert _kline_chart_component_height([]) == 160
    assert _kline_chart_component_height([{}] * 7) >= 1200


def test_format_results_formats_forward_returns_as_percentages() -> None:
    formatted = _format_results(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "区间开始": [pd.Timestamp("2024-01-01")],
                "区间结束": [pd.Timestamp("2024-01-05")],
                "综合相似度": [0.81234],
                "区间收益": [0.1234],
                "t_plus_3_return": [0.0567],
            }
        )
    )

    assert formatted["综合相似度"].iloc[0] == "81.23%"
    assert formatted["区间收益"].iloc[0] == "12.34%"
    assert formatted["后3根收益"].iloc[0] == "5.67%"
    assert formatted["区间开始"].iloc[0] == "2024-01-01"


def test_format_results_renames_symbol_and_inserts_stock_name() -> None:
    formatted = _format_results(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000002.SZ"],
                "区间收益": [0.1, 0.2],
            }
        ),
        {"000001.SZ": "平安银行"},
    )

    assert formatted.columns[:3].tolist() == ["代码", "股票", "区间收益"]
    assert formatted["代码"].tolist() == ["000001.SZ", "000002.SZ"]
    assert formatted["股票"].tolist() == ["平安银行", ""]


def test_stock_name_map_from_akshare_code_name_table() -> None:
    names = _stock_name_map_from_table(
        pd.DataFrame({"code": ["603186", "000001"], "name": ["华正新材", "平安银行"]}),
        ("603186.SH",),
    )

    assert names == {"603186.SH": "华正新材"}


def test_format_results_formats_feature_numbers_for_display() -> None:
    formatted = _format_results(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "路径距离": [0.123456],
                "趋势斜率": [-0.987654],
                "下跌放量占比": [0.34567],
                "量价相关": [-0.45678],
                "成交规模": [19.87654],
                "特征距离": [1.23456],
            }
        )
    )

    assert formatted["路径距离"].iloc[0] == "0.12"
    assert formatted["趋势斜率"].iloc[0] == "-0.99"
    assert formatted["下跌放量占比"].iloc[0] == "34.57%"
    assert formatted["量价相关"].iloc[0] == "-0.46"
    assert formatted["成交规模"].iloc[0] == "19.88"
    assert formatted["特征距离"].iloc[0] == "1.23"


def test_cross_section_forward_summary_measures_valid_results() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "综合相似度": [0.9, 0.8, 0.7],
            "t_plus_3_return": [0.10, -0.05, 0.20],
            "t_plus_5_return": [0.12, 0.00, None],
            "t_plus_10_return": [0.30, -0.10, 0.05],
        }
    )

    summary = _cross_section_forward_summary(frame)

    row3 = summary.loc[summary["观察窗口"] == "后3根"].iloc[0]
    assert row3["样本数"] == 3
    assert row3["平均收益"] == pd.Series([0.10, -0.05, 0.20]).mean()
    assert row3["胜率"] == 2 / 3
    assert row3["最好标的"] == "000003.SZ"
    assert row3["最差标的"] == "000002.SZ"
    assert row3["相似度-收益相关"] < 0

    row5 = summary.loc[summary["观察窗口"] == "后5根"].iloc[0]
    assert row5["样本数"] == 2
    assert row5["胜率"] == 1 / 2


def test_cross_section_bucket_summary_compares_top_buckets() -> None:
    frame = pd.DataFrame(
        {
            "symbol": [f"00000{index}.SZ" for index in range(1, 13)],
            "综合相似度": [1 - index * 0.01 for index in range(12)],
            "t_plus_3_return": [0.01 * index for index in range(12)],
            "t_plus_10_return": [0.02 * index - 0.05 for index in range(12)],
        }
    )

    summary = _cross_section_bucket_summary(frame)

    assert summary["分层"].tolist() == ["Top3", "Top6", "Top10", "全部"]
    top3 = summary.loc[summary["分层"] == "Top3"].iloc[0]
    assert top3["样本数"] == 3
    assert top3["平均综合相似度"] == pd.Series([1.0, 0.99, 0.98]).mean()
    assert top3["后3根平均收益"] == pd.Series([0.0, 0.01, 0.02]).mean()
    assert top3["后10根胜率"] == 0


def test_cross_section_overview_metrics_uses_forward_returns() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "综合相似度": [0.9, 0.8, 0.7],
            "t_plus_10_return": [0.10, -0.05, 0.20],
        }
    )

    metrics = _cross_section_overview_metrics(frame)

    assert metrics == [
        ("有效结果", "3"),
        ("平均相似度", "80.00%"),
        ("后10根胜率", "66.67%"),
        ("Top6后10根均值", "8.33%"),
    ]


def test_format_cross_section_stats_keeps_correlation_as_decimal() -> None:
    formatted = _format_cross_section_stats(
        pd.DataFrame(
            {
                "观察窗口": ["后10根"],
                "平均收益": [0.1234],
                "胜率": [2 / 3],
                "相似度-收益相关": [-0.4567],
            }
        )
    )

    assert formatted["平均收益"].iloc[0] == "12.34%"
    assert formatted["胜率"].iloc[0] == "66.67%"
    assert formatted["相似度-收益相关"].iloc[0] == "-0.46"


def test_forward_stats_load_end_extends_historical_window() -> None:
    assert _forward_stats_load_end("2024-01-01", today=pd.Timestamp("2024-02-20")) == "2024-02-15"


def test_cross_section_quick_window_uses_latest_local_trading_day() -> None:
    bars = _bars("000852.SH", [10, 11, 12, 13, 14])

    start, end = _cross_section_quick_window(bars, 3)

    assert start == pd.Timestamp("2024-01-03").date()
    assert end == pd.Timestamp("2024-01-05").date()


def test_cross_section_quick_window_falls_back_to_calendar_days() -> None:
    start, end = _cross_section_quick_window(pd.DataFrame(), 5, today=pd.Timestamp("2024-02-20"))

    assert start == pd.Timestamp("2024-02-16").date()
    assert end == pd.Timestamp("2024-02-20").date()


def test_cross_section_quick_window_feedback_reports_short_history() -> None:
    bars = _bars("688603.SH", [10, 11, 12])

    start, end, message = _cross_section_quick_window_feedback(bars, 120, "688603.sh")

    assert start == pd.Timestamp("2024-01-01").date()
    assert end == pd.Timestamp("2024-01-03").date()
    assert "688603.SH 本地仅有 3 根K线" in message
    assert "不足近 120 根" in message


def test_cross_section_quick_window_feedback_reports_sparse_recent_history() -> None:
    dates = pd.to_datetime(
        [
            *pd.date_range("2024-01-02", periods=80, freq="B").strftime("%Y-%m-%d").tolist(),
            *pd.date_range("2026-03-25", periods=40, freq="B").strftime("%Y-%m-%d").tolist(),
        ]
    )
    bars = pd.DataFrame(
        {
            "date": dates,
            "stock_code": ["300750.SZ"] * len(dates),
            "open": range(len(dates)),
            "high": range(len(dates)),
            "low": range(len(dates)),
            "close": range(len(dates)),
            "volume": [1] * len(dates),
            "amount": [1] * len(dates),
        }
    )

    start, end, message = _cross_section_quick_window_feedback(bars, 120, "300750.SZ")

    assert start.year == 2026
    assert end.year == 2026
    assert "本地数据疑似不连续" in message
    assert "不足近 120 根" in message


def test_cross_section_quick_window_feedback_reports_empty_fallback() -> None:
    start, end, message = _cross_section_quick_window_feedback(
        pd.DataFrame(),
        5,
        "688603.sh",
        today=pd.Timestamp("2024-02-20"),
    )

    assert start == pd.Timestamp("2024-02-16").date()
    assert end == pd.Timestamp("2024-02-20").date()
    assert "未找到本地行情" in message


def test_pin_symbol_row_keeps_target_visible_at_top() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000503.SZ", "000852.SH", "000543.SZ"],
            "status": ["available", "available", "missing_file"],
        }
    )

    pinned = _pin_symbol_row(frame, "000852.sh", limit=2)

    assert pinned["symbol"].tolist() == ["000852.SH", "000503.SZ"]


def test_symbols_requiring_download_skips_available() -> None:
    check = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000002.SZ"],
            "status": ["available", "missing_file", "missing_window", "read_error", "missing_file"],
        }
    )

    assert _symbols_requiring_download(check) == ["000002.SZ", "000003.SZ", "000004.SZ"]


def test_download_symbols_with_progress_downloads_each_symbol(monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], str]] = []

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        calls.append((tuple(kwargs["symbols"]), str(kwargs["end"])))
        symbol = kwargs["symbols"][0]
        return pd.DataFrame(
            [{"symbol": symbol, "status": "delegated", "rows": 0, "new_rows": 0, "message": "ok"}]
        )

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        symbol = kwargs["symbols"][0]
        if symbol == "000001.SZ":
            return pd.DataFrame(
                [{"symbol": symbol, "status": "missing_file", "rows": 0, "start": None, "end": None, "message": "本地 parquet 不存在"}]
            )
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "status": "available",
                    "rows": 20,
                    "start": pd.Timestamp("2024-01-01"),
                    "end": pd.Timestamp("2024-01-31"),
                    "message": "",
                }
            ]
        )

    monkeypatch.setattr("streamlit_app.update_local_bars", fake_update_local_bars)
    monkeypatch.setattr("streamlit_app.data_check", fake_data_check)
    progress: list[tuple[int, int, str, str]] = []

    result = _download_symbols_with_progress(
        symbols=["000001.SZ", "000001.SZ", "000002.SZ"],
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        trend_repo=Path("/tmp/trend"),
        data_root=Path("/tmp/data"),
        provider="",
        download_engine="trend",
        progress_callback=lambda completed, total, symbol, status: progress.append((completed, total, symbol, status)),
    )

    assert calls == [(("000001.SZ",), "2024-01-31"), (("000002.SZ",), "2024-01-31")]
    assert progress == [
        (0, 2, "000001.SZ", "running"),
        (1, 2, "000001.SZ", "missing_file"),
        (1, 2, "000002.SZ", "running"),
        (2, 2, "000002.SZ", "available"),
    ]
    assert_frame_equal(
        result,
        pd.DataFrame(
            [
                {"symbol": "000001.SZ", "status": "missing_file", "rows": 0, "start": None, "end": None, "message": "下载命令执行后仍缺本地 parquet；本地 parquet 不存在"},
                {
                    "symbol": "000002.SZ",
                    "status": "available",
                    "rows": 20,
                    "start": pd.Timestamp("2024-01-01"),
                    "end": pd.Timestamp("2024-01-31"),
                    "message": "",
                },
            ]
        ),
    )
