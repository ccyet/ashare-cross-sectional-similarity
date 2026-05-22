from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from ashare_cross_section_similarity.history import HistorySearchResult
from ashare_cross_section_similarity.review import ReviewConfig, analyze_price_review
from ashare_cross_section_similarity.similarity import CrossSectionSearchResult
from streamlit_app import (
    _cross_section_bucket_summary,
    _cross_section_forward_summary,
    _cross_section_load_end,
    _cross_section_overview_metrics,
    _cross_section_result_metrics,
    _cross_section_search_limit,
    _cross_section_price_chart,
    _display_results,
    _cross_section_quick_window_feedback,
    _cross_section_quick_window,
    _create_download_job,
    _date_input_args,
    _directory_picker_entries,
    _download_job_summary,
    _download_progress_text,
    _download_symbols_with_progress,
    _file_picker_entries,
    _format_cross_section_stats,
    _format_data_check_status,
    _format_results,
    _full_daily_download_universe,
    _history_bucket_summary,
    _history_forward_summary,
    _history_kline_series,
    _history_quick_window_feedback,
    _local_data_fingerprint,
    _size_spread_chart,
    _size_spread_series,
    _size_spread_window_stats,
    _date_tolerance_load_start,
    _date_range_error,
    _forward_stats_load_end,
    _kline_chart_component_height,
    _lightweight_kline_chart_html,
    _lightweight_kline_series,
    _picker_initial_directory,
    _picker_path_text,
    _pin_symbol_row,
    _pick_directory_with_system_dialog,
    _prepare_full_daily_download_symbols,
    _repair_partial_download_start,
    _review_kline_chart,
    _review_multi_comparison_rows,
    _review_quick_window_feedback,
    _review_relative_chart,
    _review_result_grid_rows,
    _review_target_symbols,
    _run_download_job_step,
    _set_download_job_status,
    _akshare_etf_index_from_table,
    _etf_name_map_from_index,
    _etf_option_formatter,
    _stock_name_map_from_table,
    _symbol_data_hint,
    _symbols_requiring_download,
    _tdx_directory_default,
    _tdx_download_selection_counts,
    _top_etf_options,
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


def test_picker_initial_directory_uses_existing_directory_or_file_parent(tmp_path: Path) -> None:
    folder = tmp_path / "trend"
    folder.mkdir()
    file_path = folder / "universe.csv"
    file_path.write_text("symbol\n000001.SZ\n")

    assert _picker_initial_directory(folder, tmp_path) == str(folder)
    assert _picker_initial_directory(file_path, tmp_path) == str(folder)
    assert _picker_initial_directory(tmp_path / "missing" / "universe.csv", folder) == str(tmp_path)


def test_picker_path_text_makes_empty_and_existing_selection_clear() -> None:
    assert _picker_path_text("") == "未选择"
    assert _picker_path_text("/tmp/data") == "/tmp/data"


def test_date_input_args_allow_long_history_dates() -> None:
    args = _date_input_args("history_start_date", pd.Timestamp("2024-03-04").date(), session_state={})

    assert args["value"] == pd.Timestamp("2024-03-04").date()
    assert args["min_value"] == pd.Timestamp("1990-01-01").date()


def test_date_input_args_preserve_existing_session_value() -> None:
    args = _date_input_args(
        "history_start_date",
        pd.Timestamp("2024-03-04").date(),
        session_state={"history_start_date": pd.Timestamp("2008-01-02").date()},
    )

    assert "value" not in args
    assert args["min_value"] == pd.Timestamp("1990-01-01").date()


def test_directory_picker_entries_lists_child_directories_only(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "universe.csv").write_text("symbol\n000001.SZ\n")

    entries = _directory_picker_entries(tmp_path)

    assert [path.name for path in entries] == ["a", "b"]


def test_pick_directory_with_system_dialog_returns_selected_folder(tmp_path: Path) -> None:
    folder = tmp_path / "trend"
    folder.mkdir()
    calls: dict[str, object] = {}

    class FakeRoot:
        def withdraw(self) -> None:
            calls["withdraw"] = True

        def attributes(self, *args: object) -> None:
            calls["attributes"] = args

        def destroy(self) -> None:
            calls["destroy"] = True

    def askdirectory(**kwargs: object) -> str:
        calls["dialog"] = kwargs
        return str(folder)

    selected, error = _pick_directory_with_system_dialog(
        "本地行情根目录",
        folder,
        tk_factory=FakeRoot,
        askdirectory=askdirectory,
    )

    assert selected == str(folder)
    assert error is None
    assert calls["withdraw"] is True
    assert calls["destroy"] is True
    assert calls["dialog"] == {
        "title": "选择本地行情根目录",
        "initialdir": str(folder),
        "mustexist": True,
    }


def test_tdx_directory_default_reuses_sidebar_selection() -> None:
    assert _tdx_directory_default("", "/Applications/Tdx/PYPlugins/user") == "/Applications/Tdx/PYPlugins/user"
    assert _tdx_directory_default("/selected/tdx", "/env/tdx") == "/selected/tdx"


def test_pick_directory_with_system_dialog_reports_open_error() -> None:
    def broken_tk() -> object:
        raise RuntimeError("no display")

    selected, error = _pick_directory_with_system_dialog(
        "本地行情根目录",
        "/tmp",
        tk_factory=broken_tk,
        askdirectory=lambda **_: "",
    )

    assert selected is None
    assert error is not None
    assert "无法打开系统文件夹选择器" in error
    assert "no display" in error


def test_file_picker_entries_filters_supported_files(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "universe.csv").write_text("symbol\n000001.SZ\n")
    (tmp_path / "prices.parquet").write_bytes(b"placeholder")
    (tmp_path / "ignore.txt").write_text("ignore")

    directories, files = _file_picker_entries(tmp_path, [("数据文件", ("*.csv", "*.parquet"))])

    assert [path.name for path in directories] == ["sub"]
    assert [path.name for path in files] == ["prices.parquet", "universe.csv"]


def test_local_data_fingerprint_changes_when_parquet_file_changes(tmp_path: Path) -> None:
    root = tmp_path / "daily" / "qfq"
    root.mkdir(parents=True)
    path = root / "000001.SZ.parquet"
    pd.DataFrame({"date": ["2024-01-01"]}).to_parquet(path, index=False)

    first = _local_data_fingerprint(tmp_path / "daily", "1d", "qfq", ("000001.SZ",))
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]}).to_parquet(path, index=False)
    second = _local_data_fingerprint(tmp_path / "daily", "1d", "qfq", ("000001.SZ",))

    assert first != second


def test_date_range_error_reports_reversed_range() -> None:
    assert _date_range_error("2026-08-11", "2026-05-15") == "区间开始不能晚于区间结束。"
    assert _date_range_error("2026-05-15", "2026-08-11") == ""


def test_format_data_check_status_separates_requested_and_local_ranges() -> None:
    formatted = _format_data_check_status(
        pd.DataFrame(
            {
                "symbol": ["000017.SZ"],
                "status": ["missing_window"],
                "rows": [0],
                "requested_start": [pd.Timestamp("2026-08-11")],
                "requested_end": [pd.Timestamp("2026-05-15")],
                "local_start": [pd.Timestamp("2021-05-17")],
                "local_end": [pd.Timestamp("2026-05-15")],
                "message": ["所选区间无行情"],
            }
        )
    )

    assert formatted.columns.tolist() == ["symbol", "status", "rows", "请求开始", "请求结束", "本地开始", "本地结束", "message"]
    assert formatted["请求开始"].iloc[0] == "2026-08-11"
    assert formatted["本地开始"].iloc[0] == "2021-05-17"


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


def test_cross_section_search_limit_uses_full_universe_not_display_count() -> None:
    assert _cross_section_search_limit(["000001.SZ", "000002.SZ", "000003.SZ"], 1) == 3
    assert _cross_section_search_limit(["000001.SZ"], 5) == 5


def test_display_results_limits_only_visible_rows() -> None:
    frame = pd.DataFrame({"symbol": ["000001.SZ", "000002.SZ", "000003.SZ"]})

    visible = _display_results(frame, 2)

    assert visible["symbol"].tolist() == ["000001.SZ", "000002.SZ"]
    assert frame["symbol"].tolist() == ["000001.SZ", "000002.SZ", "000003.SZ"]


def test_history_quick_window_feedback_uses_latest_local_bar() -> None:
    bars = _bars("000001.SZ", [10, 11, 12])

    start, end, message = _history_quick_window_feedback(bars, "000001.SZ", 3)

    assert start == pd.Timestamp("2024-01-01").date()
    assert end == pd.Timestamp("2024-01-03").date()
    assert "000001.SZ 已选择近 3 根K线" in message


def test_review_quick_window_feedback_uses_latest_local_bar() -> None:
    bars = _bars("000001.SZ", [10, 11, 12])

    start, end, message = _review_quick_window_feedback(bars, "000001.SZ", 3)

    assert start == pd.Timestamp("2024-01-01").date()
    assert end == pd.Timestamp("2024-01-03").date()
    assert "000001.SZ 已选择近 3 根K线" in message


def test_history_forward_summary_measures_return_drawdown_and_favorable() -> None:
    frame = pd.DataFrame(
        {
            "窗口开始": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "窗口结束": pd.to_datetime(["2024-01-05", "2024-02-05", "2024-03-05"]),
            "综合相似度": [0.9, 0.8, 0.7],
            "t_plus_5_return": [0.10, -0.05, 0.20],
            "t_plus_5_max_drawdown": [-0.03, -0.12, -0.02],
            "t_plus_5_max_favorable": [0.12, 0.01, 0.25],
        }
    )

    summary = _history_forward_summary(frame)

    row = summary.loc[summary["观察窗口"] == "后5根"].iloc[0]
    assert row["样本数"] == 3
    assert row["平均收益"] == pd.Series([0.10, -0.05, 0.20]).mean()
    assert row["胜率"] == 2 / 3
    assert row["平均最大回撤"] == pd.Series([-0.03, -0.12, -0.02]).mean()
    assert row["平均最大浮盈"] == pd.Series([0.12, 0.01, 0.25]).mean()
    assert row["最好窗口"] == "2024-03-01 至 2024-03-05"


def test_history_bucket_summary_compares_top_buckets() -> None:
    frame = pd.DataFrame(
        {
            "综合相似度": [0.9, 0.8, 0.7, 0.6],
            "t_plus_5_return": [0.10, -0.05, 0.20, 0.00],
        }
    )

    summary = _history_bucket_summary(frame)

    top3 = summary.loc[summary["分层"] == "Top3"].iloc[0]
    assert top3["样本数"] == 3
    assert top3["后5根平均收益"] == pd.Series([0.10, -0.05, 0.20]).mean()
    assert top3["后5根胜率"] == 2 / 3


def test_history_kline_series_uses_current_and_historical_windows() -> None:
    bars = _bars("000001.SZ", list(range(10, 30)))
    current = bars.iloc[10:15].reset_index(drop=True)
    historical = bars.iloc[0:5].reset_index(drop=True)
    result = HistorySearchResult(
        symbol="000001.SZ",
        as_of=pd.Timestamp("2024-01-15"),
        window_size=5,
        current_window=current,
        historical_windows=[historical],
        results=pd.DataFrame({"窗口开始": [historical["date"].min()], "窗口结束": [historical["date"].max()]}),
    )

    series = _history_kline_series(bars, result, forward_bars=2)

    assert [item["title"] for item in series] == [
        "当前窗口（2024-01-11 至 2024-01-15）",
        "样本1（2024-01-01 至 2024-01-05）",
    ]
    assert series[0]["windowEndTime"] == "2024-01-15"
    assert series[0]["forwardSize"] == 2
    assert series[1]["windowEndTime"] == "2024-01-05"
    assert series[1]["data"][-1]["time"] == "2024-01-07"


def test_size_spread_series_normalizes_from_first_common_trading_day() -> None:
    bars = pd.concat(
        [
            _bars("000852.SH", [1000, 1100, 1050]),
            _bars("000300.SH", [5000, 5050, 5500]),
        ],
        ignore_index=True,
    )

    spread = _size_spread_series(bars, start="2024-01-01")

    assert spread["date"].tolist() == pd.date_range("2024-01-01", periods=3, freq="D").tolist()
    assert spread["中证1000归一收益"].round(4).tolist() == [0.0, 0.1, 0.05]
    assert spread["沪深300归一收益"].round(4).tolist() == [0.0, 0.01, 0.1]
    assert spread["大小盘价差率"].round(4).tolist() == [0.0, 0.09, -0.05]


def test_size_spread_series_returns_empty_when_one_index_is_missing() -> None:
    spread = _size_spread_series(_bars("000852.SH", [1000, 1100, 1050]), start="2024-01-01")

    assert spread.empty


def test_size_spread_window_stats_measure_window_change_and_point_in_time_percentile() -> None:
    spread = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "大小盘价差率": [0.0, 0.1, -0.05, 0.2, 0.15],
        }
    )
    current = _bars("000001.SZ", [10, 11, 12]).iloc[1:4].reset_index(drop=True)
    historical = _bars("000001.SZ", [10, 11, 12]).iloc[0:2].reset_index(drop=True)

    stats = _size_spread_window_stats(spread, current, [historical])

    current_row = stats.loc[stats["窗口"] == "当前窗口"].iloc[0]
    assert current_row["区间开始"] == pd.Timestamp("2024-01-02")
    assert current_row["区间结束"] == pd.Timestamp("2024-01-03")
    assert current_row["起点价差率"] == 0.1
    assert current_row["终点价差率"] == -0.05
    assert round(current_row["区间变化"], 4) == -0.15
    assert current_row["终点历史分位"] == 1 / 3


def test_size_spread_chart_marks_current_and_historical_windows() -> None:
    spread = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "大小盘价差率": [0.0, 0.1, -0.05, 0.2, 0.15],
        }
    )
    current = _bars("000001.SZ", [10, 11, 12]).iloc[1:4].reset_index(drop=True)
    historical = _bars("000001.SZ", [10, 11, 12]).iloc[0:2].reset_index(drop=True)

    fig = _size_spread_chart(spread, current, [historical])

    assert fig.data[0].name == "大小盘价差率"
    assert list(fig.data[0].y) == [0.0, 0.1, -0.05, 0.2, 0.15]
    assert fig.layout.yaxis.tickformat == ".2%"
    assert len(fig.layout.shapes) >= 3


def test_cross_section_load_range_includes_date_tolerance_buffer() -> None:
    assert _date_tolerance_load_start("2024-01-10", 5) == "2023-12-30"
    assert _cross_section_load_end("2024-01-10", 5, today=pd.Timestamp("2024-03-10")) == "2024-03-06"


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


def test_cross_section_price_chart_uses_candidate_hit_window_start() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 13, 14, 15, 16]),
            _bars("000002.SZ", [20, 21, 22, 23, 24, 25, 26]),
        ],
        ignore_index=True,
    )
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-04"),
        window_size=3,
        results=pd.DataFrame(
            {
                "symbol": ["000002.SZ"],
                "区间开始": [pd.Timestamp("2024-01-04")],
                "区间结束": [pd.Timestamp("2024-01-06")],
            }
        ),
        skipped=pd.DataFrame(),
    )

    fig = _cross_section_price_chart(bars, result)

    assert list(fig.data[0].y) == [11, 12, 13, 14, 15, 16]
    assert list(fig.data[1].y) == [23, 24, 25, 26]


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


def test_lightweight_kline_series_uses_candidate_hit_window_dates() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 13, 14, 15, 16]),
            _bars("000002.SZ", [20, 21, 22, 23, 24, 25, 26]),
        ],
        ignore_index=True,
    )
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-04"),
        window_size=3,
        results=pd.DataFrame(
            {
                "symbol": ["000002.SZ"],
                "区间开始": [pd.Timestamp("2024-01-04")],
                "区间结束": [pd.Timestamp("2024-01-06")],
            }
        ),
        skipped=pd.DataFrame(),
    )

    series = _lightweight_kline_series(bars, result, forward_bars=1)

    assert series[0]["windowEndTime"] == "2024-01-04"
    assert series[0]["data"][0]["time"] == "2024-01-02"
    assert series[1]["windowEndTime"] == "2024-01-06"
    assert series[1]["data"][0]["time"] == "2024-01-04"


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

    assert "命中区间结束" in html
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
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in html
    assert "lightweight-charts" not in html


def test_lightweight_kline_chart_empty_series_has_visible_message() -> None:
    html = _lightweight_kline_chart_html([])

    assert "没有可绘制的K线数据" in html
    assert "Powered by" not in html


def test_review_kline_chart_marks_main_segments() -> None:
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

    fig = _review_kline_chart(result)

    assert fig.layout.title.text == "目标K线与主要波段"
    assert len(fig.layout.shapes) >= 3
    assert fig.data[0].type == "candlestick"
    assert fig.layout.xaxis.type == "category"
    assert all(isinstance(item, str) for item in fig.data[0].x)


def test_review_kline_chart_uses_stock_name_when_available() -> None:
    bars = _bars("512480.SH", [10, 11, 12, 13])
    result = analyze_price_review(
        bars,
        ReviewConfig(symbol="512480.SH", start="2024-01-01", end="2024-01-04"),
    )

    fig = _review_kline_chart(result, {"512480.SH": "半导体ETF"})

    assert fig.data[0].name == "半导体ETF（512480.SH，目标）"


def test_review_target_symbols_limits_multi_review_to_eight() -> None:
    symbols, error = _review_target_symbols(
        ",".join(f"{index:06d}.SZ" for index in range(1, 38))
    )

    assert len(symbols) == 36
    assert symbols[0] == "000001.SZ"
    assert symbols[-1] == "000036.SZ"
    assert "最多支持 36 个" in error


def test_review_result_grid_rows_uses_three_columns() -> None:
    results = [
        analyze_price_review(
            _bars(f"00000{index}.SZ", [10, 11, 12]),
            ReviewConfig(symbol=f"00000{index}.SZ", start="2024-01-01", end="2024-01-03"),
        )
        for index in range(1, 9)
    ]

    rows = _review_result_grid_rows(results)

    assert [len(row) for row in rows] == [3, 3, 2]


def test_review_multi_comparison_rows_reuses_shared_comparison_frames() -> None:
    first = analyze_price_review(
        _bars("000001.SZ", [10, 11, 12]),
        ReviewConfig(symbol="000001.SZ", start="2024-01-01", end="2024-01-03"),
    )
    second = analyze_price_review(
        _bars("600519.SH", [20, 19, 21]),
        ReviewConfig(symbol="600519.SH", start="2024-01-01", end="2024-01-03"),
    )
    comparison_frames = [("沪深300", _bars("000300.SH", [10, 10.5, 11]))]

    rows = _review_multi_comparison_rows([first, second], comparison_frames, {"000001.SZ": "平安银行"})

    assert [row["代码"] for row in rows] == ["000001.SZ", "600519.SH"]
    assert rows[0]["股票"] == "平安银行"
    assert rows[0]["标的"] == "沪深300"


def test_review_relative_chart_normalizes_target_and_comparison() -> None:
    target = _bars("000001.SZ", [10, 11, 12])
    comparison = _bars("000300.SH", [20, 20, 22])

    fig = _review_relative_chart(target, [("沪深300", comparison)])

    assert fig.layout.title.text == "目标 / 指数 / 板块归一化走势"
    assert list(fig.data[0].y) == [100.0, 110.00000000000001, 120.0]
    assert list(fig.data[1].y) == [100.0, 100.0, 110.00000000000001]


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
                "覆盖率": [1.0],
                "综合相似度": [0.81234],
                "区间收益": [0.1234],
                "t_plus_3_return": [0.0567],
            }
        )
    )

    assert formatted["综合相似度"].iloc[0] == "81.23%"
    assert formatted["覆盖率"].iloc[0] == "100.00%"
    assert formatted["区间收益"].iloc[0] == "12.34%"
    assert formatted["后3根收益"].iloc[0] == "5.67%"
    assert formatted["命中区间开始"].iloc[0] == "2024-01-01"
    assert formatted["命中区间结束"].iloc[0] == "2024-01-05"


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


def test_akshare_etf_index_from_table_resolves_review_target_etf() -> None:
    index = _akshare_etf_index_from_table(
        pd.DataFrame(
            {
                "代码": ["512480", "510300"],
                "名称": ["半导体ETF", "沪深300ETF"],
                "成交额": [10_000_000, 20_000_000],
            }
        ),
    )
    names = _etf_name_map_from_index(
        index,
        ("512480.SH", "601888.SH"),
    )

    assert names == {"512480.SH": "半导体ETF"}


def test_top_etf_options_uses_name_labels_and_keeps_largest_same_theme() -> None:
    options = _top_etf_options(
        pd.DataFrame(
            [
                {"symbol": "512480.SH", "name": "半导体ETF", "amount": 1_000_000, "category": "半导体"},
                {"symbol": "159995.SZ", "name": "半导体芯片ETF", "amount": 5_000_000, "category": "半导体"},
                {"symbol": "510300.SH", "name": "沪深300ETF", "amount": 4_000_000, "category": "沪深300"},
                {"symbol": "588000.SH", "name": "科创50ETF", "amount": 3_000_000, "category": "科创50"},
            ]
        ),
        limit=2,
    )

    assert options["symbol"].tolist() == ["159995.SZ", "510300.SH"]
    formatter = _etf_option_formatter(options)
    assert formatter("159995.SZ") == "半导体芯片ETF（159995.SZ）"


def test_akshare_top_etf_options_merges_same_theme_with_issuer_suffix() -> None:
    options = _top_etf_options(
        _akshare_etf_index_from_table(
            pd.DataFrame(
                {
                    "代码": ["512480", "159995", "510300"],
                    "名称": ["半导体ETF国联安", "半导体ETF易方达", "沪深300ETF华泰柏瑞"],
                    "成交额": [1_000_000, 5_000_000, 4_000_000],
                }
            )
        ),
        limit=10,
    )

    assert "159995.SZ" in options["symbol"].tolist()
    assert "512480.SH" not in options["symbol"].tolist()


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
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000002.SZ"],
            "status": ["available", "missing_file", "missing_window", "read_error", "partial_window", "missing_file"],
        }
    )

    assert _symbols_requiring_download(check) == ["000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]


def test_repair_partial_download_start_forces_gap_backfill_from_existing_min(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame({"date": ["2024-01-02", "2026-03-25"]}).to_parquet(qfq / "000012.SZ.parquet", index=False)
    check_row = pd.Series(
        {
            "symbol": "000012.SZ",
            "status": "partial_window",
            "start": pd.Timestamp("2026-03-25"),
            "end": pd.Timestamp("2026-05-15"),
        }
    )

    repaired = _repair_partial_download_start(
        check_row,
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        requested_start="2025-07-31",
    )

    assert repaired == "2024-01-01"


def test_download_symbols_with_progress_batches_symbols_with_same_start(monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], str]] = []

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        calls.append((tuple(kwargs["symbols"]), str(kwargs["end"])))
        return pd.DataFrame(
            [
                {"symbol": symbol, "status": "delegated", "rows": 0, "new_rows": 0, "message": "ok"}
                for symbol in kwargs["symbols"]
            ]
        )

    check_calls: list[tuple[str, ...]] = []

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        symbols = list(kwargs["symbols"])
        check_calls.append(tuple(symbols))
        if len(check_calls) == 1:
            return pd.DataFrame(
                [
                    {"symbol": symbol, "status": "missing_file", "rows": 0, "start": None, "end": None, "message": "本地 parquet 不存在"}
                    for symbol in symbols
                ]
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
                for symbol in symbols
            ]
        )

    monkeypatch.setattr("streamlit_app.update_local_bars", fake_update_local_bars)
    monkeypatch.setattr("streamlit_app.data_check", fake_data_check)
    progress: list[tuple[int, int, str, str]] = []

    result = _download_symbols_with_progress(
        symbols=["000001.SZ", "000001.SZ", "000002.SZ", "600519.SH"],
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

    assert check_calls == [
        ("000001.SZ", "000002.SZ", "600519.SH"),
        ("000001.SZ", "000002.SZ", "600519.SH"),
    ]
    assert calls == [(("000001.SZ", "000002.SZ", "600519.SH"), "2024-01-31")]
    assert progress == [
        (0, 3, "000001.SZ 等 3 个", "running"),
        (1, 3, "000001.SZ", "available"),
        (2, 3, "000002.SZ", "available"),
        (3, 3, "600519.SH", "available"),
    ]
    assert_frame_equal(
        result,
        pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "status": "available",
                    "rows": 20,
                    "start": pd.Timestamp("2024-01-01"),
                    "end": pd.Timestamp("2024-01-31"),
                    "message": "",
                },
                {
                    "symbol": "000002.SZ",
                    "status": "available",
                    "rows": 20,
                    "start": pd.Timestamp("2024-01-01"),
                    "end": pd.Timestamp("2024-01-31"),
                    "message": "",
                },
                {
                    "symbol": "600519.SH",
                    "status": "available",
                    "rows": 20,
                    "start": pd.Timestamp("2024-01-01"),
                    "end": pd.Timestamp("2024-01-31"),
                    "message": "",
                },
            ]
        ),
    )


def test_download_job_can_pause_between_batches(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        symbols = tuple(kwargs["symbols"])
        calls.append(symbols)
        return pd.DataFrame(
            [
                {"symbol": symbol, "status": "delegated", "rows": 0, "new_rows": 0, "message": "ok"}
                for symbol in symbols
            ]
        )

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
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
                for symbol in kwargs["symbols"]
            ]
        )

    monkeypatch.setattr("streamlit_app.update_local_bars", fake_update_local_bars)
    monkeypatch.setattr("streamlit_app.data_check", fake_data_check)
    job = _create_download_job(
        symbols=["000001.SZ", "000002.SZ"],
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        trend_repo=Path("/tmp/trend"),
        data_root=Path("/tmp/data"),
        provider="",
        download_engine="trend",
        batch_size=1,
    )

    _set_download_job_status(job, "paused")
    paused_result = _run_download_job_step(job)

    assert paused_result.empty
    assert calls == []
    assert job["status"] == "paused"
    assert job["cursor"] == 0

    _set_download_job_status(job, "running")
    first_result = _run_download_job_step(job)

    assert calls == [("000001.SZ",)]
    assert first_result["symbol"].tolist() == ["000001.SZ"]
    assert job["status"] == "running"
    assert job["cursor"] == 1

    _set_download_job_status(job, "paused")
    second_paused_result = _run_download_job_step(job)

    assert second_paused_result.empty
    assert calls == [("000001.SZ",)]

    _set_download_job_status(job, "running")
    second_result = _run_download_job_step(job)

    assert calls == [("000001.SZ",), ("000002.SZ",)]
    assert second_result["symbol"].tolist() == ["000002.SZ"]
    assert job["status"] == "completed"
    assert job["cursor"] == 2


def test_download_job_reports_overall_progress_across_batches(monkeypatch) -> None:
    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"symbol": symbol, "status": "delegated", "rows": 0, "new_rows": 0, "message": "ok"}
                for symbol in kwargs["symbols"]
            ]
        )

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
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
                for symbol in kwargs["symbols"]
            ]
        )

    monkeypatch.setattr("streamlit_app.update_local_bars", fake_update_local_bars)
    monkeypatch.setattr("streamlit_app.data_check", fake_data_check)
    job = _create_download_job(
        symbols=["000001.SZ", "000002.SZ", "000003.SZ"],
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        trend_repo=Path("/tmp/trend"),
        data_root=Path("/tmp/data"),
        provider="",
        download_engine="trend",
        batch_size=2,
    )
    progress: list[tuple[int, int, str, str]] = []

    _run_download_job_step(
        job,
        progress_callback=lambda completed, total, symbol, status: progress.append((completed, total, symbol, status)),
    )

    assert progress == [
        (0, 3, "000001.SZ 等 2 个", "running"),
        (1, 3, "000001.SZ", "available"),
        (2, 3, "000002.SZ", "available"),
    ]
    assert job["cursor"] == 2

    progress.clear()
    _run_download_job_step(
        job,
        progress_callback=lambda completed, total, symbol, status: progress.append((completed, total, symbol, status)),
    )

    assert progress == [
        (2, 3, "000003.SZ", "running"),
        (3, 3, "000003.SZ", "available"),
    ]
    assert job["status"] == "completed"


def test_download_job_summary_uses_user_facing_counts() -> None:
    job = _create_download_job(
        symbols=["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"],
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        trend_repo=Path("/tmp/trend"),
        data_root=Path("/tmp/data"),
        provider="",
        download_engine="tdx",
        batch_size=2,
    )
    job["cursor"] = 2
    job["rows"] = [
        {"symbol": "000001.SZ", "status": "available"},
        {"symbol": "000002.SZ", "status": "failed"},
    ]

    assert _download_job_summary(job) == {
        "total": 5,
        "completed": 2,
        "remaining": 3,
        "failed": 1,
        "uncovered": 0,
        "status_label": "下载中",
        "batch_label": "当前批：第 3-4 / 5 个",
    }


def test_download_progress_text_describes_current_symbol_and_result() -> None:
    assert _download_progress_text(2, 5, "000003.SZ", "running") == "正在下载第 3/5 个：000003.SZ"
    assert _download_progress_text(3, 5, "000003.SZ", "available") == "已完成第 3/5 个：000003.SZ"
    assert _download_progress_text(3, 5, "000003.SZ", "failed") == "第 3/5 个失败：000003.SZ"


def test_prepare_full_daily_download_symbols_skips_available_daily_bars(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {"symbol": "000001.SZ", "status": "available"},
                {"symbol": "000002.SZ", "status": "missing_file"},
                {"symbol": "000003.SZ", "status": "partial_window"},
                {"symbol": "000004.SZ", "status": "read_error"},
            ]
        )

    monkeypatch.setattr("streamlit_app.data_check", fake_data_check)

    download_symbols, checked = _prepare_full_daily_download_symbols(
        symbols=["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        data_root=Path("/tmp/data"),
        adjust="qfq",
        start="1990-01-01",
        end="2026-05-19",
        skip_available=True,
    )

    assert download_symbols == ["000002.SZ", "000003.SZ", "000004.SZ"]
    assert checked["symbol"].tolist() == ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    assert calls == [
        {
            "symbols": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "data_root": Path("/tmp/data"),
            "timeframe": "1d",
            "adjust": "qfq",
            "start": "1990-01-01",
            "end": "2026-05-19",
        }
    ]


def test_prepare_full_daily_download_symbols_incremental_uses_latest_plan(monkeypatch) -> None:
    def fake_plan_incremental_downloads(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "status": "partial_window",
                    "download_required": False,
                    "download_start": "",
                },
                {
                    "symbol": "000002.SZ",
                    "status": "partial_window",
                    "download_required": True,
                    "download_start": "2026-05-16",
                },
            ]
        )

    monkeypatch.setattr("streamlit_app.plan_incremental_downloads", fake_plan_incremental_downloads)

    download_symbols, checked = _prepare_full_daily_download_symbols(
        symbols=["000001.SZ", "000002.SZ"],
        data_root=Path("/tmp/data"),
        adjust="qfq",
        start="1990-01-01",
        end="2026-05-22",
        skip_available=True,
        incremental_latest_only=True,
    )

    assert download_symbols == ["000002.SZ"]
    assert checked["download_start"].tolist() == ["", "2026-05-16"]


def test_prepare_full_daily_download_symbols_can_force_all_symbols() -> None:
    download_symbols, checked = _prepare_full_daily_download_symbols(
        symbols=["000001", "600519.SH", "000001.SZ"],
        data_root=Path("/tmp/data"),
        adjust="qfq",
        start="1990-01-01",
        end="2026-05-19",
        skip_available=False,
    )

    assert download_symbols == ["000001.SZ", "600519.SH"]
    assert checked.empty


def test_full_daily_download_universe_includes_common_indexes() -> None:
    symbols = _full_daily_download_universe(
        ["000001.SZ", "600519.SH"],
        include_indexes=True,
        extra_symbols="399006, 000300.SH",
    )

    assert symbols[:2] == ["000001.SZ", "600519.SH"]
    assert "399006.SZ" in symbols
    assert "000300.SH" in symbols
    assert "000852.SH" in symbols


def test_full_daily_download_universe_keeps_user_selected_etfs_and_indexes() -> None:
    symbols = _full_daily_download_universe(
        ["000001.SZ"],
        include_indexes=False,
        extra_symbols="",
        etf_symbols=["510300.SH"],
        index_symbols=["399006.SZ"],
    )

    assert symbols == ["000001.SZ", "510300.SH", "399006.SZ"]


def test_tdx_download_selection_counts_separates_categories() -> None:
    counts = _tdx_download_selection_counts(
        pd.DataFrame(
            [
                {"symbol": "000001.SZ", "category": "stock"},
                {"symbol": "510300.SH", "category": "etf"},
                {"symbol": "399006.SZ", "category": "index"},
                {"symbol": "880001.SH", "category": "index"},
            ]
        ),
        ["000001.SZ", "399006.SZ"],
    )

    assert counts == {"stock": 1, "etf": 0, "index": 1, "other": 0}


def test_symbol_data_hint_points_to_existing_alternate_suffix(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame({"date": ["2024-01-02"], "stock_code": ["000300.SH"]}).to_parquet(qfq / "000300.SH.parquet")

    hint = _symbol_data_hint(
        "000300",
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
    )

    assert "000300.SZ" in hint
    assert "000300.SH" in hint
