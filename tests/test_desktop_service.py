from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from ashare_cross_section_similarity import universe as universe_module
from ashare_cross_section_similarity.desktop import service as service_module
from ashare_cross_section_similarity.desktop.service import (
    CrossSectionRequest,
    DataApiBarsResult,
    DataCoverageRequest,
    DataUpdateRequest,
    DesktopAppConfig,
    DesktopSearchService,
    FullDailyUpdateRequest,
    HistoryRequest,
    MissingDataUpdateRequest,
    PriceImportRequest,
    ReviewRequest,
    export_frame_csv,
    import_symbol_file,
    parse_symbol_list,
)
from ashare_cross_section_similarity.data import update_symbol_name_map


def _write_bars(root: Path, symbol: str, closes: list[float]) -> None:
    frame = _bars(symbol, closes)
    output = root / "qfq" / f"{symbol}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
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


def test_parse_symbol_list_normalizes_and_deduplicates() -> None:
    assert parse_symbol_list("000001, 000001.SZ\n600519.SH") == ("000001.SZ", "600519.SH")


def test_import_symbol_file_accepts_csv_and_excel(tmp_path: Path) -> None:
    csv_path = tmp_path / "symbols.csv"
    excel_path = tmp_path / "symbols.xlsx"
    frame = pd.DataFrame({"代码": ["000001", "600519.SH", "000001.SZ"]})
    frame.to_csv(csv_path, index=False)
    frame.to_excel(excel_path, index=False)

    assert import_symbol_file(csv_path) == ("000001.SZ", "600519.SH")
    assert import_symbol_file(excel_path) == ("000001.SZ", "600519.SH")


def test_desktop_service_imports_custom_price_file_with_fallback_symbol(tmp_path: Path) -> None:
    csv_path = tmp_path / "upload.csv"
    csv_path.write_text(
        "\n".join(
            [
                "date,open,high,low,close,volume",
                "2024-01-01,10,11,9,10.5,100",
                "2024-01-02,10.5,12,10,11.5,120",
            ]
        ),
        encoding="utf-8",
    )
    service = DesktopSearchService(DesktopAppConfig(data_root=tmp_path / "market" / "daily", timeframe="1d", adjust="qfq"))

    status = service.import_price_file(PriceImportRequest(file_path=csv_path, fallback_symbol="000001"))

    assert status["symbol"].tolist() == ["000001.SZ"]
    assert status["status"].tolist() == ["imported"]
    bars = service.load_bars(symbols=("000001.SZ",), start="2024-01-01", end="2024-01-02")
    assert bars["close"].tolist() == [10.5, 11.5]


def test_desktop_service_lists_symbols_and_runs_existing_history_search(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(
        data_root,
        "000001.SZ",
        [
            10,
            11,
            12,
            11,
            13,
            20,
            19,
            18,
            17,
            16,
            30,
            33,
            36,
            33,
            39,
            40,
            41,
            42,
        ],
    )
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    assert service.available_symbols() == ("000001.SZ",)

    result = service.search_history(
        HistoryRequest(
            symbol="000001",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(2,),
            top_n=1,
            exclusion_bars=0,
            nearby_gap_days=0,
        )
    )

    assert result.symbol == "000001.SZ"
    assert result.results.iloc[0]["窗口开始"] == pd.Timestamp("2024-01-01")


def test_desktop_service_runs_history_for_multiple_symbols(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "000001.SZ", [10, 11, 12, 11, 13, 20, 19, 18, 17, 16, 30, 33, 36, 33, 39])
    _write_bars(data_root, "600519.SH", [20, 21, 22, 21, 23, 25, 24, 23, 22, 21, 28, 30, 32, 31, 35])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    results = service.search_history_many(
        HistoryRequest(
            symbol="000001.SZ,600519.SH",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(1,),
            top_n=1,
            exclusion_bars=0,
            nearby_gap_days=0,
        )
    )

    assert [result.symbol for result in results] == ["000001.SZ", "600519.SH"]
    assert all(not result.results.empty for result in results)


def test_desktop_service_resolves_stock_names_for_all_desktop_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = DesktopSearchService(DesktopAppConfig(data_root=tmp_path / "market" / "daily"))
    bars = pd.DataFrame({"stock_code": ["000001.SZ"], "stock_name": ["平安银行"]})
    monkeypatch.setattr(service_module, "_stock_names_from_akshare", lambda symbols: {"300750.SZ": "宁德时代"})

    names = service.resolve_stock_names(("300750.SZ", "000001.SZ", "510300.SH"), bars=bars)

    assert names == {
        "300750.SZ": "宁德时代",
        "000001.SZ": "平安银行",
        "510300.SH": "沪深300ETF",
    }


def test_desktop_service_resolves_local_sector_index_names(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    update_symbol_name_map(data_root, {"880081.SH": "通达信趋势"}, source="tdx_sector_index")
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    names = service.resolve_stock_names(("880081.SH",))

    assert names == {"880081.SH": "通达信趋势"}


def test_desktop_service_selects_latest_quick_window(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "000001.SZ", [10, 11, 12, 13, 14])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    window = service.quick_window("000001", window_size=3)

    assert window.symbol == "000001.SZ"
    assert window.start == "2024-01-03"
    assert window.end == "2024-01-05"
    assert window.rows == 3
    assert "近 3 根K线" in window.message


def test_desktop_service_selects_latest_local_close(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "000001.SZ", [10, 11, 12])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    latest = service.latest_close("000001")

    assert latest.symbol == "000001.SZ"
    assert latest.start == "2024-01-03"
    assert latest.end == "2024-01-03"
    assert latest.rows == 3
    assert "最新本地收盘日：2024-01-03" in latest.message


def test_desktop_service_history_request_accepts_custom_window_start(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "000001.SZ", [10, 11, 12, 13, 14, 20, 19, 18, 17, 16, 30, 33, 36, 33, 39])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    result = service.search_history(
        HistoryRequest(
            symbol="000001.SZ",
            window_start="2024-01-11",
            as_of="2024-01-15",
            window_size=2,
            forward_windows=(1,),
            top_n=1,
            exclusion_bars=0,
            nearby_gap_days=0,
        )
    )

    assert result.current_window["date"].min() == pd.Timestamp("2024-01-11")
    assert result.window_size == 5


def test_desktop_service_exports_frame_as_utf8_sig_csv(tmp_path: Path) -> None:
    output = tmp_path / "result.csv"
    frame = pd.DataFrame({"代码": ["000001.SZ"], "综合相似度": [0.98]})

    exported = export_frame_csv(frame, output)

    assert exported == output
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "000001.SZ" in output.read_text(encoding="utf-8-sig")


def test_desktop_service_runs_existing_cross_section_and_review(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "300750.SZ", [10, 11, 12, 13, 14, 15, 16])
    _write_bars(data_root, "000001.SZ", [20, 22, 24, 26, 28, 30, 32])
    _write_bars(data_root, "600519.SH", [30, 29, 28, 27, 26, 25, 24])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    cross_section = service.search_cross_section(
        CrossSectionRequest(
            target_symbol="300750.SZ",
            universe_symbols=("000001.SZ", "600519.SH"),
            start="2024-01-01",
            end="2024-01-05",
            top_n=1,
        )
    )
    review = service.analyze_review(
        ReviewRequest(symbol="300750.SZ", start="2024-01-01", end="2024-01-05")
    )

    assert cross_section.target_symbol == "300750.SZ"
    assert cross_section.results.iloc[0]["symbol"] == "000001.SZ"
    assert review.symbol == "300750.SZ"
    assert review.overview["return"] > 0


def test_desktop_service_checks_cross_section_coverage_with_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        calls.append(dict(kwargs))
        return pd.DataFrame({"symbol": kwargs["symbols"], "status": ["available", "missing_file"]})

    monkeypatch.setattr(service_module, "data_check", fake_data_check)
    service = DesktopSearchService(DesktopAppConfig(data_root=tmp_path / "market" / "daily"))

    result = service.cross_section_coverage(
        CrossSectionRequest(
            target_symbol="300750",
            universe_symbols=("000001.SZ",),
            start="2024-01-10",
            end="2024-01-20",
            date_tolerance_bars=5,
        )
    )

    assert result["symbol"].tolist() == ["300750.SZ", "000001.SZ"]
    assert calls[0]["symbols"] == ("300750.SZ", "000001.SZ")
    assert calls[0]["start"] == service_module._cross_section_load_start("2024-01-10", 5)
    assert calls[0]["end"] == service_module._cross_section_load_end("2024-01-20", 5)


def test_full_daily_update_honors_pause_check_between_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = DesktopSearchService(DesktopAppConfig(data_root=tmp_path / "market" / "daily"))
    download_calls: list[tuple[str, ...]] = []
    progress_events: list[dict[str, object]] = []
    pause_states = iter([False, True, True, False])
    sleeps: list[float] = []

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        symbols = tuple(kwargs["symbols"])
        download_calls.append(symbols)
        return pd.DataFrame({"symbol": list(symbols), "status": ["success"] * len(symbols)})

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        symbols = tuple(kwargs["symbols"])
        return pd.DataFrame({"symbol": list(symbols), "status": ["missing_file"] * len(symbols)})

    def pause_check() -> bool:
        return next(pause_states, False)

    monkeypatch.setattr(service_module, "_full_daily_stock_symbols", lambda engine, provider: ["000001.SZ", "600519.SH"])
    monkeypatch.setattr(service_module, "update_local_bars", fake_update_local_bars)
    monkeypatch.setattr(service_module, "data_check", fake_data_check)

    service.run_full_daily_update(
        FullDailyUpdateRequest(start="2024-01-01", end="2024-01-02", batch_size=1, include_indexes=False),
        progress_callback=progress_events.append,
        pause_check=pause_check,
        pause_sleep=sleeps.append,
    )

    assert download_calls == [("000001.SZ",), ("600519.SH",)]
    assert sleeps == [0.25, 0.25]
    assert any(event.get("paused") is True for event in progress_events)


def test_desktop_service_runs_cross_section_for_multiple_targets(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "300750.SZ", [10, 11, 12, 13, 14, 15, 16])
    _write_bars(data_root, "600519.SH", [30, 29, 28, 27, 26, 25, 24])
    _write_bars(data_root, "000001.SZ", [20, 22, 24, 26, 28, 30, 32])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    results = service.search_cross_section_many(
        CrossSectionRequest(
            target_symbol="300750.SZ,600519.SH",
            universe_symbols=("000001.SZ",),
            start="2024-01-01",
            end="2024-01-05",
            top_n=1,
        )
    )

    assert [result.target_symbol for result in results] == ["300750.SZ", "600519.SH"]
    assert all(result.results.iloc[0]["symbol"] == "000001.SZ" for result in results)


def test_desktop_service_runs_review_for_multiple_symbols_and_rankings(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "300750.SZ", [10, 11, 12, 13, 14, 15, 16])
    _write_bars(data_root, "600519.SH", [30, 29, 28, 27, 26, 25, 24])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    results = service.analyze_review_many(
        ReviewRequest(symbol="300750.SZ,600519.SH", start="2024-01-01", end="2024-01-05")
    )

    assert [result.symbol for result in results] == ["300750.SZ", "600519.SH"]
    assert [result.overview["return"] > 0 for result in results] == [True, False]


def test_desktop_service_review_bundle_attaches_script_rankings(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "300750.SZ", [10, 11, 12, 13, 14, 15, 16])
    _write_bars(data_root, "600519.SH", [30, 29, 28, 27, 26, 25, 24])
    _write_bars(data_root, "000300.SH", [10, 10.1, 10.2, 10.4, 10.5, 10.6, 10.7])
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    bundle = service.analyze_review_bundle(
        ReviewRequest(
            symbol="300750.SZ,600519.SH",
            start="2024-01-01",
            end="2024-01-07",
            index_symbols=("000300.SH",),
        )
    )

    assert bundle.ranking_frame.iloc[0]["代码"] == "300750.SZ"
    assert bundle.script_profiles[0]["代码"] == "300750.SZ"
    assert bundle.script_profiles[0]["强弱等级"]
    assert bundle.script_profiles[0]["锐评结论"]


def test_desktop_service_review_bundle_resolves_stock_names_and_direction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "300750.SZ", [10, 11, 12, 13, 14, 15, 16])
    _write_bars(data_root, "600519.SH", [30, 29, 28, 27, 26, 25, 24])
    fake_akshare = SimpleNamespace(
        stock_info_a_code_name=lambda: pd.DataFrame(
            {
                "code": ["300750", "600519"],
                "name": ["宁德时代", "贵州茅台"],
            }
        )
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    bundle = service.analyze_review_bundle(
        ReviewRequest(
            symbol="300750.SZ,600519.SH",
            start="2024-01-01",
            end="2024-01-07",
            industry_name="新能源",
            concept_name="锂电",
        )
    )

    assert {symbol: bundle.stock_names[symbol] for symbol in ("300750.SZ", "600519.SH")} == {
        "300750.SZ": "宁德时代",
        "600519.SH": "贵州茅台",
    }
    assert set(bundle.ranking_frame["股票"]) == {"宁德时代", "贵州茅台"}
    assert set(bundle.ranking_frame["所属方向"]) == {"行业:新能源 / 概念:锂电"}


def test_desktop_service_checks_coverage_and_plans_kline_migration(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "000001.SZ", [10, 11, 12])
    destination = tmp_path / "archive"
    service = DesktopSearchService(DesktopAppConfig(data_root=data_root))

    coverage = service.check_coverage(DataCoverageRequest(symbols=("000001",), start="2024-01-01", end="2024-01-03"))
    plan = service.plan_kline_migration(data_root / "qfq", destination)

    assert coverage.iloc[0]["symbol"] == "000001.SZ"
    assert coverage.iloc[0]["status"] == "available"
    assert plan.iloc[0]["destination"] == str(destination / "000001.SZ.parquet")


def test_desktop_service_updates_local_bars_with_configured_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "status": "success",
                    "rows": 3,
                    "new_rows": 3,
                    "message": "ok",
                }
            ]
        )

    monkeypatch.setattr(service_module, "update_local_bars", fake_update_local_bars)
    service = DesktopSearchService(
        DesktopAppConfig(
            data_root=tmp_path / "market" / "daily",
            timeframe="1d",
            adjust="qfq",
            tdx_path="/Applications/Tdx/PYPlugins/user",
        )
    )

    result = service.update_bars(
        DataUpdateRequest(
            symbols=("000001",),
            start="2024-01-01",
            end="2024-01-03",
            download_engine="tdx",
        )
    )

    assert result.iloc[0]["status"] == "success"
    assert calls[0]["symbols"] == ("000001.SZ",)
    assert calls[0]["download_engine"] == "tdx"
    assert calls[0]["provider"] == "/Applications/Tdx/PYPlugins/user"
    assert calls[0]["data_root"] == tmp_path / "market" / "daily"


def test_desktop_service_plans_full_daily_update_with_tdx_and_skips_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(service_module, "fetch_tdx_stock_symbols", lambda tqcenter_path="": ["000001.SZ", "600519.SH"])

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        symbols = list(kwargs["symbols"])
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "status": "available" if symbol == "000001.SZ" else "missing_file",
                    "rows": 10 if symbol == "000001.SZ" else 0,
                }
                for symbol in symbols
            ]
        )

    monkeypatch.setattr(service_module, "data_check", fake_data_check)
    service = DesktopSearchService(DesktopAppConfig(data_root=tmp_path / "market" / "daily", adjust="qfq"))

    plan = service.plan_full_daily_update(
        FullDailyUpdateRequest(
            start="2024-01-01",
            end="2024-01-31",
            provider="/Applications/Tdx/PYPlugins/user",
            include_indexes=True,
            extra_symbols="159915.SZ",
            skip_available=True,
        )
    )

    assert plan.total_count == 9
    assert plan.stock_count == 2
    assert plan.index_count == 6
    assert "000001.SZ" not in plan.download_symbols
    assert "600519.SH" in plan.download_symbols
    assert "159915.SZ" in plan.download_symbols
    assert plan.coverage.iloc[0]["symbol"] == "000001.SZ"


def test_desktop_service_runs_full_daily_update_in_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(service_module, "fetch_tdx_stock_symbols", lambda tqcenter_path="": ["000001.SZ", "600519.SH", "300750.SZ"])

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        symbols = tuple(kwargs["symbols"])
        calls.append(symbols)
        return pd.DataFrame({"symbol": symbols, "status": ["success"] * len(symbols), "rows": [1] * len(symbols)})

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        symbols = tuple(kwargs["symbols"])
        return pd.DataFrame({"symbol": symbols, "status": ["available"] * len(symbols), "rows": [1] * len(symbols)})

    monkeypatch.setattr(service_module, "update_local_bars", fake_update_local_bars)
    monkeypatch.setattr(service_module, "data_check", fake_data_check)
    service = DesktopSearchService(DesktopAppConfig(data_root=tmp_path / "market" / "daily", adjust="qfq"))
    progress_events: list[dict[str, object]] = []

    result = service.run_full_daily_update(
        FullDailyUpdateRequest(
            start="2024-01-01",
            end="2024-01-31",
            provider="/Applications/Tdx/PYPlugins/user",
            include_indexes=False,
            skip_available=False,
            batch_size=2,
        ),
        progress_callback=progress_events.append,
    )

    assert calls == [("000001.SZ", "600519.SH"), ("300750.SZ",)]
    assert result["status"].tolist() == ["available", "available", "available"]
    assert progress_events == [
        {"completed": 2, "total": 3, "batch_index": 1, "batch_count": 2, "current": "000001.SZ, 600519.SH"},
        {"completed": 3, "total": 3, "batch_index": 2, "batch_count": 2, "current": "300750.SZ"},
    ]


def test_desktop_service_updates_only_missing_checked_symbols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        symbols = tuple(kwargs["symbols"])
        if symbols == ("000001.SZ", "600519.SH"):
            return pd.DataFrame(
                {
                    "symbol": symbols,
                    "status": ["available", "missing_file"],
                    "rows": [10, 0],
                }
            )
        return pd.DataFrame({"symbol": symbols, "status": ["available"] * len(symbols), "rows": [3] * len(symbols)})

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        symbols = tuple(kwargs["symbols"])
        calls.append(symbols)
        return pd.DataFrame({"symbol": symbols, "status": ["success"] * len(symbols), "rows": [3] * len(symbols)})

    monkeypatch.setattr(service_module, "data_check", fake_data_check)
    monkeypatch.setattr(service_module, "update_local_bars", fake_update_local_bars)
    service = DesktopSearchService(DesktopAppConfig(data_root=tmp_path / "market" / "daily", adjust="qfq"))

    result = service.update_missing_bars(
        MissingDataUpdateRequest(
            symbols=("000001.SZ", "600519.SH"),
            start="2024-01-01",
            end="2024-01-31",
            download_engine="akshare",
        )
    )

    assert calls == [("600519.SH",)]
    assert result["symbol"].tolist() == ["600519.SH"]
    assert result["status"].tolist() == ["available"]


def test_desktop_service_resolves_review_auto_etf_proxies_from_fallback() -> None:
    service = DesktopSearchService()

    symbols, names, matches, warning = service.resolve_review_auto_etf_proxies(industry_name="半导体", concept_name="")

    assert warning == ""
    assert symbols == ("159995.SZ",)
    assert names["159995.SZ"] == "芯片ETF"
    assert matches.iloc[0]["query"] == "半导体"


def test_desktop_service_lists_popular_review_etfs_by_amount_and_theme() -> None:
    service = DesktopSearchService()
    index = pd.DataFrame(
        {
            "symbol": ["512480.SH", "159995.SZ", "510300.SH", "510310.SH"],
            "name": ["半导体ETF", "芯片ETF", "沪深300ETF", "300ETF"],
            "amount": [7_000_000.0, 7_500_000.0, 10_000_000.0, 9_000_000.0],
            "category": ["半导体", "半导体", "沪深300", "沪深300"],
        }
    )

    result = service.popular_review_etfs(etf_index=index, limit=10)

    assert result["symbol"].tolist() == ["510300.SH", "159995.SZ"]
    assert result["name"].tolist() == ["沪深300ETF", "芯片ETF"]


def test_desktop_service_resolves_universe_from_manual_and_akshare_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(universe_module, "fetch_index_constituents", lambda value: ["000300", "000001.SZ"])
    monkeypatch.setattr(universe_module, "fetch_industry_constituents", lambda value: ["600519"])
    monkeypatch.setattr(universe_module, "fetch_concept_constituents", lambda value: ["300750"])
    service = DesktopSearchService()

    symbols = service.resolve_universe_symbols(
        symbols="000001",
        index_code="000300",
        industry="白酒",
        concept="锂电池",
    )

    assert symbols == ("000001.SZ", "000300.SZ", "600519.SH", "300750.SZ")


class _FakeDataApiClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, object]] = []

    def fetch_bars(
        self,
        *,
        symbols: tuple[str, ...],
        start: str,
        end: str,
        source: str,
        timeframe: str,
        adjust: str,
        tdx_path: str = "",
    ) -> DataApiBarsResult:
        self.calls.append(
            {
                "symbols": symbols,
                "start": start,
                "end": end,
                "source": source,
                "timeframe": timeframe,
                "adjust": adjust,
                "tdx_path": tdx_path,
            }
        )
        selected = self.frame.loc[self.frame["stock_code"].isin(symbols)].copy()
        return DataApiBarsResult(bars=selected)


def test_desktop_service_uses_data_api_for_history_search() -> None:
    bars = _bars(
        "000001.SZ",
        [
            10,
            11,
            12,
            11,
            13,
            20,
            19,
            18,
            17,
            16,
            30,
            33,
            36,
            33,
            39,
        ],
    )
    client = _FakeDataApiClient(bars)
    service = DesktopSearchService(
        DesktopAppConfig(data_mode="api", data_source="akshare", api_base_url="http://127.0.0.1:8765"),
        data_api_client=client,
    )

    result = service.search_history(
        HistoryRequest(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(1,),
            top_n=1,
            exclusion_bars=0,
            nearby_gap_days=0,
        )
    )

    assert client.calls[0]["source"] == "akshare"
    assert client.calls[0]["symbols"] == ("000001.SZ",)
    assert client.calls[0]["end"] == "2024-01-15"
    assert result.results.iloc[0]["窗口开始"] == pd.Timestamp("2024-01-01")


def test_desktop_service_passes_tdx_path_to_data_api_for_cross_section() -> None:
    bars = pd.concat(
        [
            _bars("300750.SZ", [10, 11, 12, 13, 14, 15, 16]),
            _bars("000001.SZ", [20, 22, 24, 26, 28, 30, 32]),
        ],
        ignore_index=True,
    )
    client = _FakeDataApiClient(bars)
    service = DesktopSearchService(
        DesktopAppConfig(
            data_mode="api",
            data_source="tdx",
            api_base_url="http://127.0.0.1:8765",
            tdx_path="/Applications/Tdx/PYPlugins/user",
        ),
        data_api_client=client,
    )

    result = service.search_cross_section(
        CrossSectionRequest(
            target_symbol="300750.SZ",
            universe_symbols=("000001.SZ",),
            start="2024-01-01",
            end="2024-01-05",
            top_n=1,
        )
    )

    assert client.calls[0]["source"] == "tdx"
    assert client.calls[0]["tdx_path"] == "/Applications/Tdx/PYPlugins/user"
    assert client.calls[0]["symbols"] == ("300750.SZ", "000001.SZ")
    assert result.results.iloc[0]["symbol"] == "000001.SZ"


def test_desktop_service_rejects_unknown_data_mode() -> None:
    service = DesktopSearchService(DesktopAppConfig(data_mode="remote"))

    with pytest.raises(ValueError, match="data_mode"):
        service.analyze_review(ReviewRequest(symbol="000001.SZ", start="2024-01-01", end="2024-01-02"))
