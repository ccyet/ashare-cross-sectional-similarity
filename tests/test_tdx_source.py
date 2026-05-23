from __future__ import annotations

import sys

import pandas as pd
import pytest

from ashare_cross_section_similarity.tdx_source import (
    build_tdx_etf_index,
    fetch_tdx_bars,
    fetch_tdx_etf_index,
    fetch_tdx_kline_symbol_table,
    search_tdx_etf_index,
    fetch_tdx_kline_symbols,
    fetch_tdx_stock_symbols,
)
from ashare_cross_section_similarity import tdx_source


class _FakeTq:
    def __init__(self, payload: dict[str, pd.DataFrame] | None = None) -> None:
        self.payload = payload or {}
        self.initialize_calls: list[str] = []
        self.market_calls: list[dict[str, object]] = []

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)

    def get_market_data(self, **kwargs: object) -> dict[str, pd.DataFrame]:
        self.market_calls.append(kwargs)
        return self.payload


class _FakeTqStockList:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.initialize_calls: list[str] = []
        self.stock_list_calls = 0

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)

    def get_stock_list(self) -> object:
        self.stock_list_calls += 1
        return self.payload


class _FakeTqMarketLists:
    def __init__(self) -> None:
        self.initialize_calls: list[str] = []
        self.calls: list[tuple[object, ...]] = []

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)

    def get_stock_list(self, *args: object, **kwargs: object) -> pd.DataFrame:
        market = kwargs.get("market", args[0] if args else "")
        self.calls.append(args)
        if market == "SH":
            return pd.DataFrame({"code": ["510300", "880001"], "market": ["SH", "SH"]})
        if market == "SZ":
            return pd.DataFrame({"code": ["399006", "159915"], "market": ["SZ", "SZ"]})
        if market == "BJ":
            return pd.DataFrame({"code": ["830799"], "market": ["BJ"]})
        return pd.DataFrame({"code": ["000001"], "market": ["SZ"]})


class _FakeTqInitializeFail:
    def initialize(self, caller_path: str) -> None:
        raise RuntimeError("terminal not ready")


class _FakeTqStockListInitializeFail(_FakeTqStockList):
    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)
        raise RuntimeError("terminal not ready")


class _FakeTqStockListRequiresInitialize(_FakeTqStockList):
    def __init__(self, payload: object) -> None:
        super().__init__(payload)
        self.initialized = False

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)
        self.initialized = True

    def get_stock_list(self) -> object:
        self.stock_list_calls += 1
        if not self.initialized:
            raise RuntimeError("TQ数据接口初始化失败")
        return self.payload


class _FakeTqOfficialMarketCodes:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []
        self.initialize_calls: list[str] = []

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)

    def get_stock_list(self, *args: object, **kwargs: object) -> object:
        market = kwargs.get("market", args[0] if args else "")
        list_type = kwargs.get("list_type", "")
        self.calls.append((market, list_type))
        if isinstance(market, int):
            raise AttributeError("'int' object has no attribute 'encode'")
        if market == "5":
            return ["000001.SZ", "600519.SH"]
        if market == "91" and list_type == "1":
            return pd.DataFrame(
                {
                    "code": ["512480"],
                    "market": ["SH"],
                    "name": ["半导体ETF"],
                }
            )
        return []


class _FakeTqOfficialMarketCodesRequiresInitialize(_FakeTqOfficialMarketCodes):
    def __init__(self) -> None:
        super().__init__()
        self.initialized = False

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)
        self.initialized = True

    def get_stock_list(self, *args: object, **kwargs: object) -> object:
        if not self.initialized:
            raise RuntimeError("serverreturnnone")
        return super().get_stock_list(*args, **kwargs)


def _reset_tq_import_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tdx_source, "_TQ_CLIENT", None)
    monkeypatch.setattr(tdx_source, "_TQ_CLIENT_IMPORT_KEY", None)
    monkeypatch.setattr(tdx_source, "_TQ_CLIENT_SYS_PATHS", set())
    monkeypatch.setattr(tdx_source, "_INITIALIZED", False)
    monkeypatch.setattr(tdx_source, "_INITIALIZED_CLIENT_ID", None)
    sys.modules.pop("tqcenter", None)


def test_fetch_tdx_bars_normalizes_official_tqcenter_payload() -> None:
    index = pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:31:00"])
    fake = _FakeTq(
        {
            "Open": pd.DataFrame({"000001.SZ": [1.0, 1.1]}, index=index),
            "High": pd.DataFrame({"000001.SZ": [1.2, 1.3]}, index=index),
            "Low": pd.DataFrame({"000001.SZ": [0.9, 1.0]}, index=index),
            "Close": pd.DataFrame({"000001.SZ": [1.15, None]}, index=index),
            "Volume": pd.DataFrame({"000001.SZ": [100.0, 120.0]}, index=index),
            "Amount": pd.DataFrame({"000001.SZ": [1000.0, 1220.0]}, index=index),
        }
    )

    out = fetch_tdx_bars(
        symbols=("000001.SZ",),
        start="2024-01-02 09:30:00",
        end="2024-01-02 15:00:00",
        timeframe="1m",
        adjust="qfq",
        tq_client=fake,
    )

    assert fake.initialize_calls
    assert fake.market_calls[0]["field_list"] == ["Open", "High", "Low", "Close", "Volume", "Amount"]
    assert fake.market_calls[0]["stock_list"] == ["000001.SZ"]
    assert fake.market_calls[0]["period"] == "1m"
    assert fake.market_calls[0]["start_time"] == "20240102093000"
    assert fake.market_calls[0]["end_time"] == "20240102150000"
    assert fake.market_calls[0]["dividend_type"] == "front"
    assert out.to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-02 09:30:00"),
            "stock_code": "000001.SZ",
            "open": 1.0,
            "high": 1.2,
            "low": 0.9,
            "close": 1.15,
            "volume": 100.0,
            "amount": 1000.0,
        }
    ]


def test_fetch_tdx_bars_supports_daily_period_and_lowercase_fields() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    fake = _FakeTq(
        {
            "open": pd.DataFrame({"600519.SH": [10.0, 11.0]}, index=index),
            "high": pd.DataFrame({"600519.SH": [12.0, 13.0]}, index=index),
            "low": pd.DataFrame({"600519.SH": [9.0, 10.0]}, index=index),
            "close": pd.DataFrame({"600519.SH": [11.5, 12.5]}, index=index),
            "volume": pd.DataFrame({"600519.SH": [1000.0, 1200.0]}, index=index),
            "amount": pd.DataFrame({"600519.SH": [10000.0, 12200.0]}, index=index),
        }
    )

    out = fetch_tdx_bars(
        symbols=("600519.SH",),
        start="2024-01-01",
        end="2024-01-03",
        timeframe="1d",
        adjust="hfq",
        tq_client=fake,
    )

    assert fake.market_calls[0]["period"] == "1d"
    assert fake.market_calls[0]["dividend_type"] == "back"
    assert out["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert out["close"].tolist() == [11.5, 12.5]


def test_fetch_tdx_bars_requests_multiple_symbols_once() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    fake = _FakeTq(
        {
            "Open": pd.DataFrame({"000001.SZ": [1.0, 1.1], "600519.SH": [10.0, 11.0]}, index=index),
            "High": pd.DataFrame({"000001.SZ": [1.2, 1.3], "600519.SH": [12.0, 13.0]}, index=index),
            "Low": pd.DataFrame({"000001.SZ": [0.9, 1.0], "600519.SH": [9.0, 10.0]}, index=index),
            "Close": pd.DataFrame({"000001.SZ": [1.15, 1.25], "600519.SH": [11.5, 12.5]}, index=index),
            "Volume": pd.DataFrame({"000001.SZ": [100.0, 120.0], "600519.SH": [1000.0, 1200.0]}, index=index),
            "Amount": pd.DataFrame({"000001.SZ": [1000.0, 1220.0], "600519.SH": [10000.0, 12200.0]}, index=index),
        }
    )

    out = fetch_tdx_bars(
        symbols=("000001.SZ", "600519.SH"),
        start="2024-01-01",
        end="2024-01-03",
        timeframe="1d",
        adjust="qfq",
        tq_client=fake,
    )

    assert len(fake.market_calls) == 1
    assert fake.market_calls[0]["stock_list"] == ["000001.SZ", "600519.SH"]
    assert out["stock_code"].tolist() == ["000001.SZ", "000001.SZ", "600519.SH", "600519.SH"]
    assert out["close"].tolist() == [1.15, 1.25, 11.5, 12.5]


def test_fetch_tdx_bars_reports_unsupported_timeframe() -> None:
    with pytest.raises(ValueError, match="timeframe 仅支持"):
        fetch_tdx_bars(
            symbols=("000001.SZ",),
            start="2024-01-01",
            end="2024-01-02",
            timeframe="2m",
            tq_client=_FakeTq(),
        )


def test_fetch_tdx_bars_reports_missing_symbol_column() -> None:
    index = pd.to_datetime(["2024-01-02"])
    fake = _FakeTq(
        {
            "Open": pd.DataFrame({"600000.SH": [1.0]}, index=index),
            "High": pd.DataFrame({"600000.SH": [1.1]}, index=index),
            "Low": pd.DataFrame({"600000.SH": [0.9]}, index=index),
            "Close": pd.DataFrame({"600000.SH": [1.05]}, index=index),
            "Volume": pd.DataFrame({"600000.SH": [10.0]}, index=index),
            "Amount": pd.DataFrame({"600000.SH": [100.0]}, index=index),
        }
    )

    with pytest.raises(ValueError, match="missing symbol column"):
        fetch_tdx_bars(
            symbols=("000001.SZ",),
            start="2024-01-02",
            end="2024-01-02",
            timeframe="1d",
            tq_client=fake,
        )


def test_fetch_tdx_stock_symbols_reads_tdx_stock_list_and_filters_indexes() -> None:
    fake = _FakeTqStockList(
        pd.DataFrame(
            {
                "code": ["000001", "600519", "688603", "399006", "510300", "830799"],
                "name": ["平安银行", "贵州茅台", "天承科技", "创业板指", "沪深300ETF", "艾融软件"],
            }
        )
    )

    symbols = fetch_tdx_stock_symbols(tq_client=fake)

    assert fake.initialize_calls == []
    assert fake.stock_list_calls == 1
    assert symbols == ["000001.SZ", "600519.SH", "688603.SH", "830799.BJ"]


def test_fetch_tdx_kline_symbols_keeps_stocks_etfs_indexes_and_blocks() -> None:
    fake = _FakeTqStockList(
        pd.DataFrame(
            {
                "code": ["000001", "600519", "399006", "510300", "159915", "880001", "885001"],
                "market": ["SZ", "SH", "SZ", "SH", "SZ", "SH", "SH"],
                "name": ["平安银行", "贵州茅台", "创业板指", "沪深300ETF", "创业板ETF", "通达信行业", "通达信概念"],
            }
        )
    )

    symbols = fetch_tdx_kline_symbols(tq_client=fake)

    assert fake.initialize_calls == []
    assert fake.stock_list_calls == 1
    assert symbols == [
        "000001.SZ",
        "600519.SH",
        "399006.SZ",
        "510300.SH",
        "159915.SZ",
        "880001.SH",
        "885001.SH",
    ]


def test_fetch_tdx_kline_symbol_table_classifies_stocks_etfs_and_indexes() -> None:
    fake = _FakeTqStockList(
        pd.DataFrame(
            {
                "code": ["000001", "600519", "399006", "510300", "159915", "880001", "885001"],
                "market": ["SZ", "SH", "SZ", "SH", "SZ", "SH", "SH"],
                "name": ["平安银行", "贵州茅台", "创业板指", "沪深300ETF", "创业板ETF", "通达信行业", "通达信概念"],
            }
        )
    )

    table = fetch_tdx_kline_symbol_table(tq_client=fake)

    assert table.to_dict("records") == [
        {"symbol": "000001.SZ", "name": "平安银行", "category": "stock"},
        {"symbol": "600519.SH", "name": "贵州茅台", "category": "stock"},
        {"symbol": "399006.SZ", "name": "创业板指", "category": "index"},
        {"symbol": "510300.SH", "name": "沪深300ETF", "category": "etf"},
        {"symbol": "159915.SZ", "name": "创业板ETF", "category": "etf"},
        {"symbol": "880001.SH", "name": "通达信行业", "category": "index"},
        {"symbol": "885001.SH", "name": "通达信概念", "category": "index"},
    ]


def test_fetch_tdx_kline_symbol_table_does_not_require_market_connection() -> None:
    fake = _FakeTqStockListInitializeFail(
        pd.DataFrame(
            {
                "code": ["000001", "510300", "399006"],
                "market": ["SZ", "SH", "SZ"],
                "name": ["平安银行", "沪深300ETF", "创业板指"],
            }
        )
    )

    table = fetch_tdx_kline_symbol_table(tq_client=fake)

    assert fake.initialize_calls == []
    assert table["symbol"].tolist() == ["000001.SZ", "510300.SH", "399006.SZ"]


def test_fetch_tdx_kline_symbol_table_initializes_once_when_tdx_requires_it() -> None:
    fake = _FakeTqStockListRequiresInitialize(
        pd.DataFrame(
            {
                "code": ["000001", "510300", "399006"],
                "market": ["SZ", "SH", "SZ"],
                "name": ["平安银行", "沪深300ETF", "创业板指"],
            }
        )
    )

    table = fetch_tdx_kline_symbol_table(tq_client=fake)

    assert len(fake.initialize_calls) == 1
    assert fake.stock_list_calls == 2
    assert table["symbol"].tolist() == ["000001.SZ", "510300.SH", "399006.SZ"]


def test_fetch_tdx_etf_index_does_not_require_market_connection() -> None:
    fake = _FakeTqStockListInitializeFail(
        pd.DataFrame(
            {
                "code": ["512480", "159995", "600519"],
                "market": ["SH", "SZ", "SH"],
                "name": ["半导体ETF", "芯片ETF", "贵州茅台"],
                "amount": [1_000_000, 5_000_000, 9_000_000],
            }
        )
    )

    index = fetch_tdx_etf_index(tq_client=fake)

    assert fake.initialize_calls == []
    assert index["symbol"].tolist() == ["512480.SH", "159995.SZ"]


def test_load_tq_honors_changed_tqcenter_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_tq_import_state(monkeypatch)
    first_dir = tmp_path / "first" / "PYPlugins" / "user"
    second_dir = tmp_path / "second" / "PYPlugins" / "user"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first_dir.joinpath("tqcenter.py").write_text("class Tq:\n    marker = 'first'\ntq = Tq()\n", encoding="utf-8")
    second_dir.joinpath("tqcenter.py").write_text("class Tq:\n    marker = 'second'\ntq = Tq()\n", encoding="utf-8")

    try:
        first = tdx_source._load_tq(str(first_dir))
        second = tdx_source._load_tq(str(second_dir))
        assert str(first_dir) not in sys.path
        assert sys.path[0] == str(second_dir)
    finally:
        sys.modules.pop("tqcenter", None)
        for item in [str(first_dir), str(second_dir)]:
            if item in sys.path:
                sys.path.remove(item)

    assert first.marker == "first"
    assert second.marker == "second"
    assert first is not second


def test_fetch_tdx_kline_symbols_uses_market_hint_for_mapping_payload() -> None:
    fake = _FakeTqStockList({"code": ["000300", "000001"], "market": ["SH", "SZ"]})

    symbols = fetch_tdx_kline_symbols(tq_client=fake)

    assert symbols == ["000300.SH", "000001.SZ"]


def test_fetch_tdx_kline_symbols_collects_market_specific_lists() -> None:
    fake = _FakeTqMarketLists()

    symbols = fetch_tdx_kline_symbols(tq_client=fake)

    assert fake.initialize_calls == []
    assert "000001.SZ" in symbols
    assert "510300.SH" in symbols
    assert "880001.SH" in symbols
    assert "399006.SZ" in symbols
    assert "159915.SZ" in symbols
    assert "830799.BJ" in symbols


def test_fetch_tdx_kline_symbol_table_uses_tdx_string_market_codes() -> None:
    fake = _FakeTqOfficialMarketCodes()

    table = fetch_tdx_kline_symbol_table(tq_client=fake)

    assert table.to_dict("records") == [
        {"symbol": "000001.SZ", "name": "", "category": "stock"},
        {"symbol": "600519.SH", "name": "", "category": "stock"},
        {"symbol": "512480.SH", "name": "半导体ETF", "category": "etf"},
    ]
    assert all(not isinstance(market, int) for market, _list_type in fake.calls)
    assert ("5", "1") in fake.calls
    assert ("91", "1") in fake.calls


def test_fetch_tdx_kline_symbol_table_initializes_real_tdx_before_reading_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTqOfficialMarketCodesRequiresInitialize()
    monkeypatch.setattr(tdx_source, "_INITIALIZED", False)
    monkeypatch.setattr(tdx_source, "_INITIALIZED_CLIENT_ID", None)
    monkeypatch.setattr(tdx_source, "_load_tq", lambda _path="": fake)

    table = fetch_tdx_kline_symbol_table(tqcenter_path=r"F:\new_tdx64\PYPlugins")

    assert fake.initialize_calls
    assert table["symbol"].tolist() == ["000001.SZ", "600519.SH", "512480.SH"]


def test_candidate_import_paths_expands_windows_pyplugins_path() -> None:
    paths = tdx_source._candidate_import_paths(r"F:\new_tdx64\PYPlugins")

    assert [str(path) for path in paths] == [
        "F:/new_tdx64/PYPlugins/user",
        "F:/new_tdx64/PYPlugins",
    ]


def test_build_tdx_etf_index_keeps_largest_turnover_for_same_theme() -> None:
    index = build_tdx_etf_index(
        pd.DataFrame(
            {
                "code": ["512480", "159995", "510300", "600519"],
                "market": ["SH", "SZ", "SH", "SH"],
                "name": ["半导体ETF", "芯片ETF", "沪深300ETF", "贵州茅台"],
                "amount": [1_000_000, 5_000_000, 2_000_000, 9_000_000],
            }
        )
    )

    matches = search_tdx_etf_index(index, ["半导体", "芯片"])

    assert matches["query"].tolist() == ["半导体", "芯片"]
    assert matches["symbol"].tolist() == ["512480.SH", "159995.SZ"]
    assert matches["name"].tolist() == ["半导体ETF", "芯片ETF"]


def test_build_tdx_etf_index_preserves_duplicate_rows_and_names() -> None:
    index = build_tdx_etf_index(
        pd.DataFrame(
            {
                "code": ["512480", "512480", "159995", "600519"],
                "market": ["SH", "SH", "SZ", "SH"],
                "name": ["半导体ETF", "半导体ETF国联安", "芯片ETF", "贵州茅台"],
                "amount": [1_000_000, 2_000_000, 5_000_000, 9_000_000],
            }
        )
    )

    assert index["symbol"].tolist() == ["512480.SH", "512480.SH", "159995.SZ"]
    assert index["name"].tolist() == ["半导体ETF", "半导体ETF国联安", "芯片ETF"]
    assert index["amount"].tolist() == [1_000_000.0, 2_000_000.0, 5_000_000.0]


def test_search_tdx_etf_index_merges_same_query_to_largest_amount() -> None:
    index = build_tdx_etf_index(
        pd.DataFrame(
            {
                "code": ["512480", "159995", "588000"],
                "market": ["SH", "SZ", "SH"],
                "name": ["半导体ETF", "半导体芯片ETF", "科创50ETF"],
                "amount": [1_000_000, 5_000_000, 2_000_000],
            }
        )
    )

    matches = search_tdx_etf_index(index, ["半导体"])

    assert len(matches) == 1
    assert matches.iloc[0]["symbol"] == "159995.SZ"
    assert matches.iloc[0]["name"] == "半导体芯片ETF"


def test_fetch_tdx_stock_symbols_reports_missing_stock_list_api() -> None:
    fake = _FakeTq()

    with pytest.raises(RuntimeError, match="TDX 未能获取股票清单"):
        fetch_tdx_stock_symbols(tq_client=fake)


def test_tdx_initialize_error_preserves_root_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tdx_source, "_INITIALIZED", False)
    monkeypatch.setattr(tdx_source, "_INITIALIZED_CLIENT_ID", None)

    with pytest.raises(RuntimeError) as exc_info:
        tdx_source._ensure_initialized(_FakeTqInitializeFail())

    message = str(exc_info.value)
    assert "TDX 初始化失败" in message
    assert "根因: RuntimeError: terminal not ready" in message
