from __future__ import annotations

import pandas as pd
import pytest

from ashare_cross_section_similarity.tdx_source import fetch_tdx_bars, fetch_tdx_stock_symbols


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

    assert fake.initialize_calls
    assert fake.stock_list_calls == 1
    assert symbols == ["000001.SZ", "600519.SH", "688603.SH", "830799.BJ"]


def test_fetch_tdx_stock_symbols_reports_missing_stock_list_api() -> None:
    fake = _FakeTq()

    with pytest.raises(RuntimeError, match="TDX 未能获取股票清单"):
        fetch_tdx_stock_symbols(tq_client=fake)
