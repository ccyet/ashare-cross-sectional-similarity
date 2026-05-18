from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_cross_section_similarity.cli import _parse_args
from ashare_cross_section_similarity.cli import _resolve_download_symbols


def test_parse_download_command() -> None:
    args = _parse_args(
        [
            "download",
            "--symbols",
            "000001.SZ,600519.SH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--trend-repo",
            "/tmp/trend-backtest",
            "--provider",
            "tdx",
        ]
    )

    assert args.command == "download"
    assert args.symbols == "000001.SZ,600519.SH"
    assert args.timeframe == "1d"
    assert args.download_engine == "trend"
    assert args.trend_repo == "/tmp/trend-backtest"
    assert args.provider == "tdx"


def test_parse_search_date_tolerance_defaults_to_zero() -> None:
    args = _parse_args(
        [
            "search",
            "--target-symbol",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert args.command == "search"
    assert args.date_tolerance_bars == 0


def test_parse_search_date_tolerance_argument() -> None:
    args = _parse_args(
        [
            "search",
            "--target-symbol",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--date-tolerance-bars",
            "5",
        ]
    )

    assert args.date_tolerance_bars == 5


def test_parse_openbb_download_command() -> None:
    args = _parse_args(
        [
            "download",
            "--download-engine",
            "openbb",
            "--data-root",
            "/tmp/market/daily",
            "--symbols",
            "600519.SH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--provider",
            "akshare",
        ]
    )

    assert args.command == "download"
    assert args.download_engine == "openbb"
    assert args.data_root == "/tmp/market/daily"
    assert args.provider == "akshare"


def test_parse_tdx_download_command() -> None:
    args = _parse_args(
        [
            "download",
            "--download-engine",
            "tdx",
            "--data-root",
            "/tmp/market/daily",
            "--symbols",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--provider",
            "/Applications/Tdx/PYPlugins/user",
        ]
    )

    assert args.command == "download"
    assert args.download_engine == "tdx"
    assert args.provider == "/Applications/Tdx/PYPlugins/user"


def test_parse_check_command() -> None:
    args = _parse_args(
        [
            "check",
            "--symbols",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert args.command == "check"
    assert args.symbols == "000001.SZ"


def test_parse_import_data_command() -> None:
    args = _parse_args(
        [
            "import-data",
            "--input",
            "/tmp/prices.csv",
            "--data-root",
            "/tmp/market/daily",
            "--fallback-symbol",
            "000001.SZ",
        ]
    )

    assert args.command == "import-data"
    assert args.input == "/tmp/prices.csv"
    assert args.data_root == "/tmp/market/daily"
    assert args.fallback_symbol == "000001.SZ"


def test_parse_search_command() -> None:
    args = _parse_args(
        [
            "search",
            "--target-symbol",
            "300750.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--universe-symbols",
            "000001.SZ",
        ]
    )

    assert args.command == "search"
    assert args.target_symbol == "300750.SZ"


def test_parse_history_command() -> None:
    args = _parse_args(
        [
            "history",
            "--symbol",
            "300750.SZ",
            "--as-of",
            "2024-03-31",
            "--window-size",
            "20",
        ]
    )

    assert args.command == "history"
    assert args.symbol == "300750.SZ"
    assert args.window_size == 20


def test_download_symbols_do_not_fallback_to_full_local_universe(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "stock_code": ["600519.SH"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
            "amount": [1],
        }
    ).to_parquet(qfq / "600519.SH.parquet", index=False)
    args = _parse_args(
        [
            "check",
            "--data-root",
            str(tmp_path / "market" / "daily"),
            "--symbols",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert _resolve_download_symbols(args) == ["000001.SZ"]
