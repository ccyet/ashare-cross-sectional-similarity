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
        ]
    )

    assert args.command == "download"
    assert args.symbols == "000001.SZ,600519.SH"
    assert args.timeframe == "1d"


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
